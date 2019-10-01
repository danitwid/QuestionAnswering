import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from util import masked_softmax


class LSTMEncoder(nn.Module):
    """LSTM RNN Layer for encoding a sequence."""
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 drop_prob=0.):
        super(LSTMEncoder, self).__init__()
        self.drop_prob = drop_prob
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True,
                           bidirectional=True,
                           dropout=drop_prob if num_layers > 1 else 0.)

    def forward(self, x, lengths):
        # Save original padded length for use by pad_packed_sequence
        orig_len = x.size(1)

        # Sort by length and pack sequence for RNN
        lengths, sort_idx = lengths.sort(0, descending=True)
        x = x[sort_idx]     # (batch_size, seq_len, input_size)
        x = pack_padded_sequence(x, lengths, batch_first=True)

        self.rnn.flatten_parameters()
        # Apply RNN
        x, _ = self.rnn(x)  # (batch_size, seq_len, 2 * hidden_size)

        # Unpack and reverse sort
        x, _ = pad_packed_sequence(x, batch_first=True, total_length=orig_len)
        _, unsort_idx = sort_idx.sort(0)
        x = x[unsort_idx]   # (batch_size, seq_len, 2 * hidden_size)

        # Apply dropout (RNN applies dropout after all but the last layer)
        x = F.dropout(x, self.drop_prob, self.training)

        return x


class DotProductAttention(nn.Module):
    """A scaled dot-product attention layer over a sequence C w.r.t Q  (Context2Query attention):
       a_i = softmax((c_i'Q)*scale) for c_i in C.

       outputs context_hiddens and attn_output concatenation
    """
    def __init__(self, question_dim):
        super().__init__()

        self.scale = 1.0 / np.sqrt(question_dim)
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, context_hiddens, context_mask, question_hiddens, question_mask):
        # question_hiddens: [B,Q,H]
        # question_mask: [B,Q]
        # context_hiddens: [B,C,H]

        query = question_hiddens.permute(0, 2, 1)  # [B,Q,H] -> [B,H,Q]
        attention_score = torch.bmm(context_hiddens, query)  # [B,C,H]*[B,H,Q] = [B,C,Q]
        question_mask = question_mask.unsqueeze(1)  # [B,Q] -> [B,1,Q]
        attention_dist = masked_softmax(attention_score.mul_(self.scale), question_mask, dim=2)  # [B,C,Q]

        attn_output = torch.bmm(attention_dist, question_hiddens)  # [B,C,Q]*[B,Q,H] -> [B,C,H]
        # return context_hiddens and attn_output concatenation
        x = torch.cat([context_hiddens, attn_output], dim=2)  # [B,C,2*H]
        return x


class BilinearSeqAttn(nn.Module):
    """A scaled bilinear attention layer over a sequence C w.r.t Q (Context2Query attention):
       a_i = softmax((c_i'WQ)*scale) for c_i in C.

       outputs context_hiddens and attn_output concatenation
    """
    def __init__(self, context_size, question_size):
        super(BilinearSeqAttn, self).__init__()
        self.linear = nn.Linear(question_size, context_size)
        self.scale = 1.0 / np.sqrt(question_size)

    def forward(self, context_hiddens, context_mask, question_hiddens, question_mask):

        batch_size, output_len, dimensions = question_hiddens.size()
        query = question_hiddens.view(batch_size * output_len, dimensions)
        query = self.linear(query)
        query = query.view(batch_size, output_len, dimensions)

        query = query.permute(0, 2, 1)  # [B,Q,H] -> [B,H,Q]
        attention_score = torch.bmm(context_hiddens, query)  # [B,C,H]*[B,H,Q] = [B,C,Q]
        question_mask = question_mask.unsqueeze(1)  # [B,Q] -> [B,1,Q]
        attention_dist = masked_softmax(attention_score.mul_(self.scale), question_mask, dim=2)  # [B,C,Q]

        attn_output = torch.bmm(attention_dist, question_hiddens)  # [B,C,Q]*[B,Q,H] -> [B,C,H]
        # return context_hiddens and attn_output concatenation
        x = torch.cat([context_hiddens, attn_output], dim=2)  # [B,C,2*H]
        return x


class BiDAFAttention(nn.Module):
    """Bidirectional attention originally used by BiDAF.
    Based on the paper:
    "Bidirectional Attention Flow for Machine Comprehension"
    by Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi
    (https://arxiv.org/abs/1611.01603).

    Bidirectional attention computes attention in two directions:
    The context attends to the query and the query attends to the context.
    The output of this layer is the concatenation of [context, c2q_attention,
    context * c2q_attention, context * q2c_attention]. This concatenation allows
    the attention vector at each timestep, along with the embeddings from
    previous layers, to flow through the attention layer to the modeling layer.
    The output has shape (batch_size, context_len, 8 * hidden_size).

    Args:
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob=0.1):
        super(BiDAFAttention, self).__init__()
        self.drop_prob = drop_prob
        self.c_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.q_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.cq_weight = nn.Parameter(torch.zeros(1, 1, hidden_size))
        for weight in (self.c_weight, self.q_weight, self.cq_weight):
            nn.init.xavier_uniform_(weight)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, context_hiddens, context_mask, question_hiddens, question_mask):
        batch_size, c_len, _ = context_hiddens.size()
        q_len = question_hiddens.size(1)
        s = self.get_similarity_matrix(context_hiddens, question_hiddens)        # (batch_size, c_len, q_len)
        context_mask = context_mask.view(batch_size, c_len, 1)  # (batch_size, c_len, 1)
        question_mask = question_mask.view(batch_size, 1, q_len)  # (batch_size, 1, q_len)
        s1 = masked_softmax(s, question_mask, dim=2)       # (batch_size, c_len, q_len)
        s2 = masked_softmax(s, context_mask, dim=1)       # (batch_size, c_len, q_len)

        # (bs, c_len, q_len) x (bs, q_len, hid_size) => (bs, c_len, hid_size)
        a = torch.bmm(s1, question_hiddens)
        # (bs, c_len, c_len) x (bs, c_len, hid_size) => (bs, c_len, hid_size)
        b = torch.bmm(torch.bmm(s1, s2.transpose(1, 2)),context_hiddens)

        x = torch.cat([context_hiddens, a, context_hiddens * a, context_hiddens * b], dim=2)  # (bs, c_len, 4 * hid_size)

        return x

    def get_similarity_matrix(self, c, q):
        """Get the "similarity matrix" between context and query (using the
        terminology of the BiDAF paper).

        A naive implementation as described in BiDAF would concatenate the
        three vectors then project the result with a single weight matrix. This
        method is a more memory-efficient implementation of the same operation.

        See Also:
            Equation 1 in https://arxiv.org/abs/1611.01603
        """
        c_len, q_len = c.size(1), q.size(1)
        c = F.dropout(c, self.drop_prob, self.training)  # (bs, c_len, hid_size)
        q = F.dropout(q, self.drop_prob, self.training)  # (bs, q_len, hid_size)

        # Shapes: (batch_size, c_len, q_len)
        s0 = torch.matmul(c, self.c_weight).expand([-1, -1, q_len])
        s1 = torch.matmul(q, self.q_weight).transpose(1, 2)\
                                           .expand([-1, c_len, -1])
        s2 = torch.matmul(c * self.cq_weight, q.transpose(1, 2))
        s = s0 + s1 + s2 + self.bias

        return s


class SoftmaxOutputLayer(nn.Module):
    """Simplified Output layer for question answering.
    models the [context,attention] input vector through a bi-lstm and
    computes two linear transformations of the modeling
    output, then takes the softmax of the result, to get the start and end positions.
    """
    def __init__(self, hidden_size, drop_prob):
        super(SoftmaxOutputLayer, self).__init__()
        self.modeling = LSTMEncoder(input_size=4 * hidden_size,
                                    hidden_size=hidden_size,
                                    num_layers=2,
                                    drop_prob=drop_prob)
        self.att_linear_1 = nn.Linear(2 * hidden_size, 1)
        self.att_linear_2 = nn.Linear(2 * hidden_size, 1)

    def forward(self, att, mask, c_len):

        mod = self.modeling(att, c_len)  # (batch_size, c_len, 2 * hidden_size)
        # Shapes: (batch_size, seq_len, 1)
        logits_1 = self.att_linear_1(mod)
        logits_2 = self.att_linear_2(mod)

        # Shapes: (batch_size, seq_len)
        log_p1 = masked_softmax(logits_1.squeeze(), mask, log_softmax=True)
        log_p2 = masked_softmax(logits_2.squeeze(), mask, log_softmax=True)

        return log_p1, log_p2

