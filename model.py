import layers
import torch
import torch.nn as nn


class QA_Model(nn.Module):
    """
    Simplified Question Answering model for SQuAD v1.1
    Args:
        word_vectors: Pre-trained word vectors.
        hidden_size: Number of features in the hidden state at each layer.
        drop_prob: Dropout probability.
        attention_type:
            attention layer type:
            'DotProduct': layers.DotProductAttention
            'Bilinear': layers.BilinearSeqAttn
            'BiDAF': layers.BiDAFAttention
        train_embeddings: indicates whether to tune word vectors while training the model
    """
    def __init__(self, word_vectors, hidden_size, drop_prob=0., attention_type="BiDAF", train_embeddings=False):
        super(QA_Model, self).__init__()

        ATTENTION_TYPES = {'DotProduct': layers.DotProductAttention, 'Bilinear': layers.BilinearSeqAttn, 'BiDAF': layers.BiDAFAttention}
        self.attention_type = ATTENTION_TYPES[attention_type]
        self.embedding_size = word_vectors.shape[1]
        self.embedding = nn.Embedding(word_vectors.shape[0], self.embedding_size)
        self.embedding.weight.data.copy_(word_vectors)
        self.embedding.weight.requires_grad = train_embeddings

        self.encoder = layers.LSTMEncoder(input_size=self.embedding_size,
                                          hidden_size=hidden_size,
                                          num_layers=1,
                                          drop_prob=drop_prob)

        attention_output_size = hidden_size

        if self.attention_type == ATTENTION_TYPES['DotProduct']:
            self.att = layers.DotProductAttention(2 * hidden_size)
        elif self.attention_type == ATTENTION_TYPES['Bilinear']:
            self.att = layers.BilinearSeqAttn(2 * hidden_size, 2 * hidden_size)
        elif self.attention_type == ATTENTION_TYPES['BiDAF']:
            self.att = layers.BiDAFAttention(hidden_size=2 * hidden_size, drop_prob=drop_prob)
            attention_output_size *= 2  # BiDAFAttention output is larger

        self.out = layers.SoftmaxOutputLayer(hidden_size=attention_output_size, drop_prob=drop_prob)

    def forward(self, context_word_idxs, question_word_idxs):
        context_mask = torch.zeros_like(context_word_idxs) != context_word_idxs
        question_mask = torch.zeros_like(question_word_idxs) != question_word_idxs
        context_len, question_len = context_mask.sum(-1), question_mask.sum(-1)

        context_embedded = self.embedding(context_word_idxs)         # (batch_size, context_len, hidden_size)
        question_embedded = self.embedding(question_word_idxs)         # (batch_size, question_len, hidden_size)

        context_encoded = self.encoder(context_embedded, context_len)    # (batch_size, context_len, 2 * hidden_size)
        question_encoded = self.encoder(question_embedded, question_len)    # (batch_size, question_len, 2 * hidden_size)

        attention_output = self.att(context_encoded, context_mask, question_encoded, question_mask)
        output = self.out(attention_output, context_mask, context_len)  # 2 tensors of (batch_size, context_len)

        return output
