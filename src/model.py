import torch
import torch.nn as nn
import math

print(torch.__version__)

class InputEmbedding(nn.Module):
    def __init__(self, d_model : int, vocab_size : int):
        super(InputEmbedding, self).__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model : int, seq_length : int, dropout : float):
        super().__init__()
        self.d_model = d_model
        self.seq_length = seq_length
        self.dropout = nn.Dropout(dropout)
    
        pe = torch.zeros(seq_length, d_model)

        position = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe.unsqueeze(0) # pe.shape => (1, seq_length, d_model)

        self.register_buffer("pe", pe)

    def forward(self,x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad(False)

