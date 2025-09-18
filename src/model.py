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