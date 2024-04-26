import torch
import torch.nn as nn

from cs336_basics.model.layers import TransformerBlock, RMSNorm, RoPEEmbedding
from cs336_systems.layers import RMSNormTriton

class Transformer(nn.Module):
    def __init__(self, vocab_size, context_length, num_layers, d_model, num_heads, d_ff, attn_pdrop, residual_pdrop, norm_type="rms"):
        super(Transformer, self).__init__()


        norm_function = RMSNorm

        if norm_type == "layer":
            norm_function == nn.LayerNorm
        elif norm_type == "rms_triton":
            norm_function == RMSNormTriton

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.attn_pdrop = attn_pdrop
        self.residual_pdrop = residual_pdrop
        self.positional_embeddings = nn.Embedding(context_length, d_model)
        self.norm_type = norm_type

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads, d_ff, attn_pdrop, residual_pdrop, norm_function)
                for _ in range(num_layers)
            ]
        )
        self.output_norm = norm_function(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)
        # self.softmax = nn.Softmax(dim=-1)

    def forward(self, x): # (batch_size, seq_len)
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)  # Generate positions tensor
        x = nn.functional.dropout(self.token_embedding(x) + self.positional_embeddings(positions), self.residual_pdrop)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.output_norm(x)
        x = self.output_proj(x)
        return x
    
class Optimus_Prime(nn.Module):
    def __init__(self, vocab_size, context_length, num_layers, d_model, num_heads, d_ff, attn_pdrop, residual_pdrop):
        super(Optimus_Prime, self).__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.attn_pdrop = attn_pdrop
        self.residual_pdrop = residual_pdrop
        self.positional_embeddings = RoPEEmbedding(d_model, context_length)

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads, d_ff, attn_pdrop, residual_pdrop)
                for _ in range(num_layers)
            ]
        )
        self.output_norm = RMSNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)
        self.output_proj.weight = self.token_embedding.weight
        # self.softmax = nn.Softmax(dim=-1)

    def forward(self, x): # (batch_size, seq_len)
        seq_len = x.size(1)
        x = nn.functional.dropout(self.token_embedding(x) + self.positional_embeddings(x), self.residual_pdrop)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.output_norm(x)
        x = self.output_proj(x)
        return x