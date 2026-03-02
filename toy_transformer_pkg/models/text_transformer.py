"""Text transformer for sequence classification (accepts tokenized text input)."""
import math
from typing import Optional

import torch
from torch import nn


class TextTransformer(nn.Module):
    """
    Transformer encoder for text classification.
    Accepts input_ids and attention_mask (tokenized text).
    """

    def __init__(
        self,
        vocab_size: int = 1000,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        max_length: int = 64,
        num_classes: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_length, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        x = x + self.pos_embedding[:, :seq_len, :]
        src_key_padding_mask = None
        if attention_mask is not None:
            src_key_padding_mask = (attention_mask == 0)
        encoded = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        if attention_mask is not None:
            expanded = attention_mask.unsqueeze(-1).float()
            pooled = (encoded * expanded).sum(dim=1) / expanded.sum(dim=1).clamp(min=1)
        else:
            pooled = encoded[:, 0, :]
        logits = self.classifier(pooled)
        return logits
