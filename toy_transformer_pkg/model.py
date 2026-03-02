import torch
from torch import nn


class ToyTransformerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int = 100,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        num_classes: int = 2,
        max_seq_len: int = 32,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = nn.Parameter(
            torch.zeros(1, max_seq_len, d_model), requires_grad=True
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        input_ids: (batch_size, seq_len) of token indices.
        """
        x = self.embedding(input_ids) * (self.d_model**0.5)
        seq_len = x.size(1)
        x = x + self.positional_encoding[:, :seq_len, :]
        encoded = self.encoder(x)
        pooled = encoded.mean(dim=1)
        logits = self.classifier(pooled)
        return logits

