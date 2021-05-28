import torch
from torch.nn import *
from torch.nn import functional as F


def fourier_transform2d(x: torch.Tensor) -> torch.Tensor:
    """Applies 2d fourier transform

    Args:
        x (torch.Tensor):

    Returns:
        torch.Tensor: 
    """
    return torch.fft.fft2(x, dim=(-1, -2)).real


class DecoderLayer(Module):
    def __init__(self, d_model: int, dim_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.ff = Sequential(
            Linear(d_model, dim_ff),
            GELU(),
            Dropout(dropout),
            Linear(dim_ff, d_model),
            Dropout(dropout),
        )
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.norm2 = LayerNorm(d_model)

    def forward(self, x):
        x = x + self.dropout1(fourier_transform2d(x))
        x = self.norm1(x)
        x = x + self.ff(x)
        x = self.norm2(x)
        return x


class FNet(TransformerDecoder):
    """Apply Fast Fourier Transform instead of Transformer.
    Apply idea of using the input multiple times with FFT throughout the layers.
    """
    def __init__(
        self, num_layers:int, d_model: int, dim_ff: int = 2048, dropout: float = 0.1
    ):
        """Args:
            num_layers (int): self-descriptive
            d_model (int): dimensions of model
            dim_ff (int, optional): dimension of feedforward layer. Defaults to 2048.
            dropout (float, optional): Defaults to 0.1.
        """
        decoder_layer = DecoderLayer(d_model, dim_ff, dropout)
        super().__init__(decoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

