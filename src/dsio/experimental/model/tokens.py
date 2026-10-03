"""Token-sequence encoders with explicit structural validity."""

from __future__ import annotations

import torch
from torch import Tensor, nn

_INTEGER_DTYPES = {
    torch.uint8,
    torch.uint16,
    torch.uint32,
    torch.uint64,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}


class EmbeddingEncoder(nn.Module):
    """Embed a token sequence and mean-pool only structurally valid positions.

    Consumes:
        An integer tensor ``[batch, tokens, 2]``. ``token_channel`` contains token IDs
        and ``validity_channel`` contains exactly zero or one. Validity, never a token
        value, determines padding.

    Produces:
        A tensor ``[batch, embed_dim]`` containing the arithmetic mean of each sample's
        valid token embeddings.

    Parameters:
        ``vocab_size``: number of embedding rows. ``embed_dim``: embedding width (default
        64). ``token_channel`` and ``validity_channel``: the two input-channel positions.
        ``safe_token_id``: in-vocabulary row substituted before lookup at invalid
        positions (default 0); it does not determine validity.

    Devices:
        CPU and accelerators. The returned dtype and device follow the embedding table.

    Limitations:
        Exactly one token stream and one validity stream are accepted. Each sample must
        contain at least one valid token; tokenization and multi-stream fusion are outside
        this block.

    Example:
        >>> import torch
        >>> encoder = EmbeddingEncoder(vocab_size=16, embed_dim=4)
        >>> x = torch.tensor([[[2, 1], [3, 1], [99, 0]]])
        >>> tuple(encoder(x).shape)
        (1, 4)
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        token_channel: int = 0,
        validity_channel: int = 1,
        safe_token_id: int = 0,
    ) -> None:
        super().__init__()
        self.vocab_size = _positive("vocab_size", vocab_size)
        embed_dim = _positive("embed_dim", embed_dim)
        indices = (token_channel, validity_channel)
        if any(
            isinstance(index, bool) or not isinstance(index, int) or index not in (0, 1)
            for index in indices
        ):
            raise ValueError("channel indices must each be zero or one")
        if token_channel == validity_channel:
            raise ValueError("token_channel and validity_channel must be distinct")
        if (
            isinstance(safe_token_id, bool)
            or not isinstance(safe_token_id, int)
            or not 0 <= safe_token_id < vocab_size
        ):
            raise ValueError(f"safe_token_id must be in [0, {vocab_size}), got {safe_token_id!r}")
        self.token_channel = token_channel
        self.validity_channel = validity_channel
        self.safe_token_id = safe_token_id
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        with torch.no_grad():
            self.embedding.weight[safe_token_id].zero_()

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"EmbeddingEncoder expects rank 3 [batch, tokens, channels], "
                f"got shape {tuple(x.shape)}"
            )
        if x.shape[0] < 1:
            raise ValueError("batch axis 0 must be non-empty")
        if x.shape[1] < 1:
            raise ValueError("token axis 1 must be non-empty")
        if x.shape[2] != 2:
            raise ValueError(f"channel axis 2 expected 2, got {x.shape[2]}")
        if x.dtype not in _INTEGER_DTYPES:
            raise ValueError(f"EmbeddingEncoder expects an integer tensor, got {x.dtype}")

        tokens = x[:, :, self.token_channel]
        validity = x[:, :, self.validity_channel]
        binary = (validity == 0) | (validity == 1)
        valid = validity.bool()
        safe_tokens = torch.where(valid, tokens, self.safe_token_id).to(torch.int64)
        torch._assert_async(
            binary.all()
            & valid.any(dim=1).all()
            & ((safe_tokens >= 0) & (safe_tokens < self.vocab_size)).all(),
            "invalid token batch: validity channel must contain only zero or one; "
            "each sample must contain at least one valid token; valid token ids outside "
            "configured vocabulary are forbidden",
        )
        embedded = self.embedding(safe_tokens)
        observed = valid.unsqueeze(-1)
        reduction_dtype = (
            torch.float32 if embedded.dtype in (torch.float16, torch.bfloat16) else embedded.dtype
        )
        values = torch.where(observed, embedded, 0.0).to(reduction_dtype)
        pooled = values.sum(dim=1) / valid.sum(dim=1, keepdim=True).to(reduction_dtype)
        return pooled.to(embedded.dtype)


def _positive(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


__all__ = ["EmbeddingEncoder"]
