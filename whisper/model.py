from dataclasses import dataclass
from typing import Dict
from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn

from .transcribe import transcribe as transcribe_function
from .decoding import detect_language as detect_language_function, decode as decode_function


@dataclass
class ModelDimensions:
    n_mels: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int
    n_vocab: int
    n_text_ctx: int
    n_text_state: int
    n_text_head: int
    n_text_layer: int


class LayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)


class Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x, self.weight.to(x.dtype), None if self.bias is None else self.bias.to(x.dtype)
        )


class Conv1d(nn.Conv1d):
    def _conv_forward(self, x: Tensor, weight: Tensor, bias: Optional[Tensor]) -> Tensor:
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
        )


def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


class MultiHeadAttention(nn.Module):
    layer_count = 0

    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)
        self.layer_id = MultiHeadAttention.layer_count
        MultiHeadAttention.layer_count += 1

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[Tensor] = None,
        af_cache_write: Optional[Tensor] = None,
        af_cache_read: Optional[Tensor] = None,
        encoder = False
    ):

        

        # if kv_cache is None or xa is None or self.key not in kv_cache:
        #     # hooks, if installed (i.e. kv_cache is not None), will prepend the cached kv tensors;
        #     # otherwise, perform key/value projections for self- or cross-attention as usual.
        if xa is not None or af_cache_read is not None:
            if af_cache_read is None:
                k = self.key(xa)
                v = self.value(xa)
                # af_cache_write[self.layer_id - 13, :, :, :] = k
                # af_cache_write[self.layer_id - 13 + 1, :, :, :] = v
                if encoder:
                    return [k, v]
                # print(f"1st {af_cache_write.shape=}")
            else:
                # k_ = self.key(xa)
                # v_ = self.value(xa)
                k = af_cache_read[self.layer_id - 13]
                v = af_cache_read[self.layer_id - 13 + 1]
                # print("2nd")
                # assert (k_ == k).all()
                # assert (v_ == v).all()

        else:
            k = self.key(x)
            v = self.value(x)
            # print("3rd")
            if kv_cache is not None:
                # print("4th")
                key_id = self.layer_id - 4 - 8
                value_id = key_id + 1
                size = k.shape[1]
                kv_cache[key_id, :, -size:, :] = k
                kv_cache[value_id, :, -size:, :] = v
                k = kv_cache[key_id]
                v = kv_cache[value_id]

        assert not encoder
        q = self.query(x)
        # else:
        #     # for cross-attention, calculate keys and values once and reuse in subsequent calls.
        #     k = kv_cache[self.key]
        #     v = kv_cache[self.value]

        wv = self.qkv_attention(q, k, v, mask)
        return self.out(wv)

    def qkv_attention(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None):
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        qk = q @ k
        if mask is not None:
            qk = qk + mask[:n_ctx, :n_ctx]

        w = F.softmax(qk.float(), dim=-1).to(q.dtype)
        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False):
        super().__init__()

        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = LayerNorm(n_state)

        self.cross_attn = MultiHeadAttention(n_state, n_head) if cross_attention else None
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state))
        self.mlp_ln = LayerNorm(n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[Tensor] = None,
        af_cache_write: Optional[Tensor] = None,
        af_cache_read: Optional[Tensor] = None,
        encoder = False,
    ):
        if encoder:
            return self.cross_attn(None, xa, kv_cache=None, af_cache_write=af_cache_write, af_cache_read=None, encoder=True)

        x = x + self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache, af_cache_write=af_cache_write)
        if self.cross_attn:
            x = x + self.cross_attn(self.cross_attn_ln(x), xa, kv_cache=kv_cache, af_cache_write=af_cache_write, af_cache_read=af_cache_read)
        x = x + self.mlp(self.mlp_ln(x))
        return x


class AudioEncoder(nn.Module):
    def __init__(self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int):
        super().__init__()
        self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        )
        self.ln_post = LayerNorm(n_state)

    def forward(self, x: Tensor):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)

        assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
        x = (x + self.positional_embedding).to(x.dtype)

        for block in self.blocks:
            x = block(x)

        x = self.ln_post(x)
        return x


class TextDecoder(nn.Module):
    def __init__(self, n_vocab: int, n_ctx: int, n_state: int, n_head: int, n_layer: int):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_state)
        self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head, cross_attention=True) for _ in range(n_layer)]
        )
        self.ln = LayerNorm(n_state)

        mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: Tensor, xa: Optional[Tensor], kv_cache_read: Tensor, af_cache_write: Optional[Tensor] = None, af_cache_read: Optional[Tensor] = None, encoder = False):
        """
        x : torch.LongTensor, shape = (batch_size, <= n_ctx)
            the text tokens
        xa : torch.Tensor, shape = (batch_size, n_mels, n_audio_ctx)
            the encoded audio features to be attended on
        """
        if encoder:
            assert kv_cache_read is None
            kv_lst = []
            for block in self.blocks:
                kv_lst += block(None, xa, mask=None, kv_cache=None, af_cache_write=af_cache_write, af_cache_read=None, encoder=True)
            af_cache_write2 = torch.stack(kv_lst, dim=0)
            # assert torch.equal(af_cache_write2, af_cache_write)
            return None, None, af_cache_write2

        # print(f"{x.shape=} {kv_cache_read.shape=}")
        (kv_write_shape := list(kv_cache_read.shape)).__setitem__(-2, x.shape[-1])
        kv_cache_write = torch.zeros(kv_write_shape, dtype=kv_cache_read.dtype, device=kv_cache_read.device)
        # assert kv_cache_write_.equal(kv_cache_write)
        

        kv_cache = torch.cat((kv_cache_read, kv_cache_write), dim=-2)
        offset = kv_cache_read.shape[2]

        x = self.token_embedding(x) + self.positional_embedding[offset : offset + x.shape[-1]]
        audio_tensor = xa if xa is not None else af_cache_read
        x = x.to(audio_tensor.dtype)

        for block in self.blocks:
            x = block(x, xa, mask=self.mask, kv_cache=kv_cache, af_cache_write=af_cache_write, af_cache_read=af_cache_read)

        x = self.ln(x)
        logits = (x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)).float()

        return logits, kv_cache, af_cache_write

    def forward_encoder(self, x: Tensor, xa: Tensor, kv_cache: Tensor, af_cache_write: Optional[Tensor] = None):
        _, __, af_cache_write_out = self.forward(x, xa, kv_cache, af_cache_write, af_cache_read=None)
        return af_cache_write_out


class Whisper(nn.Module):
    def __init__(self, dims: ModelDimensions, name: str):
        super().__init__()
        self.name = name
        self.dims = dims
        self.encoder = AudioEncoder(
            self.dims.n_mels,
            self.dims.n_audio_ctx,
            self.dims.n_audio_state,
            self.dims.n_audio_head,
            self.dims.n_audio_layer,
        )
        self.decoder = TextDecoder(
            self.dims.n_vocab,
            self.dims.n_text_ctx,
            self.dims.n_text_state,
            self.dims.n_text_head,
            self.dims.n_text_layer,
        )

    def embed_audio(self, mel: torch.Tensor):
        return self.encoder(mel)

    def logits(self, tokens: torch.Tensor, audio_features: torch.Tensor):
        return self.decoder(tokens, audio_features)

    def forward(self, mel: torch.Tensor, tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.decoder(tokens, self.encoder(mel))

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def is_multilingual(self):
        return self.dims.n_vocab == 51865

    def install_kv_cache_hooks(self, cache: Optional[dict] = None):
        """
        The `MultiHeadAttention` module optionally accepts `kv_cache` which stores the key and value
        tensors calculated for the previous positions. This method returns a dictionary that stores
        all caches, and the necessary hooks for the key and value projection modules that save the
        intermediate tensors to be reused during later calculations.

        Returns
        -------
        cache : Dict[nn.Module, torch.Tensor]
            A dictionary object mapping the key/value projection modules to its cache
        hooks : List[RemovableHandle]
            List of PyTorch RemovableHandle objects to stop the hooks to be called
        """
        cache = {**cache} if cache is not None else {}
        hooks = []

        def save_to_cache(module, _, output):
            if module not in cache or output.shape[1] > self.decoder.positional_embedding.shape[0]:
                cache[module] = output  # save as-is, for the first token or cross attention
            else:
                cache[module] = torch.cat([cache[module], output], dim=1).detach()
            return cache[module]

        def install_hooks(layer: nn.Module):
            if isinstance(layer, MultiHeadAttention):
                hooks.append(layer.key.register_forward_hook(save_to_cache))
                hooks.append(layer.value.register_forward_hook(save_to_cache))

        self.decoder.apply(install_hooks)
        return cache, hooks

    def new_kv_cache(self, n_group: int, length: int, device: torch.device, dtype: torch.dtype):
        if self.name == "tiny.en" or self.name == "tiny":
            size = [8, n_group, length, 384]
        elif self.name == "base.en" or self.name == "base":
            size = [12, n_group, length, 512]
        elif self.name == "small.en" or self.name == "small":
            size = [24, n_group, length, 768]
        elif self.name == "medium.en" or self.name == "medium":
            size = [48, n_group, length, 1024]
        elif self.name == "large":
            size = [64, n_group, length, 1280]
        else:
            raise ValueError(f"Unsupported model type: {self.name}")
        return torch.zeros(size, dtype=dtype, device=device)

    def new_af_cache(self, n_group: int, device: torch.device, dtype: torch.dtype):
        length = 1500
        if self.name == "tiny.en" or self.name == "tiny":
            size = [8, n_group, length, 384]
        elif self.name == "base.en" or self.name == "base":
            size = [12, n_group, length, 512]
        elif self.name == "small.en" or self.name == "small":
            size = [24, n_group, length, 768]
        elif self.name == "medium.en" or self.name == "medium":
            size = [48, n_group, length, 1024]
        elif self.name == "large":
            size = [64, n_group, length, 1280]
        else:
            raise ValueError(f"Unsupported model type: {self.name}")
        return torch.zeros(size, dtype=dtype, device=device)

    detect_language = detect_language_function
    transcribe = transcribe_function
    decode = decode_function
