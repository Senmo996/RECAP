"""Rewindable KV buffers. Compaction uses live buffers even after growth."""

import torch


class RecapKVCache:
    def __init__(self, data, current_length):
        self.data = data
        self.current_length = current_length

    @property
    def shape(self):
        return (*self.data.shape[:2], int(self.current_length), self.data.shape[3])

    def _ensure_capacity(self, needed_length, dim=2):
        capacity = self.data.shape[dim]
        if needed_length > capacity:
            shape = list(self.data.shape)
            shape[dim] = max(needed_length, capacity + max(capacity // 2, 256))
            grown = torch.empty(shape, device=self.data.device, dtype=self.data.dtype)
            grown.narrow(dim, 0, capacity).copy_(self.data)
            self.data = grown

    def cat(self, tensor, dim=2):
        start = int(self.current_length)
        self._ensure_capacity(start + tensor.shape[dim], dim)
        self.data.narrow(dim, start, tensor.shape[dim]).copy_(tensor)
        self.current_length.add_(tensor.shape[dim])
        return self.data.narrow(dim, 0, int(self.current_length))

    def copy(self, indices, prev_length, dim=2):
        selected = self.data.index_select(dim, indices.to(self.data.device))
        self._ensure_capacity(prev_length + selected.shape[dim], dim)
        self.data.narrow(dim, prev_length, selected.shape[dim]).copy_(selected)
        self.current_length.fill_(prev_length + selected.shape[dim])


def initialize_recap_cache(model, capacity):
    decoder = getattr(model, "language_model", model)
    config = decoder.config
    if not hasattr(config, "num_hidden_layers"):
        config = model.config.text_config
    layers = getattr(decoder, "model", decoder).layers
    lengths = torch.zeros(config.num_hidden_layers * 2, dtype=torch.long)
    caches = []
    for i, layer in enumerate(layers):
        weight = layer.self_attn.q_proj.weight
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        shape = (2, 1, config.num_key_value_heads, capacity, head_dim)
        data = torch.empty(shape, dtype=weight.dtype, device=weight.device)
        caches.append([RecapKVCache(data[j], lengths[2 * i + j]) for j in range(2)])
    return caches, lengths
