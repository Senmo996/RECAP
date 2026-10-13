"""Minimal greedy RECAP inference for Qwen2.5-VL and LLaVA."""

from dataclasses import dataclass, field
import time

import torch
from transformers import AutoConfig, AutoProcessor, AutoTokenizer

from .cache import initialize_recap_cache
from .config import RecapConfig
from .memory import RecapMemory
from .suffix import RecapSuffixIndex
from .tree import build_recap_tree, verify_recap_tree


@dataclass
class RecapOutput:
    sequences: torch.Tensor
    prompt_length: int
    elapsed_seconds: float
    acceptance_lengths: list = field(default_factory=list)
    candidate_counts: list = field(default_factory=list)

    @property
    def generated_tokens(self):
        return self.sequences.shape[1] - self.prompt_length

    @property
    def tokens_per_second(self):
        return self.generated_tokens / self.elapsed_seconds if self.elapsed_seconds else 0.0


class RecapModel(torch.nn.Module):
    """One target model, persistent U, and fresh C/G for each generation call.

    Generation is greedy, unpadded batch size one, on one device. The model is
    stateful and is intended for sequential requests, not concurrent callers.
    """

    def __init__(self, base_model, tokenizer, config=None, processor=None):
        super().__init__()
        self.base_model = base_model.eval()
        self.tokenizer = tokenizer
        self.processor = processor
        self.recap_config = config or RecapConfig()
        self.memory = RecapMemory(self.recap_config.top_k)
        self._topology = {}
        self._cache = None

    @classmethod
    def from_pretrained(cls, model_path, *, device="cuda:0", dtype=torch.float16,
                        config=None, local_files_only=False, attn_implementation=None):
        hf_config = AutoConfig.from_pretrained(model_path, local_files_only=local_files_only)
        arch = hf_config.architectures[0]
        if arch == "Qwen2_5_VLForConditionalGeneration":
            from .backbones.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
            model_class = Qwen2_5_VLForConditionalGeneration
            attention = attn_implementation or "sdpa"
        elif arch in ("LlavaForConditionalGeneration", "LlavaNextForConditionalGeneration"):
            from .backbones.llava import RecapLlavaForConditionalGeneration, RecapLlavaNextForConditionalGeneration
            model_class = (RecapLlavaNextForConditionalGeneration if "Next" in arch
                           else RecapLlavaForConditionalGeneration)
            attention = attn_implementation or "eager"
        else:
            raise ValueError(f"RECAP does not support {arch}")
        if attention not in ("eager", "sdpa"):
            raise ValueError("RECAP tree verification requires eager or sdpa attention")
        base = model_class.from_pretrained(
            model_path, torch_dtype=dtype, device_map=device,
            local_files_only=local_files_only, attn_implementation=attention,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, local_files_only=local_files_only)
        processor = AutoProcessor.from_pretrained(model_path, local_files_only=local_files_only)
        return cls(base, tokenizer, config, processor)

    @property
    def device(self):
        return self.base_model.get_input_embeddings().weight.device

    def reset_recap_cache(self):
        """Clear persistent prediction memory at warmup/dataset boundaries."""
        self.memory.reset()
        self._topology.clear()

    def prepare_inputs(self, prompt=None, image=None, *, messages=None):
        """Encode a single image/text request or explicit chat history."""
        if self.processor is None:
            raise ValueError("prepare_inputs requires a processor")
        if messages is None:
            content = ([{"type": "image"}] if image is not None else [])
            content.append({"type": "text", "text": prompt or "Describe the image."})
            messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image] if image is not None else None,
                                return_tensors="pt")
        return inputs.to(self.device)

    def _stop_ids(self):
        # Retain the original evaluation's special-token stopping convention.
        result = set(getattr(self.tokenizer, "all_special_ids", []))
        for obj in (self.tokenizer, self.base_model.config, self.base_model.generation_config):
            value = getattr(obj, "eos_token_id", None)
            if value is not None:
                result.update(value if isinstance(value, (list, tuple)) else [value])
        eod = getattr(self.tokenizer, "eod_id", None)
        if eod is not None:
            result.add(eod)
        return result

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @torch.inference_mode()
    def recap_generate(self, input_ids, *, max_new_tokens=200, method="recap", **inputs):
        """Return token IDs and compact metrics; decoding text is caller-owned.

        method='target' uses the same prefill and KV implementation as RECAP.
        Time includes prefill, image encoding, proposing, verification and commit.
        """
        if method not in ("recap", "target"):
            raise ValueError("method must be recap or target")
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
            raise ValueError("RECAP requires a nonempty prompt with batch size one")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be nonnegative")
        unsupported = set(inputs) - {"attention_mask", "pixel_values", "image_grid_thw", "image_sizes"}
        if unsupported:
            raise ValueError(f"Unsupported generation inputs: {sorted(unsupported)}")
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None and not bool(attention_mask.bool().all()):
            raise ValueError("RECAP requires unpadded batch-size-one inputs")
        self.memory.reset_request()
        prompt_length = input_ids.shape[1]
        if max_new_tokens == 0:
            return RecapOutput(input_ids.clone(), prompt_length, 0.0)
        arch = self.base_model.config.architectures[0]
        qwen = arch == "Qwen2_5_VLForConditionalGeneration"
        blocked = getattr(self.base_model.config, "image_token_id" if qwen else "image_token_index", None)
        capacity = prompt_length + max_new_tokens + self.recap_config.node_budget + 5
        # Allocate outside the timed region, just as the original reusable cache.
        if self._cache is None:
            self._cache, self._lengths = initialize_recap_cache(self.base_model, capacity)
        for layer in self._cache:
            for cache in layer:
                cache._ensure_capacity(capacity)
        self._lengths.zero_()
        if method == "recap" and self.recap_config.backend == "cpp":
            from .fast_tree import load_recap_extension
            load_recap_extension()  # JIT compilation is not inference latency.
        self._sync()
        start = time.perf_counter()
        output = self.base_model(input_ids=input_ids, past_key_values=self._cache,
                                 use_cache=True, return_dict=True, **inputs)
        prompt_tokens = input_ids[0].tolist()
        if method == "recap":
            self.memory.seed_prompt(output.logits[0], prompt_tokens)
        next_token = int(output.logits[0, -1].argmax())
        history = prompt_tokens + [next_token]
        suffix = RecapSuffixIndex()
        if method == "recap" and self.recap_config.suffix_tokens:
            suffix.update(history)
        stop_ids = self._stop_ids()
        acceptance, candidates = [], []
        decoder = getattr(self.base_model, "language_model", self.base_model)
        mask_model = getattr(decoder, "model", decoder)
        while len(history) - prompt_length < max_new_tokens and history[-1] not in stop_ids:
            remaining = max_new_tokens - (len(history) - prompt_length)
            root = history[-1]
            prev = history[-2] if len(history) >= 2 else None
            prevprev = history[-3] if len(history) >= 3 else None
            tree = None
            if method == "recap":
                tree = build_recap_tree(self.memory, root, prev, prevprev, self.recap_config,
                                        blocked, suffix.continuation(self.recap_config.suffix_tokens),
                                        self.device, self._topology)
            packed = tree is not None and len(tree.tokens) > 1 and remaining > 1
            values = tree.tokens if packed else [root]
            tree_input = torch.tensor([values], device=self.device, dtype=input_ids.dtype)
            base_length = int(self._lengths[0])
            forward_kwargs = {}
            if packed:
                positions = tree.positions + base_length
                if qwen:
                    positions = (positions[None] + self.base_model.rope_deltas)[None].expand(3, -1, -1)
                else:
                    positions = positions[None]
                forward_kwargs["position_ids"] = positions
                mask_model.tree_mask = tree.mask
            try:
                output = self.base_model(input_ids=tree_input, past_key_values=self._cache,
                                         use_cache=True, return_dict=True, **forward_kwargs)
            finally:
                mask_model.tree_mask = None
            predictions = output.logits[0].argmax(dim=-1)
            if method == "recap":
                self.memory.update(output.logits[0], values,
                                   tree.previous if packed else [prev],
                                   tree.previous_previous if packed else [prevprev])
            if packed:
                accepted, new_tokens = verify_recap_tree(predictions.tolist(), tree, remaining)
            else:
                accepted, new_tokens = [], [int(predictions[0])]
            # Stop at the first special token, even if it is inside a matched path.
            for i, token in enumerate(new_tokens):
                if token in stop_ids:
                    new_tokens = new_tokens[:i + 1]
                    break
            accepted = accepted[:len(new_tokens)]
            if any(index != i for i, index in enumerate(accepted, 1)):
                indices = torch.tensor([base_length] + [base_length + n for n in accepted],
                                       device=self.device, dtype=torch.long)
                for layer in self._cache:
                    for cache in layer:
                        cache.copy(indices, base_length)
            self._lengths.fill_(base_length + len(accepted) + 1)
            history.extend(new_tokens)
            if method == "recap" and self.recap_config.suffix_tokens:
                suffix.update(new_tokens)
            acceptance.append(len(accepted))
            candidates.append(len(values) - 1)
        sequences = torch.tensor([history], device=self.device, dtype=input_ids.dtype)
        self._sync()
        elapsed = time.perf_counter() - start
        # C/G are request-local, including after returning to the caller.
        self.memory.reset_request()
        return RecapOutput(sequences, prompt_length, elapsed, acceptance, candidates)
