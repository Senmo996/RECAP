"""Semantic checks for extraction, memory scope, tree attention and commitment."""

import json
import os
import random
from types import SimpleNamespace

import pytest
import torch

from recap import RecapConfig, RecapModel
from recap.cache import RecapKVCache
from recap.evaluation import recap_record, summarize_recap
from recap.memory import RecapMemory, recap_fuse
from recap.suffix import RecapSuffixIndex
from recap.tree import build_recap_tree, verify_recap_tree


def test_memory_scope_first_occurrence_and_full_tree_updates():
    memory = RecapMemory(2)
    logits = torch.tensor([[4., 1., 0.], [0., 1., 5.], [0., 5., 1.]])
    memory.update(logits, [8, 8, 9], [1, 2, 3], [4, 5, 6])
    assert memory.rows[0][8][0] == 0
    assert memory.rows[1][(2, 8)][0] == 2
    assert memory.rows[2][(6, 3, 9)][0] == 1
    memory.reset_request()
    assert memory.rows[0] and not memory.rows[1] and not memory.rows[2]
    memory.reset()
    assert all(not rows for rows in memory.rows)


def test_available_source_ranks_and_confidence_fusion():
    # Only C and U are present: their weights are .70 and .20.
    rows = [([1, 2], [.5, .5]), ([3, 2], [.99, .01])]
    tokens, values = recap_fuse(rows, 3, confidence=False)
    assert tokens == [2, 1, 3]
    assert values == pytest.approx([.352 / .9, .35 / .9, .198 / .9])
    tokens, values = recap_fuse(rows, 3, confidence=True)
    assert tokens[0] == 3
    assert sum(values) == pytest.approx(1.)


def test_suffix_uses_current_root_once():
    suffix = RecapSuffixIndex()
    suffix.update([5, 6, 7, 8, 5, 6])
    assert suffix.continuation(2) == [7, 8]
    suffix.update([7])
    assert suffix.continuation(1) == [8]


def test_tree_budget_ancestry_blocking_and_verification():
    memory = RecapMemory(2)
    memory.rows[0] = {1: [2, 9], 2: [3, 1], 3: [4, 5]}
    memory.scores[0] = {1: [.9, .1], 2: [.9, .1], 3: [.9, .1]}
    tree = build_recap_tree(memory, 1, 8, 7, RecapConfig(node_budget=3),
                            blocked=9, suffix=[2, 3, 6, 7])
    assert len(tree.tokens) <= 1 + 3 + 4
    assert 9 not in tree.tokens
    for i, path in enumerate(tree.paths, 1):
        assert set(torch.where(tree.mask[0, 0, i])[0].tolist()) == {0, *path}
        assert tree.depths[i] == len(path)
        assert tree.previous[i] == tree.tokens[tree.parents[i]]
        assert tree.previous_previous[i] == tree.previous[tree.parents[i]]
    predictions = [{1: 2, 2: 3, 3: 6, 6: 7}.get(t, 0) for t in tree.tokens]
    accepted, tokens = verify_recap_tree(predictions, tree, 3)
    assert len(accepted) == 2 and tokens == [2, 3, 6]


def test_cache_growth_then_noncontiguous_compaction():
    cache = RecapKVCache(torch.zeros(1, 1, 2, 1), torch.tensor(0))
    cache.cat(torch.arange(6.).view(1, 1, 6, 1))
    cache.copy(torch.tensor([2, 5]), 2)
    assert cache.shape[-2] == 4
    assert cache.data[0, 0, :4, 0].tolist() == [0., 1., 2., 5.]


@pytest.mark.skipif(os.environ.get("RECAP_TEST_CPP") != "1", reason="opt-in compiler check")
@pytest.mark.parametrize("fusion", ["fixed", "prior_confidence"])
def test_cpp_python_parity(fusion):
    rng = random.Random(42)
    memory = RecapMemory(8)
    for _ in range(40):
        for _ in range(100):
            history = [rng.randrange(12) for _ in range(3)]
            for order in range(3):
                key = history[-1] if order == 0 else tuple(history[-order - 1:])
                tokens = rng.sample(range(12), 8)
                values = [rng.random() for _ in tokens]
                total = sum(values)
                memory.rows[order][key] = tokens
                memory.scores[order][key] = [v / total for v in values]
        kwargs = dict(memory=memory, root=3, previous=2, previous_previous=1,
                      blocked=11, suffix=[8, 6, 2, 5])
        py = build_recap_tree(config=RecapConfig(fusion=fusion), **kwargs)
        cpp = build_recap_tree(config=RecapConfig(fusion=fusion, backend="cpp"), **kwargs)
        assert py.tokens == cpp.tokens and py.parents == cpp.parents
        assert torch.equal(py.mask, cpp.mask)


def _tiny_model(architecture):
    torch.manual_seed(42)
    tokenizer = SimpleNamespace(all_special_ids=[], eos_token_id=None)
    if architecture == "qwen":
        from transformers import Qwen2_5_VLConfig
        from recap.backbones.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
        config = Qwen2_5_VLConfig(
            vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2,
            rope_scaling={"type": "mrope", "mrope_section": [1, 1, 2]},
            vision_config={"depth": 1, "hidden_size": 32, "intermediate_size": 64,
                           "num_heads": 4, "out_hidden_size": 32},
            eos_token_id=None,
        )
        config.architectures = ["Qwen2_5_VLForConditionalGeneration"]
        config._attn_implementation = "sdpa"
        base = Qwen2_5_VLForConditionalGeneration(config)
    else:
        from transformers import LlavaConfig
        from recap.backbones.llava import RecapLlavaForConditionalGeneration, _patch_tree_mask_for_llama_model
        config = LlavaConfig(
            text_config={"vocab_size": 64, "hidden_size": 32, "intermediate_size": 64,
                         "num_hidden_layers": 2, "num_attention_heads": 4,
                         "num_key_value_heads": 2, "eos_token_id": None},
            vision_config={"hidden_size": 32, "intermediate_size": 64,
                           "num_hidden_layers": 2, "num_attention_heads": 4,
                           "image_size": 28, "patch_size": 14}, image_token_index=63,
        )
        config.architectures = ["LlavaForConditionalGeneration"]
        config._attn_implementation = "eager"
        base = RecapLlavaForConditionalGeneration(config)
        _patch_tree_mask_for_llama_model(base)
    return RecapModel(base, tokenizer, RecapConfig(node_budget=15, depth=6))


@pytest.mark.parametrize("architecture", ["qwen", "llava"])
def test_tiny_backbones_match_target_across_requests(architecture, tmp_path):
    torch.set_num_threads(1)
    model = _tiny_model(architecture)
    for ids in ([1, 5, 7, 1, 5], [9, 2, 3, 9, 2]):
        inputs = torch.tensor([ids])
        target = model.recap_generate(inputs, method="target", max_new_tokens=24)
        recap = model.recap_generate(inputs, max_new_tokens=24)
        assert torch.equal(target.sequences, recap.sequences)
        assert recap.generated_tokens == 24
        assert any(recap.candidate_counts)
        assert any(recap.acceptance_lengths)
        assert not model.memory.rows[1] and not model.memory.rows[2]
    row = recap_record("tiny", target, recap)
    path = tmp_path / "results.jsonl"
    path.write_text(json.dumps(row) + "\n")
    summary = summarize_recap(path)
    assert summary["output_token_match_ratio"] == 1.
    assert summary["num_records"] == 1
    assert "decoded_output" not in row
    assert model.recap_generate(inputs, max_new_tokens=0).generated_tokens == 0
    assert model.recap_generate(inputs, max_new_tokens=1).generated_tokens == 1
    model.reset_recap_cache()
    assert not model.memory.rows[0]


def test_stop_token_inside_tree_and_no_decode():
    model = _tiny_model("qwen")
    inputs = torch.tensor([[1, 5, 7, 1, 5]])
    target = model.recap_generate(inputs, method="target", max_new_tokens=24)
    sequence = target.sequences[0, inputs.shape[1]:].tolist()
    stop = next(token for token in sequence[1:] if token != sequence[0])
    model.tokenizer.all_special_ids = [stop]
    result = model.recap_generate(inputs, max_new_tokens=24)
    assert result.sequences[0, -1] == stop
    assert result.generated_tokens == sequence.index(stop) + 1
    assert result.sequences.tolist() == target.sequences[:, :inputs.shape[1] + result.generated_tokens].tolist()


def test_input_contract():
    model = _tiny_model("qwen")
    with pytest.raises(ValueError, match="batch size"):
        model.recap_generate(torch.ones(2, 3, dtype=torch.long))
    with pytest.raises(ValueError, match="unpadded"):
        model.recap_generate(torch.ones(1, 3, dtype=torch.long), attention_mask=torch.tensor([[1, 1, 0]]))
    with pytest.raises(ValueError, match="Unsupported"):
        model.recap_generate(torch.ones(1, 3, dtype=torch.long), temperature=1.0)
