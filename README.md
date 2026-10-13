# RECAP

**English** | [简体中文](README.zh-CN.md)

This repository provides the **official source code** for the paper:

> **RECAP: Context-Aware Prediction Recycling with Request-Local and Persistent Memories for Multimodal Speculative Decoding**

**RECAP** (REcycling Context-Aware Predictions) is a training-free method for multimodal
speculative decoding. It accelerates vision-language model inference by reusing the target model's
own predictions, without an auxiliary draft model. To account for differences in images and text
prefixes, RECAP combines persistent token-level memory with request-local memories indexed by
two- and three-token contexts. It fuses retrieved predictions using key specificity and
entropy-based confidence, constructs a candidate tree with suffix-matched continuations, and
verifies the tree in a single target-model pass.

This release provides a standalone implementation of the paper's core inference method for
**Qwen2.5-VL** and **LLaVA / LLaVA-Next**, including Python/C++ tree builders, runnable examples,
and compact paired evaluation. For the evaluation protocol and its relationship to the paper's
experiments, see [Compact evaluation](#compact-evaluation). See [Algorithm and API](#algorithm-and-api)
for the method's components and default configuration.

## Installation

Validated environment: Python 3.10, PyTorch 2.7.1 / CUDA 12.6, and Transformers 4.57.1.
Transformers is pinned because packed tree attention relies on its internal interfaces.
Install PyTorch and torchvision for your machine, then run:

```bash
git clone https://github.com/Senmo996/RECAP.git recap
cd recap
python -m pip install -e .
python -m recap self-check
```

The Python tree builder requires no compiler. The optional C++ builder provides the optimized path
used in the original main experiments and requires a C++17 compiler and Ninja:

```bash
python -m pip install -e '.[cpp,test]'
python -m recap self-check --backend cpp
```

The C++ extension is compiled on first use. Its build cache defaults to
`outputs/cache/recap_tree/` under the current working directory; set `RECAP_BUILD_DIR` to override it.
Compilation time is excluded from inference timing.
To run without installing the package, use `PYTHONPATH=src python -m recap ...` from this directory.

## Single-image inference

Provide a local model directory or a Hugging Face model ID. Model weights are not included.

```bash
python -m recap generate \
  --model /path/to/Qwen2.5-VL-7B-Instruct \
  --device cuda:0 --backend cpp --local-files-only \
  --image examples/shapes.png \
  --prompt 'What are the colors and shapes in this image?' \
  --max-new-tokens 64
```

For LLaVA-1.6, replace `--model` with its model directory in Hugging Face format.
Qwen uses SDPA by default; LLaVA uses eager attention. Omit `--image` for text-only requests.
Omitting `--local-files-only` allows downloads from the model Hub.

```python
from PIL import Image
from recap import RecapConfig, RecapModel

model = RecapModel.from_pretrained(
    '/path/to/Qwen2.5-VL-7B-Instruct',
    device='cuda:0',
    config=RecapConfig(backend='cpp'),
    local_files_only=True,
)
inputs = model.prepare_inputs('Describe this image.', Image.open('examples/shapes.png').convert('RGB'))
result = model.recap_generate(**inputs, max_new_tokens=64)
answer = model.tokenizer.decode(result.sequences[0, result.prompt_length:], skip_special_tokens=True)
print(answer)

# Clear persistent prediction memory after warmup, between benchmarks,
# or when separate request streams need independent memory.
model.reset_recap_cache()
```

## Compact evaluation

Input is an ordered JSONL file with one request per line. Image paths are relative to that file.
Each record can supply a `prompt`, or a complete conversation history through `messages` in the
format accepted by the processor's chat template. Every generation call creates fresh contextual
memories; supplying conversation history does not reuse the previous request's KV cache.

```json
{"id":"sample-1","image":"shapes.png","prompt":"Describe the image."}
```

For a small foreground run:

```bash
python -m recap evaluate \
  --model /path/to/Qwen2.5-VL-7B-Instruct \
  --device cuda:0 --backend cpp --local-files-only \
  --input examples/requests.jsonl --output outputs/eval/example \
  --max-new-tokens 64 --require-token-match
```

For long evaluations, use the file-based launcher with a named tmux session:

```bash
PYTHON_BIN=/path/to/python bash scripts/launch_recap.sh \
  --model /path/to/Qwen2.5-VL-7B-Instruct \
  --device cuda:0 --backend cpp --local-files-only \
  --input /path/to/requests.jsonl --output outputs/eval/my-run
```

The launcher prints the session name and the log/status file paths. Logs are stored under
`outputs/logs/eval/` with timestamp-prefixed filenames. Inspect a run with
`tmux attach -t <session>` and stop it with `tmux kill-session -t <session>`.
A successful run writes `exit_code=0` to its `.status` file. Existing evaluation results are not overwritten.

The evaluator alternates the target/RECAP execution order between samples and clears prediction
memory after warmup. It writes:

- `results.jsonl`: acceptance lengths, candidate counts, timings, token hashes, and token equality.
- `summary.json`: record count, mean/median/P90 acceptance length, speedup, and token match ratio.

By default, evaluation neither decodes token IDs into text nor saves generated text.
Use `--save-decoded-output` to opt in to text output. With `--require-token-match`, differing paired
outputs cause a nonzero exit status while retaining results for diagnosis.
Per-request speedup is RECAP tokens/s divided by target tokens/s.
Acceptance length excludes the root and correction token; tokens per iteration excludes the first
token produced by prefill.

Timing includes image encoding, prefill, tree construction, verification, and commitment. It excludes
input preprocessing, model loading, KV buffer preallocation, and extension compilation.
This evaluator supports runnable examples and paired performance checks; it does not replace the
paper's fixed dataset manifests, request order, or timing protocol. The minimal implementation
encodes images on each request and does not retain the original evaluator's image feature cache
across conversation turns. Packed and token-by-token matrix operations can differ through fp16/bf16
rounding, so the evaluator reports the observed token match ratio.

## Algorithm and API

Defaults follow the main paper configuration: Top-8 prediction rows, maximum depth 10, and 63
transition candidate nodes. Suffix retrieval can add up to 4 more nodes; the root is excluded from
this budget. With `fusion=prior_confidence`, available sources receive priors `(0.70, 0.20, 0.10)`
in decreasing specificity order, multiplied by `exp(-entropy)`.
Use `fusion=fixed` for the fixed-prior control.

- `RecapMemory`: U persists across requests; C/G are recreated for each request. Prefill seeds only U.
- `recap_fuse()`: confidence-weighted fusion of prediction rows with different context lengths.
- `build_recap_tree()`: Python/C++ best-first tree construction and suffix path insertion.
- `RecapSuffixIndex`: a suffix automaton containing only the prompt and committed tokens.
- `RecapModel.recap_generate()`: one target pass verifies the entire tree, then commits the longest matching path and a target correction.
- `reset_recap_cache()`: clears persistent memory.

Every target-verified node updates prediction memory, including nodes outside the committed path.
Within a single target forward pass, the first prediction row for each memory key takes precedence.
Decoding supports unpadded inputs, batch size 1, a single device, and greedy generation. Sampling
parameters are not supported. Stopping follows the original evaluator: EOS/EOD and all tokenizer
special tokens are stop boundaries. The model retains state and should be called sequentially.

## Layout and validation

```text
src/recap/
  model.py          # RecapModel and decoding loop
  memory.py         # U/C/G memories and fusion
  tree.py           # Tree construction, topology packing, and matching
  recap_tree.cpp    # Optional C++ CPU implementation
  suffix.py         # Suffix automaton
  cache.py          # Rewindable KV cache with compaction
  backbones/        # Qwen/LLaVA tree attention adapters
  cli.py            # Standalone command-line interface
  evaluation.py     # Compact metrics and summaries
tests/              # Memory scope, tree backends, caches, and tiny-model parity
examples/           # Two requests and a self-contained example image
scripts/            # tmux launcher
```

```bash
mkdir -p outputs
RECAP_TEST_CPP=1 python -m pytest -q --basetemp=outputs/test-tmp
```

Without `RECAP_TEST_CPP=1`, compiler checks are skipped. All other tests run on CPU without
downloading model weights. See [NOTICE.md](NOTICE.md) for source provenance and licensing.

Extraction validation on 2026-10-13: 11 tests passed; 160 tree constructions and 200 suffix updates
matched the original implementation. Qwen2.5-VL-7B and LLaVA-1.6-7B each processed one text request
and one image request with a maximum of 32 generated tokens, matching token-by-token decoding with
the same target model. The wheel passed import and C++ self-checks after installation in an isolated
directory. These are functionality checks, not a full reproduction of the performance experiments.
