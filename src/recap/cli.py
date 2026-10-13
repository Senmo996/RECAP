"""recap generate / recap evaluate / recap self-check."""

import argparse
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path

from .config import RecapConfig


def _parser():
    parser = argparse.ArgumentParser(description="RECAP multimodal speculative decoding")
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("self-check", help="CPU algorithm smoke check, no weights required")
    check.add_argument("--backend", choices=("python", "cpp"), default="python")
    for name in ("generate", "evaluate"):
        command = sub.add_parser(name)
        command.add_argument("--model", required=True, help="Local model directory or Hugging Face model ID")
        command.add_argument("--device", default="cuda:0")
        command.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
        command.add_argument("--backend", choices=("python", "cpp"), default="python")
        command.add_argument("--fusion", choices=("prior_confidence", "fixed"), default="prior_confidence")
        command.add_argument("--max-new-tokens", type=int, default=200)
        command.add_argument("--local-files-only", action="store_true")
        command.add_argument("--attn-implementation", choices=("eager", "sdpa"))
        if name == "generate":
            command.add_argument("--image", type=Path)
            command.add_argument("--prompt", default="Describe the image.")
        else:
            command.add_argument("--input", type=Path, required=True, help="Ordered JSONL requests; image paths relative to this file")
            command.add_argument("--output", type=Path, help="New output directory")
            command.add_argument("--limit", type=int)
            command.add_argument("--warmup-tokens", type=int, default=16)
            command.add_argument("--save-decoded-output", action="store_true", help="Opt in to qualitative text output")
            command.add_argument("--require-token-match", action="store_true", help="Fail the evaluation if any paired outputs differ")
    return parser


def _self_check(backend):
    from .memory import RecapMemory
    from .suffix import RecapSuffixIndex
    from .tree import build_recap_tree, verify_recap_tree
    memory = RecapMemory()
    memory.rows[0].update({1: [2, 3], 2: [4]})
    memory.scores[0].update({1: [0.8, 0.2], 2: [1.0]})
    tree = build_recap_tree(memory, 1, None, None, RecapConfig(backend=backend))
    predictions = [2 if t == 1 else 4 if t == 2 else 9 for t in tree.tokens]
    accepted, tokens = verify_recap_tree(predictions, tree, 8)
    assert tokens == [2, 4, 9] and len(accepted) == 2
    suffix = RecapSuffixIndex()
    suffix.update([1, 2, 3, 4, 1, 2])
    assert suffix.continuation(2) == [3, 4]
    print(json.dumps({"method": "recap", "backend": backend, "self_check": "passed"}))


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "self-check":
        return _self_check(args.backend)
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    if args.command == "evaluate" and (args.warmup_tokens < 0 or (args.limit is not None and args.limit < 1)):
        parser.error("--warmup-tokens must be nonnegative and --limit must be positive")
    import torch
    from PIL import Image
    from .model import RecapModel
    from .evaluation import recap_record, summarize_recap

    config = RecapConfig(backend=args.backend, fusion=args.fusion)
    records = []
    if args.command == "evaluate":
        with args.input.open() as stream:
            for line in stream:
                if line.strip():
                    records.append(json.loads(line))
                if args.limit is not None and len(records) >= args.limit:
                    break
        if not records:
            parser.error("--input must contain at least one request")
        output = args.output or Path("outputs/eval") / datetime.now().strftime("%Y%m%d-%H%M%S-recap")
        output.mkdir(parents=True, exist_ok=True)
        result_path = output / "results.jsonl"
        if result_path.exists() or (output / "summary.json").exists():
            parser.error(f"Evaluation outputs already exist in {output}; choose a new directory")
    model = RecapModel.from_pretrained(
        args.model, device=args.device, dtype=getattr(torch, args.dtype), config=config,
        local_files_only=args.local_files_only, attn_implementation=args.attn_implementation,
    )
    if args.command == "generate":
        image = Image.open(args.image).convert("RGB") if args.image else None
        inputs = model.prepare_inputs(args.prompt, image)
        result = model.recap_generate(**inputs, max_new_tokens=args.max_new_tokens)
        print(model.tokenizer.decode(result.sequences[0, result.prompt_length:], skip_special_tokens=True))
        return

    def encode(record):
        image = None
        if record.get("image"):
            with Image.open(args.input.parent / record["image"]) as source:
                image = source.convert("RGB")
        if "prompt" not in record and "messages" not in record:
            raise ValueError("Each request requires prompt or messages")
        return model.prepare_inputs(record.get("prompt"), image, messages=record.get("messages"))

    if args.warmup_tokens:
        first = encode(records[0])
        for method in ("target", "recap"):
            model.recap_generate(**first, method=method, max_new_tokens=args.warmup_tokens)
    model.reset_recap_cache()
    with result_path.open("x") as stream:
        for index, record in enumerate(records):
            inputs = encode(record)
            order = ("target", "recap") if index % 2 == 0 else ("recap", "target")
            paired = {method: model.recap_generate(**inputs, method=method,
                                                 max_new_tokens=args.max_new_tokens) for method in order}
            row = recap_record(record.get("id", index), paired["target"], paired["recap"])
            row["execution_order"] = order
            if args.save_decoded_output:
                result = paired["recap"]
                row["decoded_output"] = model.tokenizer.decode(result.sequences[0, result.prompt_length:], skip_special_tokens=True)
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            print(f"sample={index + 1}/{len(records)} tokens={row['generated_tokens']} match={row['output_token_match']}", flush=True)
    summary = summarize_recap(result_path)
    summary.update(config=asdict(config), model=args.model, device=args.device, dtype=args.dtype,
                   warmup_tokens=args.warmup_tokens, complete=True,
                   timing="end_to_end_generation_including_prefill_excluding_input_processing_and_allocation")
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"summary={summary_path}", flush=True)
    if args.require_token_match and summary["output_token_match_ratio"] != 1.0:
        raise SystemExit("Paired token outputs differ; see summary.json")
