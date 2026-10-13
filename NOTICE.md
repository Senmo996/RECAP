# Source provenance

This standalone extraction is based on the local RECAP implementation in the
MMSpec experiment repository, extracted on 2026-10-13. It retains that source
tree's Apache-2.0 license in `LICENSE`. Original copyright notices in the Qwen
backbone are preserved.

Source mapping (paths describe provenance; they are not runtime dependencies):

| Standalone code | Original source |
| --- | --- |
| `model.py`, `memory.py`, `tree.py` | `MMSpec/method/sam_grounded/tree_recycling_model.py`, main RECAP configuration |
| `memory.py:recap_fuse` | `MMSpec/method/sam_grounded/fusion_weighting.py` |
| `recap_tree.cpp` | `MMSpec/method/sam_grounded/fast_tree_builder.cpp` |
| `suffix.py` | `MMSpec/method/sam/samd/dyn_sam.py` and `sam_draft.py`, corrected continuation lookup |
| `cache.py` | `MMSpec/method/vispec/kv_cache.py`, simplified to live per-layer buffers |
| `backbones/qwen2_5_vl.py` | `MMSpec/method/vispec/modeling_qwen2_5_vl_kv.py` |
| `backbones/llava.py` | `MMSpec/method/llava_adapter.py` |

Changes include the `recap` public naming, removal of unrelated algorithms and
diagnostic/ablation branches, a standalone input/evaluation interface, an
optional compiler-free Python tree builder, and KV compaction that operates on
live cache tensors after capacity growth. Request-local memories are cleared at
request boundaries. The C++ implementation has its unused root-backoff control
removed. The main confidence-weighted fusion and tree ordering are retained.

`examples/shapes.png` is a synthetic red-square/blue-circle test fixture created
for this extraction. Model weights and third-party benchmark datasets are not
redistributed.
