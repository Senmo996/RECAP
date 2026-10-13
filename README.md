# RECAP

RECAP（REcycling Context-Aware Predictions）的独立、最小推理代码库。
支持 **Qwen2.5-VL** 和 **LLaVA / LLaVA-Next**，无需训练或额外 draft model。
整个目录可以单独复制使用，不依赖原工程的 `MMSpec`、`new_dream`、`method` 或数据路径。

## 安装

已验证环境：Python 3.10、PyTorch 2.7.1 / CUDA 12.6、Transformers 4.57.1。
底层 packed-tree attention 与 Transformers 内部接口有关，因此固定该版本。
先根据机器安装匹配的 PyTorch / torchvision，再运行：

```bash
git clone https://github.com/Senmo996/RECAP.git recap
cd recap
python -m pip install -e .
python -m recap self-check
```

Python 树构建器无需编译器。C++ 实现用于原主实验的加速路径，需要 C++17 编译器和 Ninja：

```bash
python -m pip install -e '.[cpp,test]'
python -m recap self-check --backend cpp
```

C++ 首次运行会编译，缓存默认写入当前目录的 `outputs/cache/recap_tree/`。
可通过 `RECAP_BUILD_DIR` 指定目录；编译时间不计入推理时间。
也可不安装包，在本目录使用 `PYTHONPATH=src python -m recap ...`。

## 单张图片生成

模型权重需自行提供本地目录或 Hugging Face 模型 ID。代码库不包含模型权重。

```bash
python -m recap generate \
  --model /path/to/Qwen2.5-VL-7B-Instruct \
  --device cuda:0 --backend cpp --local-files-only \
  --image examples/shapes.png \
  --prompt 'What are the colors and shapes in this image?' \
  --max-new-tokens 64
```

将 `--model` 换为 LLaVA-1.6 的 Hugging Face 格式模型目录即可。
Qwen 默认使用 SDPA，LLaVA 默认使用 eager attention。纯文本请求可以省略 `--image`。
不使用 `--local-files-only` 时允许从模型 Hub 下载。

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

# 基准边界、warmup 后或需要隔离请求流时清空持久预测记忆。
model.reset_recap_cache()
```

## 紧凑评测

输入为有序 JSONL，每行一个请求。`image` 路径相对 JSONL 所在目录。
记录可以使用 `prompt`，也可以使用处理器 chat template 接受的 `messages` 提供完整多轮历史。
每次生成都创建新的上下文记忆；显式提供历史不会复用前一次请求的 KV cache。

```json
{"id":"sample-1","image":"shapes.png","prompt":"Describe the image."}
```

小规模前台运行：

```bash
python -m recap evaluate \
  --model /path/to/Qwen2.5-VL-7B-Instruct \
  --device cuda:0 --backend cpp --local-files-only \
  --input examples/requests.jsonl --output outputs/eval/example \
  --max-new-tokens 64 --require-token-match
```

长评测使用文件 launcher 和命名 tmux session：

```bash
PYTHON_BIN=/path/to/python bash scripts/launch_recap.sh \
  --model /path/to/Qwen2.5-VL-7B-Instruct \
  --device cuda:0 --backend cpp --local-files-only \
  --input /path/to/requests.jsonl --output outputs/eval/my-run
```

launcher 会打印 session、日志和状态文件路径；日志在 `outputs/logs/eval/`，
名称以时间戳开头。用 `tmux attach -t <session>` 查看，用
`tmux kill-session -t <session>` 停止。正常完成的 `.status` 包含 `exit_code=0`。
同名评测结果目录不会被覆盖。

每个样本交替执行 target / recap 顺序，warmup 后清空预测记忆。输出：

- `results.jsonl`：接受长度、候选节点数、耗时、token 哈希和逐 token 一致性。
- `summary.json`：记录数、均值/中位数/P90 接受长度、speedup 和一致性比例。

评测默认不 decode、不保存生成文本；只有显式传入 `--save-decoded-output` 才保存文本。
`--require-token-match` 在输出不一致时以非零状态退出，同时保留结果供诊断。
speedup 为每个请求的 RECAP tokens/s 除以 target tokens/s。
接受长度不包含 root 和 correction；每次迭代产出量不包含 prefill 得到的首个 token。

计时包含图像编码、prefill、树构建、验证和提交，不包含输入预处理、模型加载、
KV 缓冲预分配或扩展编译。这是可运行性与配对性能检查入口，不能直接替代原论文
固定数据清单、请求顺序与计时协议。最小版每次重新编码图片，不保留旧评测器的跨轮图像特征缓存。
fp16/bf16 的 packed 与逐 token 矩阵计算可能产生舍入差异，因此记录实际 token 一致率。

## 保留的算法与命名

默认使用论文主配置：Top-8 预测行、最大深度 10、63 个 transition 候选节点，
后缀检索最多额外加入 4 个节点；root 不计入该预算。`fusion=prior_confidence`
按可用源的具体程度排序分配 `(0.70, 0.20, 0.10)` 先验，并乘以 `exp(-entropy)`。
`fusion=fixed` 保留固定先验对照。

- `RecapMemory`：U 跨请求保留，C/G 每个请求重新建立；prefill 只初始化 U。
- `recap_fuse()`：多粒度预测行的置信度融合。
- `build_recap_tree()`：Python / C++ best-first tree 与后缀路径合并。
- `RecapSuffixIndex`：只写入 prompt 和已提交 token 的后缀自动机。
- `RecapModel.recap_generate()`：整棵树一次验证，提交最长匹配路径和 target correction。
- `reset_recap_cache()`：清空持久记忆。

所有经过 target 验证的节点都更新预测记忆，包括最终未提交的节点。
同一次 target forward 中，同一个记忆 key 的首个预测行生效。
解码当前只支持无 padding、batch size 1、单设备、greedy，不接受 sampling 参数。
停止条件沿用原评测器：EOS/EOD 和 tokenizer 的全部 special token。
模型对象带有持久状态，请按顺序调用。

## 目录与验证

```text
src/recap/
  model.py          # RecapModel 和解码流程
  memory.py         # U/C/G 记忆与融合
  tree.py           # 树构建、topology packing 和匹配
  recap_tree.cpp    # 可选 C++ CPU hot path
  suffix.py         # 后缀自动机
  cache.py          # 可回退、可压缩的 KV cache
  backbones/        # Qwen/LLaVA 的树注意力适配
  cli.py            # 独立运行入口
  evaluation.py     # 紧凑指标与 summary
tests/              # 记忆作用域、两种树实现、缓存与微型模型一致性
examples/           # 两条请求和自包含示例图片
scripts/            # tmux launcher
```

```bash
mkdir -p outputs
RECAP_TEST_CPP=1 python -m pytest -q --basetemp=outputs/test-tmp
```

不开启 `RECAP_TEST_CPP=1` 时跳过编译器检查；其余测试在 CPU 上运行，不下载模型。
来源及许可证见 [NOTICE.md](NOTICE.md)。

2026-10-13 抽取验证：11 项测试通过；160 组树与 200 次后缀更新和原实现一致。
Qwen2.5-VL-7B、LLaVA-1.6-7B 各运行 1 条文本请求和 1 条图像请求，
最多生成 32 tokens，均与相同 target 的逐 token 解码一致。wheel 在隔离目录
安装后通过独立导入及 C++ 自检。这些是可运行性检查，不是完整性能复现实验。
