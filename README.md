# vllm_musa

> 主要更新：
> - 非常感谢踩坑文章[中国科学院大学GPU架构与编程大作业二 摩尔线程赛道  (MTT S4000) AUTODL部署与测试指南 - 求索者freedom的文章 - 知乎](https://zhuanlan.zhihu.com/p/1989072964000453732)，修复了提到的原仓库中的许多bug
> - 构建了适配AutoDL摩尔线程的vLLM轮子


摩尔线程致力于构建完善好用的国产GPU应用生态，自主研发了MUSA架构及软件平台。vllm项目是业界广泛使用的大语言模型的推理和服务引擎，使用CUDA/ROCm提供GPU加速能力。为了方便摩尔线程GPU用户使用vllm框架，我们发起vllm_musa开源项目为vllm提供MUSA加速，让用户可释放摩尔线程GPU的澎湃算力。

现有的vllm代码不支持摩尔线程GPU作为后端，因此我们新增了MUSA设备后端。vllm_musa接口与官方接口一致，用户无需改动业务代码，开箱即用。

MUSA的一大优势是CUDA兼容，通过musify工具，我们可以快速将官方代码porting至MUSA软件栈，用户可以根据文档自行升级vllm版本并适配MUSA软件栈。

## 支持模型列表⚠️
此版vllm最后更新时间为2024年四月，请注意不支持最新版的DeepSeek和Qwen3等系列模型，详细支持列表如下：

| Architecture | Models | Example HuggingFace Models | LoRA |
|-------------|--------|----------------------------|------|
| `AquilaForCausalLM` | Aquila | `BAAI/Aquila-7B`, `BAAI/AquilaChat-7B`, etc. | ✅ |
| `BaiChuanForCausalLM` | Baichuan | `baichuan-inc/Baichuan2-13B-Chat`, `baichuan-inc/Baichuan-7B`, etc. | ✅ |
| `ChatGLMModel` | ChatGLM | `THUDM/chatglm2-6b`, `THUDM/chatglm3-6b`, etc. | ✅ |
| `CohereForCausalLM` | Command-R | `CohereForAI/c4ai-command-r-v01`, etc. |  |
| `DbrxForCausalLM` | DBRX | `databricks/dbrx-base`, `databricks/dbrx-instruct`, etc. |  |
| `DeciLMForCausalLM` | DeciLM | `Deci/DeciLM-7B`, `Deci/DeciLM-7B-instruct`, etc. |  |
| `BloomForCausalLM` | BLOOM, BLOOMZ, BLOOMChat | `bigscience/bloom`, `bigscience/bloomz`, etc. |  |
| `FalconForCausalLM` | Falcon | `tiiuae/falcon-7b`, `tiiuae/falcon-40b`, `tiiuae/falcon-rw-7b`, etc. |  |
| `GemmaForCausalLM` | Gemma | `google/gemma-2b`, `google/gemma-7b`, etc. | ✅ |
| `GPT2LMHeadModel` | GPT-2 | `gpt2`, `gpt2-xl`, etc. |  |
| `GPTBigCodeForCausalLM` | StarCoder, SantaCoder, WizardCoder | `bigcode/starcoder`, `bigcode/gpt_bigcode-santacoder`, `WizardLM/WizardCoder-15B-V1.0`, etc. |  |
| `GPTJForCausalLM` | GPT-J | `EleutherAI/gpt-j-6b`, `nomic-ai/gpt4all-j`, etc. |  |
| `GPTNeoXForCausalLM` | GPT-NeoX, Pythia, OpenAssistant, Dolly V2, StableLM | `EleutherAI/gpt-neox-20b`, `EleutherAI/pythia-12b`, `OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5`, `databricks/dolly-v2-12b`, `stabilityai/stablelm-tuned-alpha-7b`, etc. |  |
| `InternLMForCausalLM` | InternLM | `internlm/internlm-7b`, `internlm/internlm-chat-7b`, etc. | ✅ |
| `InternLM2ForCausalLM` | InternLM2 | `internlm/internlm2-7b`, `internlm/internlm2-chat-7b`, etc. |  |
| `JAISLMHeadModel` | Jais | `core42/jais-13b`, `core42/jais-13b-chat`, `core42/jais-30b-v3`, `core42/jais-30b-chat-v3`, etc. |  |
| `LlamaForCausalLM` | LLaMA, Llama 2, Meta Llama 3, Vicuna, Alpaca, Yi | `meta-llama/Meta-Llama-3-8B-Instruct`, `meta-llama/Meta-Llama-3-70B-Instruct`, `meta-llama/Llama-2-13b-hf`, `meta-llama/Llama-2-70b-hf`, `openlm-research/open_llama_13b`, `lmsys/vicuna-13b-v1.3`, `01-ai/Yi-6B`, `01-ai/Yi-34B`, etc. | ✅ |
| `MiniCPMForCausalLM` | MiniCPM | `openbmb/MiniCPM-2B-sft-bf16`, `openbmb/MiniCPM-2B-dpo-bf16`, etc. |  |
| `MistralForCausalLM` | Mistral, Mistral-Instruct | `mistralai/Mistral-7B-v0.1`, `mistralai/Mistral-7B-Instruct-v0.1`, etc. | ✅ |
| `MixtralForCausalLM` | Mixtral-8x7B, Mixtral-8x7B-Instruct | `mistralai/Mixtral-8x7B-v0.1`, `mistralai/Mixtral-8x7B-Instruct-v0.1`, `mistral-community/Mixtral-8x22B-v0.1`, etc. | ✅ |
| `MPTForCausalLM` | MPT, MPT-Instruct, MPT-Chat, MPT-StoryWriter | `mosaicml/mpt-7b`, `mosaicml/mpt-7b-storywriter`, `mosaicml/mpt-30b`, etc. |  |
| `OLMoForCausalLM` | OLMo | `allenai/OLMo-1B-hf`, `allenai/OLMo-7B-hf`, etc. |  |
| `OPTForCausalLM` | OPT, OPT-IML | `facebook/opt-66b`, `facebook/opt-iml-max-30b`, etc. |  |
| `OrionForCausalLM` | Orion | `OrionStarAI/Orion-14B-Base`, `OrionStarAI/Orion-14B-Chat`, etc. |  |
| `PhiForCausalLM` | Phi | `microsoft/phi-1_5`, `microsoft/phi-2`, etc. |  |
| `Phi3ForCausalLM` | Phi-3 | `microsoft/Phi-3-mini-4k-instruct`, `microsoft/Phi-3-mini-128k-instruct`, etc. |  |
| `QWenLMHeadModel` | Qwen | `Qwen/Qwen-7B`, `Qwen/Qwen-7B-Chat`, etc. |  |
| `Qwen2ForCausalLM` | Qwen2/Qwen2.5 | `Qwen/Qwen2-beta-7B`, `Qwen/Qwen2-beta-7B-Chat`, `Qwen/Qwen2.5-7B`, `Qwen/Qwen2.5-7B-Instruct`, etc. | ✅ |
| `Qwen2MoeForCausalLM` | Qwen2MoE | `Qwen/Qwen1.5-MoE-A2.7B`, `Qwen/Qwen1.5-MoE-A2.7B-Chat`, etc. |  |
| `StableLmForCausalLM` | StableLM | `stabilityai/stablelm-3b-4e1t/`, `stabilityai/stablelm-base-alpha-7b-v2`, etc. |  |

## 依赖

- musa_toolkit >= dev3.0.0
- pytorch == v2.2.0
- [torch_musa](https://github.com/MooreThreads/torch_musa) >= v1.3.0
- triton >= v2.2.0
- ray >= 2.9
- vllm v0.4.2

## 安装
### 直接安装
从Release中下载编译好的wheel文件直接安装。已经在AutoDL摩尔线程专区MTT S4000机器上验证过，其余环境不保证可用性，请优先使用手动编译的方式安装。

```
pip install vllm-0.4.2+musa-cp310-cp310-linux_x86_64.whl
```
如果已经安装过没打补丁的vllm，可以用下面的命令覆盖安装
```
pip install --force-reinstall --no-deps vllm-0.4.2+musa-cp310-cp310-linux_x86_64.whl
```
### 手动编译
```
bash build_musa.sh
```
## 测试示例
```
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, LlamaForCausalLM
import transformers
import time
import torch
import torch_musa


model_path = <path_to_llm_model>

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
llm = LLM(model=model_path, dtype="float16",trust_remote_code=True, device="musa")

outputs = llm.generate(prompts, sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

```

## Porting

当前仓库porting自vllm v0.4.2版本。如果用户希望使用更高版本的vllm，只需要运行`musa_porting.py`将原生CUDA代码适配到MUSA代码即可。当然随着vllm的迭代可能会有些代码成为漏网之鱼，没有porting成功，用户可自行修改`musa_porting.py`文件中的文本替换规则。从而发挥MUSA强大的CUDA兼容能力。

### 步骤
1. 运行 `python musa_porting.py`
2. 将`CMakeLists.txt`中需要编译的文件后缀从`.cu`修改为`.mu`
3. 编译运行vllm_musa

## 贡献

欢迎广大用户及开发者使用、反馈，助力vllm_musa功能及性能持续完善。

社区共建，期待广大开发者与我们一道，共同打造MUSA软件生态。我们将陆续推出一系列开源软件MUSA加速项目。

<details>
  <summary>VLLM原始README</summary>
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/source/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/source/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

<h3 align="center">
Easy, fast, and cheap LLM serving for everyone
</h3>

<p align="center">
| <a href="https://docs.vllm.ai"><b>Documentation</b></a> | <a href="https://vllm.ai"><b>Blog</b></a> | <a href="https://arxiv.org/abs/2309.06180"><b>Paper</b></a> | <a href="https://discord.gg/jz7wjKhh6g"><b>Discord</b></a> |

</p>

*Latest News* 🔥
- [2024/04] We hosted [the third vLLM meetup](https://robloxandvllmmeetup2024.splashthat.com/) with Roblox! Please find the meetup slides [here](https://docs.google.com/presentation/d/1A--47JAK4BJ39t954HyTkvtfwn0fkqtsL8NGFuslReM/edit?usp=sharing).
- [2024/01] We hosted [the second vLLM meetup](https://lu.ma/ygxbpzhl) in SF! Please find the meetup slides [here](https://docs.google.com/presentation/d/12mI2sKABnUw5RBWXDYY-HtHth4iMSNcEoQ10jDQbxgA/edit?usp=sharing).
- [2024/01] Added ROCm 6.0 support to vLLM.
- [2023/12] Added ROCm 5.7 support to vLLM.
- [2023/10] We hosted [the first vLLM meetup](https://lu.ma/first-vllm-meetup) in SF! Please find the meetup slides [here](https://docs.google.com/presentation/d/1QL-XPFXiFpDBh86DbEegFXBXFXjix4v032GhShbKf3s/edit?usp=sharing).
- [2023/09] We created our [Discord server](https://discord.gg/jz7wjKhh6g)! Join us to discuss vLLM and LLM serving! We will also post the latest announcements and updates there.
- [2023/09] We released our [PagedAttention paper](https://arxiv.org/abs/2309.06180) on arXiv!
- [2023/08] We would like to express our sincere gratitude to [Andreessen Horowitz](https://a16z.com/2023/08/30/supporting-the-open-source-ai-community/) (a16z) for providing a generous grant to support the open-source development and research of vLLM.
- [2023/07] Added support for LLaMA-2! You can run and serve 7B/13B/70B LLaMA-2s on vLLM with a single command!
- [2023/06] Serving vLLM On any Cloud with SkyPilot. Check out a 1-click [example](https://github.com/skypilot-org/skypilot/blob/master/llm/vllm) to start the vLLM demo, and the [blog post](https://blog.skypilot.co/serving-llm-24x-faster-on-the-cloud-with-vllm-and-skypilot/) for the story behind vLLM development on the clouds.
- [2023/06] We officially released vLLM! FastChat-vLLM integration has powered [LMSYS Vicuna and Chatbot Arena](https://chat.lmsys.org) since mid-April. Check out our [blog post](https://vllm.ai).

---
## About
vLLM is a fast and easy-to-use library for LLM inference and serving.

vLLM is fast with:

- State-of-the-art serving throughput
- Efficient management of attention key and value memory with **PagedAttention**
- Continuous batching of incoming requests
- Fast model execution with CUDA/HIP graph
- Quantization: [GPTQ](https://arxiv.org/abs/2210.17323), [AWQ](https://arxiv.org/abs/2306.00978), [SqueezeLLM](https://arxiv.org/abs/2306.07629), FP8 KV Cache
- Optimized CUDA kernels

vLLM is flexible and easy to use with:

- Seamless integration with popular Hugging Face models
- High-throughput serving with various decoding algorithms, including *parallel sampling*, *beam search*, and more
- Tensor parallelism support for distributed inference
- Streaming outputs
- OpenAI-compatible API server
- Support NVIDIA GPUs and AMD GPUs
- (Experimental) Prefix caching support
- (Experimental) Multi-lora support

vLLM seamlessly supports many Hugging Face models, including the following architectures:

- Aquila & Aquila2 (`BAAI/AquilaChat2-7B`, `BAAI/AquilaChat2-34B`, `BAAI/Aquila-7B`, `BAAI/AquilaChat-7B`, etc.)
- Baichuan & Baichuan2 (`baichuan-inc/Baichuan2-13B-Chat`, `baichuan-inc/Baichuan-7B`, etc.)
- BLOOM (`bigscience/bloom`, `bigscience/bloomz`, etc.)
- ChatGLM (`THUDM/chatglm2-6b`, `THUDM/chatglm3-6b`, etc.)
- Command-R (`CohereForAI/c4ai-command-r-v01`, etc.)
- DBRX (`databricks/dbrx-base`, `databricks/dbrx-instruct` etc.)
- DeciLM (`Deci/DeciLM-7B`, `Deci/DeciLM-7B-instruct`, etc.)
- Falcon (`tiiuae/falcon-7b`, `tiiuae/falcon-40b`, `tiiuae/falcon-rw-7b`, etc.)
- Gemma (`google/gemma-2b`, `google/gemma-7b`, etc.)
- GPT-2 (`gpt2`, `gpt2-xl`, etc.)
- GPT BigCode (`bigcode/starcoder`, `bigcode/gpt_bigcode-santacoder`, etc.)
- GPT-J (`EleutherAI/gpt-j-6b`, `nomic-ai/gpt4all-j`, etc.)
- GPT-NeoX (`EleutherAI/gpt-neox-20b`, `databricks/dolly-v2-12b`, `stabilityai/stablelm-tuned-alpha-7b`, etc.)
- InternLM (`internlm/internlm-7b`, `internlm/internlm-chat-7b`, etc.)
- InternLM2 (`internlm/internlm2-7b`, `internlm/internlm2-chat-7b`, etc.)
- Jais (`core42/jais-13b`, `core42/jais-13b-chat`, `core42/jais-30b-v3`, `core42/jais-30b-chat-v3`, etc.)
- LLaMA, Llama 2, and Meta Llama 3 (`meta-llama/Meta-Llama-3-8B-Instruct`, `meta-llama/Meta-Llama-3-70B-Instruct`, `meta-llama/Llama-2-70b-hf`, `lmsys/vicuna-13b-v1.3`, `young-geng/koala`, `openlm-research/open_llama_13b`, etc.)
- MiniCPM (`openbmb/MiniCPM-2B-sft-bf16`, `openbmb/MiniCPM-2B-dpo-bf16`, etc.)
- Mistral (`mistralai/Mistral-7B-v0.1`, `mistralai/Mistral-7B-Instruct-v0.1`, etc.)
- Mixtral (`mistralai/Mixtral-8x7B-v0.1`, `mistralai/Mixtral-8x7B-Instruct-v0.1`, `mistral-community/Mixtral-8x22B-v0.1`, etc.)
- MPT (`mosaicml/mpt-7b`, `mosaicml/mpt-30b`, etc.)
- OLMo (`allenai/OLMo-1B-hf`, `allenai/OLMo-7B-hf`, etc.)
- OPT (`facebook/opt-66b`, `facebook/opt-iml-max-30b`, etc.)
- Orion (`OrionStarAI/Orion-14B-Base`, `OrionStarAI/Orion-14B-Chat`, etc.)
- Phi (`microsoft/phi-1_5`, `microsoft/phi-2`, etc.)
- Phi-3 (`microsoft/Phi-3-mini-4k-instruct`, `microsoft/Phi-3-mini-128k-instruct`, etc.)
- Qwen (`Qwen/Qwen-7B`, `Qwen/Qwen-7B-Chat`, etc.)
- Qwen2 (`Qwen/Qwen1.5-7B`, `Qwen/Qwen1.5-7B-Chat`, etc.)
- Qwen2MoE (`Qwen/Qwen1.5-MoE-A2.7B`, `Qwen/Qwen1.5-MoE-A2.7B-Chat`, etc.)
- StableLM(`stabilityai/stablelm-3b-4e1t`, `stabilityai/stablelm-base-alpha-7b-v2`, etc.)
- Starcoder2(`bigcode/starcoder2-3b`, `bigcode/starcoder2-7b`, `bigcode/starcoder2-15b`, etc.)
- Xverse (`xverse/XVERSE-7B-Chat`, `xverse/XVERSE-13B-Chat`, `xverse/XVERSE-65B-Chat`, etc.)
- Yi (`01-ai/Yi-6B`, `01-ai/Yi-34B`, etc.)

Install vLLM with pip or [from source](https://vllm.readthedocs.io/en/latest/getting_started/installation.html#build-from-source):

```bash
pip install vllm
```

## Getting Started

Visit our [documentation](https://vllm.readthedocs.io/en/latest/) to get started.
- [Installation](https://vllm.readthedocs.io/en/latest/getting_started/installation.html)
- [Quickstart](https://vllm.readthedocs.io/en/latest/getting_started/quickstart.html)
- [Supported Models](https://vllm.readthedocs.io/en/latest/models/supported_models.html)

## Contributing

We welcome and value any contributions and collaborations.
Please check out [CONTRIBUTING.md](./CONTRIBUTING.md) for how to get involved.

## Citation

If you use vLLM for your research, please cite our [paper](https://arxiv.org/abs/2309.06180):
```bibtex
@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Lianmin Zheng and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang and Ion Stoica},
  booktitle={Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles},
  year={2023}
}
```

</details>