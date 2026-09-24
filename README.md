# Qwen Image 2.1 Prompt Rewrite T8

**语言 / Language：简体中文 · [English](README.en.md)**

适用于 ComfyUI 的本地提示词改写节点。输入文字，或文字加 1–10 张参考图，节点会调用对应的 Qwen Image 2.1 PE GGUF 模型，输出适合文生图或图像编辑的提示词。支持中文/英文、指定画幅、透明背景提示词和模型卸载。

> **模型分工：**本项目的 GGUF 只负责**改写提示词**。实际出图还需要 Qwen Image 2.1 的扩散模型、文本编码器和 VAE；它们不包含在本节点中。

[GGUF 模型镜像](https://huggingface.co/t8star/qwen-image-2.1-comfy) · [ComfyUI Registry 页面](https://registry.comfy.org/zh/publishers/t8star/nodes/qwen-image-prompt-rewrite-t8) · [示例工作流](workflows/) · [反馈问题](https://github.com/T8mars/Comfyui-Qwen-Image-Prompt-Rewrite-T8/issues/new)

## 快速安装

1. 将仓库放到 `ComfyUI/custom_nodes/Comfyui-Qwen-Image-Prompt-Rewrite-T8/`，然后重启 ComfyUI：

   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/T8mars/Comfyui-Qwen-Image-Prompt-Rewrite-T8.git
   cd Comfyui-Qwen-Image-Prompt-Rewrite-T8
   ```

2. 在节点目录中，用 **ComfyUI 所使用的 Python** 下载四个 Q4_K_M 主模型和两个 I2I 视觉组件：

   ```bash
   python tools/download_models.py
   ```

   下载器会校验文件大小和 SHA256，并支持断点续传。也可以从[基础模型镜像](https://huggingface.co/t8star/qwen-image-2.1-comfy)与 [Heretic I2I 来源仓库](https://huggingface.co/pottokao/Qwen-Image-2.1-PE-I2I-Heretic-GGUF)手动下载以下六个文件，保持文件名不变，放入 `models/llm/qwenimage-pe/`：

   | 用途 | 文件 |
   | --- | --- |
   | 文生图 | `Qwen-Image-2.1-PE-T2I.Q4_K_M.gguf` |
   | 图像编辑 | `Qwen-Image-2.1-PE-I2I.Q4_K_M.gguf` |
   | 图像编辑的视觉组件 | `Qwen-Image-2.1-PE-I2I.mmproj-bf16.gguf` |
   | 可选的 Heretic 文生图模型 | `pe_t2i_heretic-Q4_K_M.gguf` |
   | 可选的 Heretic 图像编辑模型 | `pe_i2i_heretic-Q4_K_M.gguf` |
   | Heretic 图像编辑的视觉组件 | `pe_i2i_heretic.mmproj-bf16.gguf` |

   视觉组件的来源仓库没有 Q4 版本，因此使用 BF16 文件。模型目录也可以是 `ComfyUI/models/llm/qwenimage-pe/`，或通过 `QWEN_PE_MODEL_DIR` 指定。模型权重不随 GitHub 源码或 Registry 安装包提供。

3. 安装兼容的 `llama-server`。Windows 可在节点目录运行：

   ```powershell
   powershell -ExecutionPolicy Bypass -File tools/download_runtime.ps1
   ```

   该脚本获取 llama.cpp 的 Windows CUDA 12.4 运行包。其他平台请自行安装兼容的 `llama-server`，并用 `QWEN_PE_LLAMA_SERVER` 指向其可执行文件。ComfyUI 的 Python 环境需有 `torch`、`numpy`、`Pillow`。

启动后，在 **Qwen Image 2.1 / Prompt Rewrite** 分类中查找节点。Registry 的发布状态可能变化；如果 Manager 尚未提供安装入口，请使用上面的仓库安装方式。

## 使用节点

| 节点 | 用途 |
| --- | --- |
| **PE Rewrite T8** | 改写提示词，输出文字、`PE_RESULT` 和诊断 JSON。 |
| **PE Canvas T8** | 根据模型比例或参考图尺寸生成宽、高和 Qwen Image 2.1 空 latent。 |
| **PE Unload T8** | 显式卸载由本节点启动、处于 `keep_loaded` 模式的模型。 |
| **PE Local Models T8** | 列出本机发现的 GGUF 主模型和视觉组件。 |

在 **PE Rewrite T8** 中输入 `user_prompt`，然后选择：

- `task`：`auto` 根据**实际有图**的输入选择文生图或编辑；`t2i` 只接受文字；`edit` 至少需要一张图。
- `t2i_model` / `edit_model`：分别用于文生图与编辑。可在 `edit_model` 中选择 `pe_i2i_heretic-Q4_K_M.gguf`；其配套视觉组件会在 `vision_model=Auto` 时自动匹配。
- `aspect_ratio`：`auto` 或 `1:1`、`1:2`、`2:3`、`3:4`、`4:5`、`16:9`、`9:16`、`21:9`、`9:21`、`5:4`、`4:3`、`2:1`。指定比例会覆盖模型建议，并传给 PE Canvas。
- `output_language`：`auto`、`中文`、`English`。指定语言时，节点会检查改写结果；必要时使用当前本地模型翻译，图内明确要求的引号文字保持原样。
- `transparent_rgba`：在最终提示词中加入 RGBA、alpha 通道和透明背景要求。**它控制提示词，不保证下游生成的图片一定带 alpha 通道。**
- `model_lifetime`：`after_run` 在每次完成或报错后卸载；`keep_loaded` 便于连续调用，之后可接 **PE Unload T8**。

`image_1` 至 `image_10` 每个端口只接一张 IMAGE，不接受批次。输出为 `None` 的端口会跳过，剩余图片按端口顺序连续编号。例如 `image_1` 有图、`image_2` 为 `None`、`image_9` 有图时，前一张是 `<image1>`，后一张是 `<image2>`；`PE_RESULT.image_input_ports` 记录它们原本来自哪些端口。单图编辑可自然描述或使用 `<image1>`；多图编辑的改写结果必须包含全部 `<image1>` 至 `<imageN>`。连接下游 `TextEncodeQwenImage21` 时，务必给它同一批图片、同一顺序。

## 示例工作流

把 [workflows/](workflows/) 中的 JSON 拖入 ComfyUI，或放进 ComfyUI 的工作流目录。建议从以下文件开始：

| 场景 | 工作流 | 额外要求 |
| --- | --- | --- |
| 只改写文生图提示词 | [Text-to-Image-Ready](workflows/Qwen-PE-2.1-Text-to-Image-Ready.json) | PE GGUF 与 llama-server |
| 两图编辑提示词 | [Edit-2-Images-Ready](workflows/Qwen-PE-2.1-Edit-2-Images-Ready.json) | 导入后在两个 `LoadImage` 节点选择图片 |
| 完整文生图 | [Full-T2I](workflows/Qwen-PE-2.1-Full-T2I.json) | 另装扩散模型、文本编码器和 VAE |
| 完整双图编辑 | [Full-Edit-2-Images](workflows/Qwen-PE-2.1-Full-Edit-2-Images.json) | 同上；重新选择参考图片 |
| 完整十图编辑 | [Full-Edit-10-Images](workflows/Qwen-PE-2.1-Full-Edit-10-Images.json) | 同上；示例用内置色块图 |

目录中还有 Heretic 文生图、中英文透明图、中文输出、模型列表及显式卸载示例。完整出图工作流依赖带 `TextEncodeQwenImage21` 的新版 ComfyUI，以及另行安装的 [Qwen Image 2.1 出图模型](https://huggingface.co/Comfy-Org/Qwen-Image-2.1)。双图完整示例附带 [fixtures](workflows/fixtures/)；在其他 ComfyUI 安装中，需要把图片复制到其 `input/` 目录或重新在 `LoadImage` 中选择。示例中的 512 像素、12 步用于快速运行，可按实际需求调整。

## 常见问题

| 现象 | 检查方法 |
| --- | --- |
| 模型列表为空或提示找不到 GGUF | 确认模型文件名和目录；可用 **PE Local Models T8** 查看扫描结果。 |
| 编辑模式提示缺少 mmproj | 确认 I2I 主模型和视觉文件同目录，或在 `vision_model` 中选择匹配文件。 |
| `model response reached the generation or context limit` | 节点会关闭思考重试一次；仍失败时查看错误中的 token 用量，减少输入图片或缩短原始要求。多图模式会占用更多显存。 |
| `model failed format validation` | 模型两次都未返回合规的 JSON、比例、语言或图片编号；查看报错详情。成功运行时可用 `diagnostics` 核对任务及图片端口映射。 |
| 找不到 `TextEncodeQwenImage21` | 更新到支持 Qwen Image 2.1 的 ComfyUI；纯改写示例不需要该节点。 |
| 透明图最终仍是不透明 PNG | 检查下游图像模型、VAE 和保存流程；PE 节点只生成透明图提示词。 |

本地 `llama-server` 只监听 `127.0.0.1`。输入图会缩小为最多约 100 万像素、最长边 4096 像素的视觉副本；PE Canvas 仍使用原图尺寸。模型返回内容经过 JSON、图片引用、语言及模式校验，失败时会重试一次并报告原因。

## 可选：Viggle Turbo 4 步 LoRA

[下载 ComfyUI 转换版 LoRA](https://huggingface.co/t8star/Qwen-Image-2.1-viggle-turbo-4step-r64-comfy/resolve/main/Qwen-Image-2.1-viggle-turbo-4step-r64-comfyui-T8.safetensors?download=true)。这是**出图扩散模型**使用的 LoRA，和上面的 PE GGUF 分开安装；本仓库只转换了 [Viggle 原始 LoRA](https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo) 的 ComfyUI 键名，没有重新训练。将文件放入 `ComfyUI/models/loras/`，在基础模型后接 `LoraLoaderModelOnly`，`strength_model=1.0`。依照原模型说明使用 **4 步、CFG 1.0、空负面提示词**；上面完整工作流的 12 步是基础模型示例，使用该 LoRA 时需调整。转换与校验信息见 [Hugging Face 模型卡](https://huggingface.co/t8star/Qwen-Image-2.1-viggle-turbo-4step-r64-comfy)。该转换版尚未完成独立的完整出图验收。

## 来源与许可

系统提示词模板来自 [Qwen Image 2.1 官方仓库](https://github.com/QwenLM/Qwen-Image-2.1/tree/main/prompt_rewrite/prompts)。模板、模型和可选 LoRA 的来源及修改说明见 [NOTICE.md](NOTICE.md)；相关模型受 [Qwen Research License Agreement](LICENSE-QWEN-RESEARCH) 约束，仅限非商业研究或评估，商用需向权利人另行取得许可。

## T8star

[B站](https://space.bilibili.com/385085361) · [YouTube](https://www.youtube.com/@T8star-Aix/) · [API](https://api.seedance.nz/sign-up?aff=5f4w) · [免费画廊](https://www.openzhenzhen.com) · [在线 AI 应用](https://www.runninghub.ai/zh-cn/user-center/1907375370302308353/userPost?inviteCode=rh-v1121) · [ComfyUI 整合包](https://pan.quark.cn/s/264edb7e36bd) · [Hugging Face](https://huggingface.co/t8star)
