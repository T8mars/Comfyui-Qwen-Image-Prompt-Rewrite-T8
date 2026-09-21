# Qwen Image 2.1 Prompt Rewrite for ComfyUI

[ComfyUI Registry 节点 / Registry node](https://registry.comfy.org/zh/publishers/t8star/nodes/qwen-image-prompt-rewrite-t8) · [GGUF 模型镜像 / GGUF model mirror](https://huggingface.co/t8star/qwen-image-2.1-comfy)

一个入口处理纯文字文生图提示词和带 1–10 张参考图的编辑提示词。`auto` 根据是否连接图片选择 T2I 或 I2I 权重，分别使用 Qwen Image 2.1 官方系统提示词模板。节点运行本地 llama.cpp；默认每次改写结束即卸载模型，也可暂时保持加载并用独立卸载节点释放。

这里下载的 GGUF 是**提示词改写模型**。要根据改写结果实际生成图像，还需要在原有 ComfyUI 工作流里安装 Qwen Image 2.1 的图像生成模型、文本编码器和 VAE。

## 安装

1. 当前 Registry 版本仍待审核；先将[本仓库](https://github.com/T8mars/Comfyui-Qwen-Image-Prompt-Rewrite-T8)克隆或解压到 `ComfyUI/custom_nodes/Comfyui-Qwen-Image-Prompt-Rewrite-T8/`，然后重启 ComfyUI。待 Registry 状态转为 Active 后，也可在 ComfyUI Manager 搜索 `qwen-image-prompt-rewrite-t8` 安装，或运行 `comfy node install qwen-image-prompt-rewrite-t8`。
2. 在本目录运行 `python tools/download_models.py`。下载器将三个 Q4_K_M 主模型及 I2I 的 BF16 视觉组件放入 `models/llm/qwenimage-pe/`，对现有文件按大小和 SHA256 校验，对中断下载续传。视觉组件的发布仓库没有 Q4 文件。
3. Windows 上运行 `powershell -ExecutionPolicy Bypass -File tools/download_runtime.ps1`，获取官方 llama.cpp b11068 CUDA 12.4 发行包。其他平台可自行安装兼容的 `llama-server`，并将其可执行文件路径设置为环境变量 `QWEN_PE_LLAMA_SERVER`。
4. ComfyUI 的 Python 环境须已有 `torch`、`numpy`、`Pillow`。启动后在“Qwen Image 2.1 / Prompt Rewrite”分类查找节点。

也可从 [T8star 的 Hugging Face 镜像](https://huggingface.co/t8star/qwen-image-2.1-comfy)手动下载四个文件，保持原文件名放入上述目录；镜像 README 列出来源及 SHA256。模型未包含在 GitHub 或节点管理器安装包中。安装节点后仍需自行安装兼容的 `llama-server` 并配置运行路径。

模型目录也可使用 `ComfyUI/models/llm/qwenimage-pe/`，或以 `QWEN_PE_MODEL_DIR` 指向另一个本地目录。这里的任何 GGUF 主模型都会出现在两个模型下拉列表中；名字含 `mmproj` 的文件进入视觉模型列表。附加模型是否兼容所选任务需由用户自行确认，编辑模式缺少匹配的 mmproj 会明确报错。

## 节点与工作流

- **PE Rewrite T8**：输入文字、任务类型、画幅比例、输出语言、透明图开关、T2I 与编辑模型、视觉组件、卸载策略、种子，以及按顺序连接的 `image_1` 至 `image_10`。比例可选 `auto`、`1:1`、`1:2`、`2:3`、`3:4`、`4:5`、`16:9`、`21:9`、`9:21`、`5:4`、`4:3`、`2:1`；语言可选 `auto`、`中文`、`English`。`auto` 任务在无图时调用 T2I，有图时调用 I2I。必须从 `image_1` 连续连接，每个端口只接一张图，不接批次。输出改写文字、结构化 `PE_RESULT` 和诊断信息。
- **PE Canvas T8**：读取 `wh_ratio` 或 `ratio_follow`，输出宽、高、Qwen 64 通道空 latent 和比例来源。`ratio_follow=<imageN>` 在 `follow_input_size` 开启时沿用参考图尺寸；大图会保持比例缩至 `resolution²` 像素预算和 4096 像素边长上限。关闭 `follow_input_size` 后只沿用比例，并按同一预算计算画布。
- **PE Unload T8**：在 `keep_loaded` 模式下显式终止本节点启动的模型服务。
- **PE Local Models T8**：列出当前发现的主模型与视觉文件。`Qwen-PE-2.1-Local-Models-Demo.json` 已将它接到 `PreviewAny`，可在 UI 中检查。

`workflows/Qwen-PE-2.1-Text-to-Image-Ready.json` 是纯文字可运行示例；`workflows/Qwen-PE-2.1-Edit-2-Images-Demo.json` 和 `workflows/Qwen-PE-2.1-Edit-10-Images-Demo.json` 使用 ComfyUI `EmptyImage` 生成色彩图，可直接验证多图视觉链路；`workflows/Qwen-PE-2.1-Edit-2-Images-Ready.json` 使用两个 `LoadImage`，导入后请分别选择自己的图片。另外提供英文透明图、强制中文输出、中文透明图和保留加载后显式卸载示例。将 JSON 拖入 ComfyUI 画布，或放入 `user/default/workflows/` 后从工作流栏打开。

仓库维护者若要运行 `tools/build_workflows.py` 或 `tools/build_downstream_workflows.py` 重新生成这些 JSON，须先将环境变量 `QWEN_PE_COMFY_DIR` 指向本机 ComfyUI 目录；后者如未使用整合包，还需通过 `QWEN_PE_QWEN_TEMPLATE` 指定官方 Qwen Image 2.1 工作流模板。

目录还提供六个**完整出图工作流**：标准 T2I、Heretic T2I、双图编辑、十图编辑、英文透明 T2I 和中文透明 T2I，文件名均以 `Qwen-PE-2.1-Full-` 开头。它们把改写提示词和同一组参考图接入新版 ComfyUI 的 `TextEncodeQwenImage21`，将 PE Canvas 的 latent 接入采样器，再由 VAE 解码并保存 PNG；预览节点同时显示最终提示词、画幅来源和诊断。双图示例附带 `workflows/fixtures/` 中的两张测试图；导入别的 ComfyUI 时，请把图片复制到其 `input/` 目录或在两个 `LoadImage` 节点重新选图。十图示例用内置 `EmptyImage` 色块生成十张参考图，无需额外素材，两个模型节点接收完全相同的图序。完整出图工作流以 512 像素、12 步作为快速验收参数，正式出图可提高分辨率和采样步数。

完整出图工作流需要兼容 Qwen Image 2.1、带 `TextEncodeQwenImage21` 的新版 ComfyUI，以及另行安装的扩散模型、文本编码器和 VAE。当前本机验证组合为 ComfyUI `5ba116a40f1944f64e2e4a8ace826656e6293bf4`、`qwen_image_2.1_int8_convrot.safetensors`、`qwen3vl_8b_int8_convrot.safetensors` 与 `qwen_image_2.1_vae_bf16.safetensors`。这些权重属于下游出图模型，和本插件的三个 PE GGUF 模型不同，不随本项目附带。旧版 ComfyUI 0.34 能运行前述纯改写工作流，但缺少 `TextEncodeQwenImage21`，不能直接运行完整出图工作流。Canvas 的 64 通道、16 倍下采样形状已在新版节点中实测。

## 运行约束

默认 `after_run` 在任务完成或报错后退出自有 `llama-server`。`keep_loaded` 只缓存当前 profile；模型、视觉组件或上下文配置变化时先关闭旧进程。服务仅绑定 `127.0.0.1`。图片按插口顺序放在同一次多模态请求中，视觉副本限制为最多 1,048,576 像素且最长边不超过 4096 像素；原图尺寸留在结果里供画布使用。模型必须返回官方要求的严格 JSON 字段，否则节点报错，不会凭空补齐或静默忽略图片。

固定比例会覆盖模型选择的 `wh_ratio`/`ratio_follow` 并传给 Canvas，诊断中保留模型原始值。遵照官方模板，数字比例只放在比例字段里，不强塞到描述性提示词。明确选择 `中文` 或 `English` 时，节点会检查描述性文字中的汉字、拉丁字母和其他文字脚本，必要时触发本地翻译；同属拉丁字母的语言仍依赖模型遵守语言指令。`auto` 遵从模型输出，并以实际改写文字标记语言。图内需要精确呈现的引号内文字仍以用户要求为准。透明图开关会在实际输出语言的最终提示词前后加入 RGBA、alpha 通道和透明背景说明；能安全识别的白底、实体背景短语和白色留白会规范为透明背景/留白，仅背景独立句会被移除，含主体信息的冲突句则要求模型重写，并记录改动数量。它是提示词约束，实际输出是否具有 alpha 通道还取决于下游图像生成模型与解码工作流。

当原始指令的语言和所选输出语言不同，模型可能先按官方模板产生原语言改写；节点会用当前本地模型再做一次受约束的翻译并复验图片标签和引号内文字。诊断中的 `translation_fallback` 和 `translation_usage` 显示这一步是否发生，因此跨语言模式可能比自动语言模式慢。

测试与未满足的验收门槛记录在本地 `roadmap.md`。模型权重、运行时和该内部规划文档被 `.gitignore` 排除。

本项目内的官方系统提示词模板来自 [QwenLM/Qwen-Image-2.1](https://github.com/QwenLM/Qwen-Image-2.1/tree/main/prompt_rewrite/prompts)，遵守 [Qwen Research License Agreement](LICENSE-QWEN-RESEARCH)；见 [第三方材料说明](NOTICE.md)。模型镜像随附该许可和 NOTICE，权重仅获准用于非商业研究或评估；商业用途需另向 Qwen 权利人取得许可。此仓库未随附任何模型权重，也不代表本节点获得商业模型授权。

本地验收已覆盖 1–10 图改写真实推理，并在 ComfyUI 界面验证了纯改写的文生图、双图、十图、语言及透明选项、本地模型列表、显式卸载与中断。六份完整出图工作流也已在新版 ComfyUI 界面跑通：标准版和 Heretic 均生成苹果图；双图编辑把第一张的红苹果放入第二张的橙色场景，`ratio_follow=<image2>` 对应实际 512×384 输出；十图编辑输出 784×336 的十列色块，按参考图 1–10 顺序排列，最后一列为黄色；中英文透明版输出 PNG 的 alpha 最小值均为 0。中文透明版的最终提示词含指定 RGBA 前缀与 alpha 透明背景后缀。标准 T2I 和 Heretic 在同一组 12 条提示词样例中均通过严格 JSON 校验。20 轮模型切换及卸载检查没有发现残留自有进程。详细运行记录保存在忽略目录 `runtime/`，不会上传。

计划中的 60 场景批量评测及 24 例盲评已准备样例，但依用户要求停止继续出图，因此未运行，也不宣称通过质量门槛。

## T8star 链接

- [B站](https://space.bilibili.com/385085361)
- [YouTube](https://www.youtube.com/@T8star-Aix/)
- [API](https://api.seedance.nz/sign-up?aff=5f4w)
- [免费画廊](https://www.openzhenzhen.com)
- [在线 AI 应用](https://www.runninghub.ai/zh-cn/user-center/1907375370302308353/userPost?inviteCode=rh-v1121)
- [ComfyUI 整合包](https://pan.quark.cn/s/264edb7e36bd)
- [Hugging Face](https://huggingface.co/t8star)
