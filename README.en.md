# Qwen Image 2.1 Prompt Rewrite T8

**Language / 语言: [简体中文](README.md) · English**

A local prompt rewriting node for ComfyUI. Give it text alone, or text plus 1–10 reference images. It selects the corresponding Qwen Image 2.1 PE GGUF model and returns a prompt for text-to-image generation or image editing. It supports Chinese and English output, aspect ratios, transparent-background instructions, and model unloading.

> **What the models do:** The GGUF files in this project **rewrite prompts**. To generate images, your ComfyUI workflow also needs a Qwen Image 2.1 diffusion model, text encoder, and VAE. Those are separate downloads.

[GGUF model mirror](https://huggingface.co/t8star/qwen-image-2.1-comfy) · [ComfyUI Registry page](https://registry.comfy.org/zh/publishers/t8star/nodes/qwen-image-prompt-rewrite-t8) · [Example workflows](workflows/) · [Report an issue](https://github.com/T8mars/Comfyui-Qwen-Image-Prompt-Rewrite-T8/issues/new)

## Quick setup

1. Place the repository under `ComfyUI/custom_nodes/Comfyui-Qwen-Image-Prompt-Rewrite-T8/`, then restart ComfyUI:

   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/T8mars/Comfyui-Qwen-Image-Prompt-Rewrite-T8.git
   cd Comfyui-Qwen-Image-Prompt-Rewrite-T8
   ```

2. From the node directory, use the **same Python environment as ComfyUI** to download three Q4_K_M main models and the I2I vision component:

   ```bash
   python tools/download_models.py
   ```

   The downloader verifies file sizes and SHA256 hashes and resumes interrupted downloads. You can also download the four files from the [model mirror](https://huggingface.co/t8star/qwen-image-2.1-comfy) and place them, with their original names, in `models/llm/qwenimage-pe/`:

   | Purpose | File |
   | --- | --- |
   | Text-to-image | `Qwen-Image-2.1-PE-T2I.Q4_K_M.gguf` |
   | Image editing | `Qwen-Image-2.1-PE-I2I.Q4_K_M.gguf` |
   | Vision component for editing | `Qwen-Image-2.1-PE-I2I.mmproj-bf16.gguf` |
   | Optional Heretic text-to-image model | `pe_t2i_heretic-Q4_K_M.gguf` |

   The source repository does not provide a Q4 vision component, so this project uses the BF16 file. You may also use `ComfyUI/models/llm/qwenimage-pe/` or set `QWEN_PE_MODEL_DIR` to a custom model directory. Model weights are not bundled with the GitHub source or Registry package.

3. Install a compatible `llama-server`. On Windows, run this from the node directory:

   ```powershell
   powershell -ExecutionPolicy Bypass -File tools/download_runtime.ps1
   ```

   This script downloads a llama.cpp Windows CUDA 12.4 runtime. On other platforms, install a compatible `llama-server` yourself and set `QWEN_PE_LLAMA_SERVER` to its executable path. ComfyUI's Python environment must contain `torch`, `numpy`, and `Pillow`.

After restarting ComfyUI, find the nodes under **Qwen Image 2.1 / Prompt Rewrite**. Registry availability may change; use the repository installation steps above if the node is not offered in Manager.

## Using the nodes

| Node | Purpose |
| --- | --- |
| **PE Rewrite T8** | Rewrite a prompt and return text, a structured `PE_RESULT`, and diagnostics JSON. |
| **PE Canvas T8** | Turn the model's aspect ratio or a reference image's dimensions into width, height, and an empty Qwen Image 2.1 latent. |
| **PE Unload T8** | Explicitly stop a model retained with `keep_loaded`. |
| **PE Local Models T8** | List discovered main GGUF models and vision components. |

Enter your instruction in `user_prompt` and choose the following options in **PE Rewrite T8**:

- `task`: `auto` selects text-to-image or editing from the images **actually present**. `t2i` accepts text only; `edit` requires at least one image.
- `t2i_model` / `edit_model`: Separate model choices for the two tasks. Editing also needs a matching `vision_model`; `Auto` normally finds it.
- `aspect_ratio`: `auto`, `1:1`, `1:2`, `2:3`, `3:4`, `4:5`, `16:9`, `9:16`, `21:9`, `9:21`, `5:4`, `4:3`, or `2:1`. An explicit choice overrides the model's suggestion and feeds PE Canvas.
- `output_language`: `auto`, `中文`, or `English`. With an explicit language, the node checks the rewrite and, if needed, translates it with the local model. Exact text requested inside quotation marks is preserved.
- `transparent_rgba`: Adds RGBA, alpha-channel, and transparent-background instructions to the final prompt. **This constrains the prompt; it does not guarantee that the downstream image has an alpha channel.**
- `model_lifetime`: `after_run` unloads after success or failure. `keep_loaded` speeds up repeated calls; use **PE Unload T8** when finished.

Each `image_1` through `image_10` input accepts one IMAGE, not a batch. Inputs returning `None` are skipped; remaining images are numbered in port order. For example, if `image_1` has an image, `image_2` is `None`, and `image_9` has an image, their prompt references are `<image1>` and `<image2>`. `PE_RESULT.image_input_ports` records the original ports. A single-image rewrite may describe the image naturally or use `<image1>`; a multi-image rewrite must include every tag from `<image1>` through `<imageN>`. If you connect the output to `TextEncodeQwenImage21`, give that node the same images in the same order.

## Example workflows

Drag a JSON file from [workflows/](workflows/) onto the ComfyUI canvas, or place it in your ComfyUI workflows directory. Start with one of these:

| Scenario | Workflow | Extra requirement |
| --- | --- | --- |
| Rewrite a text-to-image prompt | [Text-to-Image-Ready](workflows/Qwen-PE-2.1-Text-to-Image-Ready.json) | PE GGUF and llama-server |
| Rewrite a two-image edit | [Edit-2-Images-Ready](workflows/Qwen-PE-2.1-Edit-2-Images-Ready.json) | Choose files in both `LoadImage` nodes |
| Generate a complete text-to-image result | [Full-T2I](workflows/Qwen-PE-2.1-Full-T2I.json) | Separate diffusion model, text encoder, and VAE |
| Generate a complete two-image edit | [Full-Edit-2-Images](workflows/Qwen-PE-2.1-Full-Edit-2-Images.json) | Same; reselect the reference images |
| Generate a complete ten-image edit | [Full-Edit-10-Images](workflows/Qwen-PE-2.1-Full-Edit-10-Images.json) | Same; example uses built-in color images |

The directory also has Heretic text-to-image, Chinese and English transparency, Chinese output, model listing, and explicit unload examples. Full generation workflows require a recent ComfyUI with `TextEncodeQwenImage21` and the separate [Qwen Image 2.1 generation models](https://huggingface.co/Comfy-Org/Qwen-Image-2.1). The full two-image example includes [fixtures](workflows/fixtures/); when importing it into another ComfyUI installation, copy the images to its `input/` directory or reselect them in `LoadImage`. The included 512-pixel, 12-step settings are quick-run examples.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| Empty model list or GGUF not found | Check the filenames and model directory. **PE Local Models T8** shows what the node discovered. |
| Editing reports a missing mmproj | Put the matching I2I main model and vision file together, or choose the matching `vision_model`. |
| `model response reached the generation or context limit` | The node retries once without thinking. If it still fails, inspect the token counts in the error, use fewer images, or shorten the instruction. Multi-image jobs use more VRAM. |
| `model failed format validation` | Both attempts failed a JSON, ratio, language, or image-reference check. Read the error details. After a successful run, `diagnostics` can confirm the task and input-port mapping. |
| `TextEncodeQwenImage21` is missing | Update to a ComfyUI release supporting Qwen Image 2.1. Prompt-only workflows do not need this node. |
| The output PNG is still opaque | Check the downstream image model, VAE, and saving workflow. This node only writes the transparency prompt. |

The local `llama-server` listens on `127.0.0.1` only. A visual copy of each input is scaled to at most about one million pixels and a 4096-pixel longest side; PE Canvas retains the original dimensions. The response is checked for JSON structure, image references, language, and mode constraints, with one retry on validation failure.

## Optional: Viggle Turbo 4-step LoRA

[Download the ComfyUI-converted LoRA](https://github.com/T8mars/Comfyui-Qwen-Image-Prompt-Rewrite-T8/releases/download/viggle-turbo-comfyui-r64-v0.1/Qwen-Image-2.1-viggle-turbo-4step-r64-comfyui-T8.safetensors). This LoRA modifies the **image-generation diffusion model** and is installed separately from the PE GGUF files. This repository only converted the key names of [Viggle's original LoRA](https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo); it did not retrain the weights. Put the file in `ComfyUI/models/loras/`, add `LoraLoaderModelOnly` after the base model, and set `strength_model=1.0`. Follow the original model's **4 steps, CFG 1.0, and empty negative prompt** settings; change the 12-step setting in the full example workflows when using this LoRA. See [VIGGLE-LORA-NOTICE.txt](VIGGLE-LORA-NOTICE.txt) for conversion and verification details. This converted release has not undergone a separate end-to-end image-generation test.

## Sources and license

The system prompt templates come from the [official Qwen Image 2.1 repository](https://github.com/QwenLM/Qwen-Image-2.1/tree/main/prompt_rewrite/prompts). See [NOTICE.md](NOTICE.md) for provenance and modifications. The relevant templates, models, and optional LoRA are subject to the [Qwen Research License Agreement](LICENSE-QWEN-RESEARCH): non-commercial research or evaluation only; commercial use requires separate permission from the rights holder.

## T8star

[Bilibili](https://space.bilibili.com/385085361) · [YouTube](https://www.youtube.com/@T8star-Aix/) · [API](https://api.seedance.nz/sign-up?aff=5f4w) · [Free gallery](https://www.openzhenzhen.com) · [Online AI apps](https://www.runninghub.ai/zh-cn/user-center/1907375370302308353/userPost?inviteCode=rh-v1121) · [ComfyUI package](https://pan.quark.cn/s/264edb7e36bd) · [Hugging Face](https://huggingface.co/t8star)
