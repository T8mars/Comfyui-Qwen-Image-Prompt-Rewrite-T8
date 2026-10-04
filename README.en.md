# Qwen Image 2.1 Prompt Rewrite T8

**Language / 语言: [简体中文](README.md) · English**

A local prompt rewriting node for ComfyUI. Give it text alone, or text plus 1–10 reference images. It selects the corresponding Qwen Image 2.1 PE GGUF model and returns a prompt for text-to-image generation or image editing. It supports Chinese and English output, aspect ratios, transparent-background instructions, and model unloading.

> **What the models do:** The GGUF files in this project **rewrite prompts**. To generate images, your ComfyUI workflow also needs a Qwen Image 2.1 diffusion model, text encoder, and VAE. Those are separate downloads.

> **Before the first run, install two separate components:** at least one PE GGUF model set and the llama.cpp `llama-server.exe` runtime. ComfyUI Manager / Registry installs the node code **without the GGUF files or EXE**. For a missing EXE, follow the [Windows instructions below](#install-llama-server-windows).

[GGUF model mirror](https://huggingface.co/t8star/qwen-image-2.1-comfy) · [ComfyUI Registry page](https://registry.comfy.org/zh/publishers/t8star/nodes/qwen-image-prompt-rewrite-t8) · [Example workflows](workflows/) · [Report an issue](https://github.com/T8mars/Comfyui-Qwen-Image-Prompt-Rewrite-T8/issues/new)

## Quick setup

1. Place the repository under `ComfyUI/custom_nodes/Comfyui-Qwen-Image-Prompt-Rewrite-T8/`, then restart ComfyUI:

   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/T8mars/Comfyui-Qwen-Image-Prompt-Rewrite-T8.git
   cd Comfyui-Qwen-Image-Prompt-Rewrite-T8
   ```

2. From the node directory, use the **same Python environment as ComfyUI** to download four Q4_K_M main models and two I2I vision components:

   ```bash
   python tools/download_models.py
   ```

   The downloader verifies file sizes and SHA256 hashes and resumes interrupted downloads. You can also get the six files from the [base model mirror](https://huggingface.co/t8star/qwen-image-2.1-comfy) and the [Heretic I2I source repository](https://huggingface.co/pottokao/Qwen-Image-2.1-PE-I2I-Heretic-GGUF), then place them, with their original names, in `models/llm/qwenimage-pe/`:

   | Purpose | File |
   | --- | --- |
   | Text-to-image | `Qwen-Image-2.1-PE-T2I.Q4_K_M.gguf` |
   | Image editing | `Qwen-Image-2.1-PE-I2I.Q4_K_M.gguf` |
   | Vision component for editing | `Qwen-Image-2.1-PE-I2I.mmproj-bf16.gguf` |
   | Optional Heretic text-to-image model | `pe_t2i_heretic-Q4_K_M.gguf` |
   | Optional Heretic image-editing model | `pe_i2i_heretic-Q4_K_M.gguf` |
   | Vision component for Heretic editing | `pe_i2i_heretic.mmproj-bf16.gguf` |

   The source repository does not provide a Q4 vision component, so this project uses the BF16 file. You may also use `ComfyUI/models/llm/qwenimage-pe/` or set `QWEN_PE_MODEL_DIR` to a custom model directory. Model weights are not bundled with the GitHub source or Registry package.

3. Install `llama-server.exe` using the next section. ComfyUI's Python environment must contain `torch`, `numpy`, and `Pillow`.

After restarting ComfyUI, find the nodes under **Qwen Image 2.1 / Prompt Rewrite**. Registry availability may change; use the repository installation steps above if the node is not offered in Manager.

## Install llama-server (Windows)

This node launches the official llama.cpp **`llama-server.exe`** locally to rewrite prompts. It is separate from the GGUF models. The `llama-cpp-python` wheel does **not** supply this executable, so `pip install llama-cpp-python` does not fix a missing-EXE error.

**Windows x64 with NVIDIA:** download **both** prebuilt CUDA 12.4 ZIPs from the [official llama.cpp b11068 release](https://github.com/ggml-org/llama.cpp/releases/tag/b11068):

1. [llama-b11068-bin-win-cuda-12.4-x64.zip — main program, including `llama-server.exe`](https://github.com/ggml-org/llama.cpp/releases/download/b11068/llama-b11068-bin-win-cuda-12.4-x64.zip)
2. [cudart-llama-bin-win-cuda-12.4-x64.zip — matching CUDA DLLs](https://github.com/ggml-org/llama.cpp/releases/download/b11068/cudart-llama-bin-win-cuda-12.4-x64.zip)

Find the **actual node directory** containing `pe_runtime.py` and `tools/` (Manager may choose a different folder name). Create `runtime/llama-b11068/` inside it and extract the **contents of both ZIPs into that same directory**. The result should contain:

```text
<node directory>/runtime/llama-b11068/llama-server.exe
<node directory>/runtime/llama-b11068/cudart64_12.dll
<node directory>/runtime/llama-b11068/cublas64_12.dll
```

Do not leave the executable inside a ZIP, put it in the GGUF model directory, or create an extra nested folder. Alternatively, open PowerShell in the node directory and download/extract the same official packages with:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\download_runtime.ps1
```

Check the installation from that directory with `Test-Path .\runtime\llama-b11068\llama-server.exe`; it should return `True`. Restart ComfyUI. If you keep the runtime elsewhere, set `QWEN_PE_LLAMA_SERVER` to the **full path** of its executable before restarting. On Linux/macOS, select the appropriate binary from the [official release](https://github.com/ggml-org/llama.cpp/releases/tag/b11068) and set this environment variable.

## Using the nodes

| Node | Purpose |
| --- | --- |
| **PE Rewrite T8** | Rewrite a prompt and return text, a structured `PE_RESULT`, and diagnostics JSON. |
| **PE Canvas T8** | Turn the model's aspect ratio or a reference image's dimensions into width, height, and an empty Qwen Image 2.1 latent. |
| **PE Unload T8** | Explicitly stop a model retained with `keep_loaded`. |
| **PE Local Models T8** | List discovered main GGUF models and vision components. |

Enter your instruction in `user_prompt` and choose the following options in **PE Rewrite T8**:

- `task`: `auto` selects text-to-image or editing from the images **actually present**. `t2i` accepts text only; `edit` requires at least one image.
- `t2i_model` / `edit_model`: Separate model choices for the two tasks. Select `pe_i2i_heretic-Q4_K_M.gguf` in `edit_model` to use Heretic editing; `vision_model=Auto` selects its matching vision component.
- `aspect_ratio`: `auto`, `1:1`, `1:2`, `2:3`, `3:4`, `4:5`, `16:9`, `9:16`, `21:9`, `9:21`, `5:4`, `4:3`, or `2:1`. An explicit choice overrides the model's suggestion and feeds PE Canvas.
- `output_language`: `auto`, `中文`, or `English`. With an explicit language, the node checks the rewrite and, if needed, translates it with the local model. The language choice applies to **descriptive prose**; quoted text requested by the user and visible English titles, signs, and labels in reference images remain verbatim in quotation marks.
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
| `llama-server.exe not found` | This means the separate runtime is missing, not the model. Download both official ZIPs and extract them under the node's `runtime/llama-b11068/` as shown in the [Windows instructions](#install-llama-server-windows). |
| Editing reports a missing mmproj | Put the matching I2I main model and vision file together, or choose the matching `vision_model`. |
| `model response reached the generation or context limit` | The node retries once without thinking. If it still fails, inspect the token counts in the error, use fewer images, or shorten the instruction. Multi-image jobs use more VRAM. |
| `model failed format validation` | Both attempts failed a JSON, ratio, language, or image-reference check. Read the error details. After a successful run, `diagnostics` can confirm the task and input-port mapping. |
| `llama-server did not become healthy` although the log shows it listening | Update to version 1.0.11 or later; local health checks and inference bypass `HTTP_PROXY`/`HTTPS_PROXY`. On older versions, `NO_PROXY=127.0.0.1,localhost` is a temporary workaround. |
| Chinese output reports `Chinese descriptive prose contains English words` | Update to version 1.0.11 or later; translation fallback now retries with feedback about the remaining English words. If it still fails, inspect the reported words and `diagnostics`. |
| Chinese output fails language validation on an English title or sign in an input image | Update to version 1.0.13 or later; visible image text can remain verbatim while the surrounding description stays Chinese. |
| `TextEncodeQwenImage21` is missing | Update to a ComfyUI release supporting Qwen Image 2.1. Prompt-only workflows do not need this node. |
| The output PNG is still opaque | Check the downstream image model, VAE, and saving workflow. This node only writes the transparency prompt. |

The local `llama-server` listens on `127.0.0.1` only, and the node's requests bypass environment proxies. A visual copy of each input is scaled to at most about one million pixels and a 4096-pixel longest side; PE Canvas retains the original dimensions. The response is checked for JSON structure, image references, language, and mode constraints, with one retry on validation failure; translation fallback is checked separately and can retry twice more.

## Optional: Viggle Turbo 4-step LoRA

[Download the ComfyUI-converted LoRA](https://huggingface.co/t8star/Qwen-Image-2.1-viggle-turbo-4step-r64-comfy/resolve/main/Qwen-Image-2.1-viggle-turbo-4step-r64-comfyui-T8.safetensors?download=true). This LoRA modifies the **image-generation diffusion model** and is installed separately from the PE GGUF files. This repository only converted the key names of [Viggle's original LoRA](https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo); it did not retrain the weights. Put the file in `ComfyUI/models/loras/`, add `LoraLoaderModelOnly` after the base model, and set `strength_model=1.0`. Follow the original model's **4 steps, CFG 1.0, and empty negative prompt** settings; change the 12-step setting in the full example workflows when using this LoRA. See the [Hugging Face model card](https://huggingface.co/t8star/Qwen-Image-2.1-viggle-turbo-4step-r64-comfy) for conversion and verification details. This converted model has not undergone a separate end-to-end image-generation test.

## Sources and license

The system prompt templates come from the [official Qwen Image 2.1 repository](https://github.com/QwenLM/Qwen-Image-2.1/tree/main/prompt_rewrite/prompts). See [NOTICE.md](NOTICE.md) for provenance and modifications. The relevant templates, models, and optional LoRA are subject to the [Qwen Research License Agreement](LICENSE-QWEN-RESEARCH): non-commercial research or evaluation only; commercial use requires separate permission from the rights holder.

## T8star

[Bilibili](https://space.bilibili.com/385085361) · [YouTube](https://www.youtube.com/@T8star-Aix/) · [API](https://api.seedance.nz/sign-up?aff=5f4w) · [Free gallery](https://www.openzhenzhen.com) · [Online AI apps](https://www.runninghub.ai/zh-cn/user-center/1907375370302308353/userPost?inviteCode=rh-v1121) · [ComfyUI package](https://pan.quark.cn/s/264edb7e36bd) · [Hugging Face](https://huggingface.co/t8star)
