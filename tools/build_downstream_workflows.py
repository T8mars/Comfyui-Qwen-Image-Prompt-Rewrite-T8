"""Build complete, drag-and-drop Qwen Image 2.1 workflows for the local ComfyUI."""

from copy import deepcopy
import json
import os
from pathlib import Path
import uuid


ROOT = Path(__file__).resolve().parents[1]
configured_comfy = os.environ.get("QWEN_PE_COMFY_DIR")
if not configured_comfy:
    raise SystemExit("Set QWEN_PE_COMFY_DIR to your ComfyUI directory before regenerating workflows")
COMFY = Path(configured_comfy).expanduser().resolve()
if not (COMFY / "main.py").is_file():
    raise SystemExit(f"QWEN_PE_COMFY_DIR is not a ComfyUI checkout: {COMFY}")
PE_TEMPLATE = ROOT / "workflows/Qwen-PE-2.1-Text-to-Image-Ready.json"
QWEN_TEMPLATE = Path(os.environ.get(
    "QWEN_PE_QWEN_TEMPLATE",
    str(COMFY.parent / "python/Lib/site-packages/comfyui_workflow_templates_json/"
        "templates/image_qwen_image_2_1_t2i.json"),
))
DEST = ROOT / "workflows"
DEST.mkdir(exist_ok=True)

pe_base = next(node for node in json.loads(PE_TEMPLATE.read_text(encoding="utf-8"))["nodes"]
               if node["type"] == "QwenPERewriteT8")
official = json.loads(QWEN_TEMPLATE.read_text(encoding="utf-8"))
qwen_base = {node["type"]: node for node in official["definitions"]["subgraphs"][0]["nodes"]}


def socket(name, dtype, link=None):
    return {"name": name, "type": dtype, "link": link}


def output(name, dtype):
    return {"name": name, "type": dtype, "links": None}


def build(name, *, edit=False, transparent=False, heretic=False, chinese=False, multi10=False):
    nodes = {}
    links = []

    def add(node_id, kind, x, y):
        node = deepcopy(qwen_base[kind])
        node["id"] = node_id
        node["pos"] = [x, y]
        node["order"] = len(nodes)
        for item in node.get("inputs", []):
            item["link"] = None
        for item in node.get("outputs", []):
            item["links"] = None
        nodes[node_id] = node
        return node

    def connect(source_id, output_name, target_id, input_name):
        source, target = nodes[source_id], nodes[target_id]
        out_index = next(i for i, item in enumerate(source["outputs"]) if item["name"] == output_name)
        in_index = next(i for i, item in enumerate(target["inputs"]) if item["name"] == input_name)
        link_id = len(links) + 1
        dtype = source["outputs"][out_index]["type"]
        source["outputs"][out_index]["links"] = (source["outputs"][out_index]["links"] or []) + [link_id]
        target["inputs"][in_index]["link"] = link_id
        links.append([link_id, source_id, out_index, target_id, in_index, dtype])

    pe = deepcopy(pe_base)
    pe["id"] = 1
    pe["pos"] = [350, 220]
    pe["size"] = [480, 590]
    pe["order"] = 0
    if len(pe["widgets_values"]) == 8:
        pe["widgets_values"][2:2] = ["auto", "auto", False]
    if len(pe["widgets_values"]) != 11:
        raise ValueError("PE workflow template has an unexpected widget layout")
    pe["widgets_values"][3:5] = ["中文" if chinese else "English", transparent]
    instruction = ("Place the red apple from <image1> at the center of the orange scene in <image2>. "
                   "Keep the apple red and recognizable, retain the orange setting, and follow "
                   "<image2>'s landscape canvas."
                   if edit else
                   "A single red apple on a dark wooden table, softly lit from the side, realistic photograph.")
    if transparent:
        instruction = "A single golden butterfly, entire body and wings visible, isolated transparent background."
    if chinese:
        instruction = "一只完整的金色蝴蝶，翅膀有细致纹理，主体居中，背景透明。"
    if multi10:
        instruction = ("Create a wide abstract collage with ten equal vertical color panels. "
                       "From left to right, use the dominant color of <image1>, <image2>, "
                       "<image3>, <image4>, <image5>, <image6>, <image7>, <image8>, "
                       "<image9>, and <image10>, in exactly that order. "
                       "Keep the final tenth panel yellow.")
    pe["widgets_values"][0] = instruction
    pe["widgets_values"][1] = "auto"
    pe["widgets_values"][2] = "21:9" if multi10 else "auto" if edit else "4:5" if chinese else "1:1"
    if heretic:
        pe["widgets_values"][5] = "pe_t2i_heretic-Q4_K_M.gguf"
    pe["widgets_values"][10] = "fixed"
    pe["widgets_values_named"].update(user_prompt=instruction, task="auto",
                                       aspect_ratio=pe["widgets_values"][2],
                                       output_language="中文" if chinese else "English",
                                       transparent_rgba=transparent,
                                       t2i_model=pe["widgets_values"][5],
                                       control_after_generate="fixed")
    for item in pe["inputs"]:
        item["link"] = None
    for item in pe["outputs"]:
        item["links"] = None
    nodes[1] = pe

    canvas = {
        "id": 2, "type": "QwenPECanvasT8", "pos": [900, 550], "size": [320, 150],
        "flags": {}, "order": 1, "mode": 0,
        "inputs": [socket("pe_result", "PE_RESULT"), socket("resolution", "INT"),
                   socket("follow_input_size", "BOOLEAN")],
        "outputs": [output("width", "INT"), output("height", "INT"),
                    output("latent", "LATENT"), output("ratio_source", "STRING")],
        "properties": {"Node name for S&R": "QwenPECanvasT8"},
        "widgets_values": [512, True],
        "widgets_values_named": {"resolution": 512, "follow_input_size": True},
    }
    nodes[2] = canvas
    add(3, "UNETLoader", 400, 950)["widgets_values"] = ["qwen_image_2.1_int8_convrot.safetensors", "default"]
    add(4, "CLIPLoader", 700, 950)["widgets_values"] = ["qwen3vl_8b_int8_convrot.safetensors", "qwen_image", "default"]
    add(5, "VAELoader", 1000, 950)["widgets_values"] = ["qwen_image_2.1_vae_bf16.safetensors"]
    encoder = add(6, "TextEncodeQwenImage21", 900, 180)
    encoder["widgets_values"] = ["", "", 128 if multi10 else 512]
    encoder["size"] = [410, 320]
    sampler = add(7, "KSampler", 1450, 340)
    sampler["widgets_values"] = [42, "fixed", 12, 1, "euler", "simple", 1]
    sampler["size"] = [340, 280]
    add(8, "VAEDecode", 1850, 370)
    save = {
        "id": 9, "type": "SaveImage", "pos": [2150, 330], "size": [340, 320],
        "flags": {}, "order": 8, "mode": 0,
        "inputs": [socket("images", "IMAGE"), socket("filename_prefix", "STRING")],
        "outputs": [output("IMAGE", "IMAGE")],
        "properties": {"Node name for S&R": "SaveImage"},
        "widgets_values": [name],
    }
    nodes[9] = save
    preview = {
        "id": 10, "type": "PreviewAny", "title": "Prompt rewrite diagnostics",
        "pos": [900, 20], "size": [420, 110], "flags": {}, "order": 9, "mode": 0,
        "inputs": [socket("source", "*")], "outputs": [output("STRING", "STRING")],
        "properties": {"Node name for S&R": "PreviewAny"}, "widgets_values": [],
    }
    nodes[10] = preview
    prompt_preview = deepcopy(preview)
    prompt_preview.update(id=13, title="Final rewritten prompt", pos=[900, -180], order=10)
    nodes[13] = prompt_preview
    ratio_preview = deepcopy(preview)
    ratio_preview.update(id=14, title="Canvas ratio source", pos=[900, 760], order=11)
    nodes[14] = ratio_preview

    connect(1, "rewritten_prompt", 6, "prompt")
    connect(1, "pe_result", 2, "pe_result")
    connect(1, "diagnostics", 10, "source")
    connect(1, "rewritten_prompt", 13, "source")
    connect(2, "ratio_source", 14, "source")
    connect(3, "MODEL", 7, "model")
    connect(4, "CLIP", 6, "clip")
    connect(5, "VAE", 8, "vae")
    connect(6, "positive", 7, "positive")
    connect(6, "negative", 7, "negative")
    connect(2, "latent", 7, "latent_image")
    connect(7, "LATENT", 8, "samples")
    connect(8, "IMAGE", 9, "images")

    if edit or multi10:
        connect(5, "VAE", 6, "vae")
        colors = [0xCF3030, 0x20834E, 0x265CB9, 0xE79719, 0x8A3B9B,
                  0x149EAA, 0xC45B80, 0x735C36, 0x4E5689, 0xE6C827]
        for index in (range(1, 11) if multi10 else (1, 2)):
            node_id = 21 + index if multi10 else 10 + index
            if multi10:
                image_node = {
                    "id": node_id, "type": "EmptyImage", "pos": [0, 100 + 145 * index],
                    "size": [280, 120], "flags": {}, "order": len(nodes), "mode": 0,
                    "inputs": [], "outputs": [output("IMAGE", "IMAGE")],
                    "properties": {"Node name for S&R": "EmptyImage"},
                    "widgets_values": [128, 128, 1, colors[index - 1]],
                    "widgets_values_named": {"width": 128, "height": 128,
                                             "batch_size": 1, "color": colors[index - 1]},
                }
            else:
                filename = f"qwen_pe_reference_{index}.png"
                source = ROOT / "workflows" / "fixtures" / filename
                destination = COMFY / "input" / filename
                if not destination.exists():
                    destination.write_bytes(source.read_bytes())
                image_node = {
                    "id": node_id, "type": "LoadImage", "pos": [0, 50 + 250 * index],
                    "size": [290, 300], "flags": {}, "order": len(nodes), "mode": 0,
                    "inputs": [], "outputs": [output("IMAGE", "IMAGE"), output("MASK", "MASK")],
                    "properties": {"Node name for S&R": "LoadImage"},
                    "widgets_values": [filename, "image"],
                    "widgets_values_named": {"image": filename, "upload": "image"},
                }
            nodes[node_id] = image_node
            input_name = f"image_{index}"
            if not any(item["name"] == f"images.{input_name}" for item in encoder["inputs"]):
                encoder["inputs"].insert(index, socket(f"images.{input_name}", "IMAGE"))
            connect(node_id, "IMAGE", 1, input_name)
            connect(node_id, "IMAGE", 6, f"images.{input_name}")

    # Autogrow sockets change indices when new reference images are inserted.
    for link in links:
        target = nodes[link[3]]
        link[4] = next(i for i, item in enumerate(target["inputs"]) if item["link"] == link[0])

    graph = {
        "id": str(uuid.uuid5(uuid.NAMESPACE_URL, name)), "revision": 0,
        "last_node_id": max(nodes), "last_link_id": len(links),
        "nodes": list(nodes.values()), "links": links, "groups": [], "config": {},
        "extra": {}, "version": 0.4,
    }
    path = DEST / f"{name}.json"
    path.write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = COMFY / "user/default/workflows" / path.name
    saved.parent.mkdir(parents=True, exist_ok=True)
    saved.write_bytes(path.read_bytes())
    print(path)


build("Qwen-PE-2.1-Full-T2I")
build("Qwen-PE-2.1-Full-T2I-Heretic", heretic=True)
build("Qwen-PE-2.1-Full-Edit-2-Images", edit=True)
build("Qwen-PE-2.1-Full-Edit-10-Images", multi10=True)
build("Qwen-PE-2.1-Full-Transparent-T2I", transparent=True)
build("Qwen-PE-2.1-Full-Transparent-Chinese", transparent=True, chinese=True)
