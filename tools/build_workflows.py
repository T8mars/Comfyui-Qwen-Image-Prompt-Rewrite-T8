"""Build drag-and-drop ComfyUI workflows from a node saved by the actual UI."""

from copy import deepcopy
import json
import os
from pathlib import Path
import uuid

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
configured_comfy = os.environ.get("QWEN_PE_COMFY_DIR")
if not configured_comfy:
    raise SystemExit("Set QWEN_PE_COMFY_DIR to your ComfyUI directory before regenerating workflows")
COMFY = Path(configured_comfy).expanduser().resolve()
if not (COMFY / "main.py").is_file():
    raise SystemExit(f"QWEN_PE_COMFY_DIR is not a ComfyUI checkout: {COMFY}")
SAVED = COMFY / "user/default/workflows/Qwen-PE-T2I-UI-Test.json"
DEST = ROOT / "workflows"
DEST.mkdir(exist_ok=True)
FIXTURES = DEST / "fixtures"
FIXTURES.mkdir(exist_ok=True)


def make_fixture(name, color, label):
    image = Image.new("RGB", (512, 384), color)
    draw = ImageDraw.Draw(image)
    draw.rectangle((48, 88, 460, 320), outline="white", width=8)
    draw.text((80, 135), label, fill="white", stroke_fill="black", stroke_width=2)
    image.save(FIXTURES / name)
    image.save(COMFY / "input" / name)


make_fixture("qwen_pe_reference_1.png", "#235f8a", "BLUE SOURCE 1")
make_fixture("qwen_pe_reference_2.png", "#9d502b", "ORANGE SCENE 2")

template = json.loads(SAVED.read_text(encoding="utf-8"))
base_node = template["nodes"][0]


def socket(name, dtype, link=None, optional=False):
    item = {"name": name, "type": dtype, "link": link}
    if optional:
        item["shape"] = 7
    return item


def output(name, dtype, links=None):
    return {"name": name, "type": dtype, "links": links}


def workflow(edit=False, demo=False, unload=False, options=False, chinese_options=False,
             chinese_transparent=False, edit_english_transparent=False):
    name = ("Qwen-PE-2.1-Edit-English-Transparent-Demo" if edit_english_transparent else
            "Qwen-PE-2.1-Chinese-Transparent-Demo" if chinese_transparent else
            "Qwen-PE-2.1-Chinese-Output-Demo" if chinese_options else
            "Qwen-PE-2.1-Options-Demo" if options else
            "Qwen-PE-2.1-Keep-Then-Unload-Demo" if unload else
            "Qwen-PE-2.1-Edit-2-Images-Demo" if demo else
            "Qwen-PE-2.1-Edit-2-Images-Ready" if edit else
            "Qwen-PE-2.1-Text-to-Image-Ready")
    main = deepcopy(base_node)
    main["pos"] = [380, 220]
    main["size"] = [440, 520]
    main["order"] = 2 if edit else 0
    main["widgets_values"][2:2] = ["auto", "auto", False]
    instruction = ("生成一只完整的蝴蝶，清楚呈现身体、头部和左右对称的两对翅膀；"
                   "把<image1>的蓝色与<image2>的橙色用于翅膀，主体抠图，背景完全透明。"
                   if edit_english_transparent else
                   "一只金色蝴蝶，翅膀有细致花纹，主体居中，背景完全透明。"
                   if chinese_transparent else
                   "A watercolor painting of a small mountain cottage beside a lake."
                   if chinese_options else
                   "一只金色蝴蝶，翅膀有细致花纹，居中展示。" if options else
                   ("将<image1>的蓝色放在画面左侧，将<image2>的橙色放在右侧，形成简洁的双色构图。"
                    if demo else "将<image1>的主体放进<image2>的场景，保留主体身份和场景透视。")
                   if edit else "一只穿蓝色雨衣的柯基坐在雨中的红色长椅上。")
    main["widgets_values"][0] = instruction
    main["widgets_values"][1] = "auto"
    main["widgets_values"][10] = "fixed"
    if unload:
        main["widgets_values"][8] = "keep_loaded"
    if options:
        main["widgets_values"][2:5] = ["21:9", "English", True]
    if chinese_options:
        main["widgets_values"][2:5] = ["4:3", "中文", False]
    if chinese_transparent:
        main["widgets_values"][2:5] = ["4:5", "中文", True]
    if edit_english_transparent:
        main["widgets_values"][2:5] = ["3:4", "English", True]
    main["widgets_values_named"].update(user_prompt=instruction, task="auto",
                                         aspect_ratio="auto", output_language="auto",
                                         transparent_rgba=False, control_after_generate="fixed")
    if unload:
        main["widgets_values_named"]["model_lifetime"] = "keep_loaded"
    if options:
        main["widgets_values_named"].update(aspect_ratio="21:9", output_language="English",
                                             transparent_rgba=True)
    if chinese_options:
        main["widgets_values_named"].update(aspect_ratio="4:3", output_language="中文",
                                             transparent_rgba=False)
    if chinese_transparent:
        main["widgets_values_named"].update(aspect_ratio="4:5", output_language="中文",
                                             transparent_rgba=True)
    if edit_english_transparent:
        main["widgets_values_named"].update(aspect_ratio="3:4", output_language="English",
                                             transparent_rgba=True)
    main["outputs"][0]["links"] = [3]
    main["outputs"][1]["links"] = [4]
    main["outputs"][2]["links"] = [5]
    nodes = [main]
    links = []

    if edit:
        for i, color in ((1, 0x235F8A), (2, 0x9D502B)):
            if demo:
                image_node = {
                    "id": i + 1, "type": "EmptyImage", "pos": [0, 120 + 210 * (i - 1)],
                    "size": [300, 190], "flags": {}, "order": i - 1, "mode": 0,
                    "inputs": [], "outputs": [output("IMAGE", "IMAGE", [i])],
                    "properties": {"Node name for S&R": "EmptyImage"},
                    "widgets_values": [512, 384, 1, color],
                    "widgets_values_named": {"width": 512, "height": 384,
                                             "batch_size": 1, "color": color},
                }
            else:
                image_node = {
                    "id": i + 1, "type": "LoadImage", "pos": [0, 120 + 210 * (i - 1)],
                    "size": [300, 190], "flags": {}, "order": i - 1, "mode": 0,
                    "inputs": [],
                    "outputs": [output("IMAGE", "IMAGE", [i]), output("MASK", "MASK")],
                    "properties": {"Node name for S&R": "LoadImage"},
                    "widgets_values": [f"qwen_pe_reference_{i}.png", "image"],
                    "widgets_values_named": {"image": f"qwen_pe_reference_{i}.png", "upload": "image"},
                }
            nodes.append(image_node)
            main["inputs"][i - 1]["link"] = i
            links.append([i, i + 1, 0, 1, i - 1, "IMAGE"])

    canvas = {
        "id": 4, "type": "QwenPECanvasT8", "pos": [900, 405], "size": [310, 130],
        "flags": {}, "order": 3 if edit else 1, "mode": 0,
        "inputs": [socket("pe_result", "PE_RESULT", 4), socket("resolution", "INT"),
                   socket("follow_input_size", "BOOLEAN")],
        "outputs": [output("width", "INT"), output("height", "INT"),
                    output("latent", "LATENT"), output("ratio_source", "STRING", [6])],
        "properties": {"Node name for S&R": "QwenPECanvasT8"},
        "widgets_values": [1024, True],
        "widgets_values_named": {"resolution": 1024, "follow_input_size": True},
    }
    nodes.append(canvas)
    links.append([4, 1, 1, 4, 0, "PE_RESULT"])

    for node_id, ypos, link_id, origin_id, origin_slot, label in (
        (5, 180, 3, 1, 0, "Rewritten prompt"),
        (6, 445, 5, 1, 2, "Model and run diagnostics"),
        (7, 650, 6, 4, 3, "Canvas source"),
    ):
        preview = {
            "id": node_id, "type": "PreviewAny", "pos": [1330, ypos], "size": [400, 160],
            "flags": {}, "order": node_id, "mode": 0,
            "inputs": [socket("source", "*", link_id)],
            "outputs": [output("STRING", "STRING")],
            "properties": {"Node name for S&R": "PreviewAny"},
            "widgets_values": [], "title": label,
        }
        nodes.append(preview)
        links.append([link_id, origin_id, origin_slot, node_id, 0, "STRING"])

    if unload:
        main["outputs"][1]["links"].append(7)
        nodes.append({
            "id": 8, "type": "QwenPEUnloadT8", "pos": [900, 630], "size": [300, 80],
            "flags": {}, "order": 8, "mode": 0,
            "inputs": [socket("pe_result", "PE_RESULT", 7)],
            "outputs": [output("pe_result", "PE_RESULT")],
            "properties": {"Node name for S&R": "QwenPEUnloadT8"},
            "widgets_values": [],
        })
        links.append([7, 1, 1, 8, 0, "PE_RESULT"])

    graph = {
        "id": str(uuid.uuid5(uuid.NAMESPACE_URL, name)), "revision": 0,
        "last_node_id": 8 if unload else 7,
        "last_link_id": 7 if unload else 6,
        "nodes": nodes, "links": links, "groups": [], "config": {}, "extra": {}, "version": 0.4,
    }
    path = DEST / (name + ".json")
    path.write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
    (COMFY / "user/default/workflows" / path.name).write_bytes(path.read_bytes())
    print(path)


workflow(False)
workflow(True)
workflow(True, demo=True)
workflow(unload=True)
workflow(options=True)
workflow(chinese_options=True)
workflow(chinese_transparent=True)
workflow(True, demo=True, edit_english_transparent=True)


def ten_image_workflow():
    name = "Qwen-PE-2.1-Edit-10-Images-Demo"
    main = deepcopy(base_node)
    main["widgets_values"][2:2] = ["auto", "auto", False]
    instruction = "使用这十张参考图制作宽幅抽象色块拼贴；按图序使用各图主色，最后一个色块保持第十张图的黄色。"
    main["widgets_values"][0] = instruction
    main["widgets_values"][10] = "fixed"
    main["widgets_values_named"].update(user_prompt=instruction, task="auto", aspect_ratio="auto",
                                         output_language="auto", transparent_rgba=False,
                                         control_after_generate="fixed")
    main["pos"] = [420, 300]
    main["size"] = [450, 550]
    main["order"] = 10
    main["outputs"][0]["links"] = [12]
    main["outputs"][1]["links"] = [11]
    main["outputs"][2]["links"] = [13]
    nodes = [main]
    links = []
    colors = [0xCF3030, 0x20834E, 0x265CB9, 0xE79719, 0x8A3B9B,
              0x149EAA, 0xC45B80, 0x735C36, 0x4E5689, 0xE6C827]
    for index, color in enumerate(colors):
        node_id = index + 2
        link_id = index + 1
        nodes.append({
            "id": node_id, "type": "EmptyImage", "pos": [-20, 40 + index * 155],
            "size": [290, 120], "flags": {}, "order": index, "mode": 0,
            "inputs": [], "outputs": [output("IMAGE", "IMAGE", [link_id])],
            "properties": {"Node name for S&R": "EmptyImage"},
            "widgets_values": [384, 384, 1, color],
            "widgets_values_named": {"width": 384, "height": 384, "batch_size": 1,
                                     "color": color},
        })
        main["inputs"][index]["link"] = link_id
        links.append([link_id, node_id, 0, 1, index, "IMAGE"])
    canvas = {
        "id": 12, "type": "QwenPECanvasT8", "pos": [980, 420], "size": [300, 130],
        "flags": {}, "order": 11, "mode": 0,
        "inputs": [socket("pe_result", "PE_RESULT", 11), socket("resolution", "INT"),
                   socket("follow_input_size", "BOOLEAN")],
        "outputs": [output("width", "INT"), output("height", "INT"),
                    output("latent", "LATENT"), output("ratio_source", "STRING", [14])],
        "properties": {"Node name for S&R": "QwenPECanvasT8"},
        "widgets_values": [1024, True],
        "widgets_values_named": {"resolution": 1024, "follow_input_size": True},
    }
    nodes.append(canvas)
    links.append([11, 1, 1, 12, 0, "PE_RESULT"])
    for node_id, link_id, origin_id, origin_slot, label in (
        (13, 12, 1, 0, "Rewritten prompt"),
        (14, 13, 1, 2, "Diagnostics"),
        (15, 14, 12, 3, "Canvas source"),
    ):
        nodes.append({
            "id": node_id, "type": "PreviewAny", "pos": [1400, 130 + (node_id - 13) * 235],
            "size": [400, 170], "flags": {}, "order": node_id, "mode": 0,
            "inputs": [socket("source", "*", link_id)],
            "outputs": [output("STRING", "STRING")],
            "properties": {"Node name for S&R": "PreviewAny"},
            "widgets_values": [], "title": label,
        })
        links.append([link_id, origin_id, origin_slot, node_id, 0, "STRING"])
    graph = {"id": str(uuid.uuid5(uuid.NAMESPACE_URL, name)), "revision": 0,
             "last_node_id": 15, "last_link_id": 14, "nodes": nodes,
             "links": links, "groups": [], "config": {}, "extra": {}, "version": 0.4}
    path = DEST / (name + ".json")
    path.write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
    (COMFY / "user/default/workflows" / path.name).write_bytes(path.read_bytes())
    print(path)


ten_image_workflow()


def local_models_workflow():
    name = "Qwen-PE-2.1-Local-Models-Demo"
    nodes = [
        {"id": 1, "type": "QwenPEModelListT8", "pos": [220, 220], "size": [290, 100],
         "flags": {}, "order": 0, "mode": 0,
         "inputs": [socket("refresh", "BOOLEAN")],
         "outputs": [output("local_models", "STRING", [1])],
         "properties": {"Node name for S&R": "QwenPEModelListT8"},
         "widgets_values": [False], "widgets_values_named": {"refresh": False}},
        {"id": 2, "type": "PreviewAny", "pos": [650, 200], "size": [480, 250],
         "flags": {}, "order": 1, "mode": 0,
         "inputs": [socket("source", "*", 1)],
         "outputs": [output("STRING", "STRING")],
         "properties": {"Node name for S&R": "PreviewAny"},
         "widgets_values": [], "title": "Local GGUF inventory"},
    ]
    graph = {"id": str(uuid.uuid5(uuid.NAMESPACE_URL, name)), "revision": 0,
             "last_node_id": 2, "last_link_id": 1, "nodes": nodes,
             "links": [[1, 1, 0, 2, 0, "STRING"]],
             "groups": [], "config": {}, "extra": {}, "version": 0.4}
    path = DEST / (name + ".json")
    path.write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
    (COMFY / "user/default/workflows" / path.name).write_bytes(path.read_bytes())
    print(path)


local_models_workflow()
