"""Download the approved PE model artifacts and verify their SHA256."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import os
from pathlib import Path
import time
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1] / "models" / "llm" / "qwenimage-pe"
FILES = (
    (
        "prithivMLmods/Qwen-Image-2.1-PE-T2I-GGUF",
        "e18d4a3e0830ab157770738b16830e6fcf5f57d4",
        "Qwen-Image-2.1-PE-T2I.Q4_K_M.gguf",
        5629108864,
        "340feb42c784e35b704a0f0d8d1c679a3fe60437bd9ea44a7a6fd2d730470506",
    ),
    (
        "prithivMLmods/Qwen-Image-2.1-PE-I2I-GGUF",
        "55b9c1a326599e142d59bcad8715d5601ccf8daa",
        "Qwen-Image-2.1-PE-I2I.Q4_K_M.gguf",
        5629108864,
        "a5c6cb28cbaf838834d1335c8392619af5809d9fe95d0ebd321e15751866dfe9",
    ),
    (
        "prithivMLmods/Qwen-Image-2.1-PE-I2I-GGUF",
        "55b9c1a326599e142d59bcad8715d5601ccf8daa",
        "Qwen-Image-2.1-PE-I2I.mmproj-bf16.gguf",
        921704608,
        "8dedb71dbc3092dc47de9108ad373d68a12854e2527d59bd9399601738f3bce1",
    ),
    (
        "pottokao/Qwen-Image-2.1-PE-T2I-Heretic-GGUF",
        "e24aa32689780a6d34bdc30d401cb68771e41c02",
        "pe_t2i_heretic-Q4_K_M.gguf",
        5891335008,
        "fe176ded062942ac8858290a33c9303a6e9c09c31405020a22ad0d47fa7b9b69",
    ),
    (
        "pottokao/Qwen-Image-2.1-PE-I2I-Heretic-GGUF",
        "564f783ebea6c244c7dae8460ece65eb174479a6",
        "pe_i2i_heretic-Q4_K_M.gguf",
        5891334720,
        "4fa7d846acc6b6366f2b959293fb073f304d18c2d0a73d7a193cb09cc87bf074",
    ),
    (
        "pottokao/Qwen-Image-2.1-PE-I2I-Heretic-GGUF",
        "564f783ebea6c244c7dae8460ece65eb174479a6",
        "pe_i2i_heretic.mmproj-bf16.gguf",
        921704608,
        "171a59a76e498252fb0ee5c145633c6f3a2eaaad03a171e3d5828de230369807",
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(spec: tuple[str, str, str, int, str]) -> str:
    repo, revision, name, size, expected_hash = spec
    ROOT.mkdir(parents=True, exist_ok=True)
    target = ROOT / name
    partial = ROOT / (name + ".part")
    if target.is_file() and target.stat().st_size == size and sha256(target) == expected_hash:
        return f"verified existing: {name}"
    if target.exists():
        raise RuntimeError(f"existing file failed verification: {target}")
    url = f"https://huggingface.co/{repo}/resolve/{revision}/{name}"
    for attempt in range(1, 7):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > size:
            raise RuntimeError(f"partial file longer than expected: {partial}")
        if offset == size:
            break
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            with urlopen(Request(url, headers=headers), timeout=120) as response:
                if offset and response.status != 206:
                    raise RuntimeError(f"server did not honor resume for {name}")
                with partial.open("ab" if offset else "wb") as output:
                    copied = offset
                    for block in iter(lambda: response.read(4 * 1024 * 1024), b""):
                        output.write(block)
                        copied += len(block)
                    print(f"transfer {name}: {copied}/{size}", flush=True)
            if partial.stat().st_size == size:
                break
        except Exception as exc:
            print(f"retry {name} attempt {attempt}: {exc}", flush=True)
            if attempt == 6:
                raise
            time.sleep(min(2 ** attempt, 20))
    actual_size = partial.stat().st_size
    if actual_size != size:
        raise RuntimeError(f"size mismatch for {name}: {actual_size} != {size}")
    actual_hash = sha256(partial)
    if actual_hash != expected_hash:
        raise RuntimeError(f"SHA256 mismatch for {name}: {actual_hash}")
    os.replace(partial, target)
    return f"verified downloaded: {name} ({size} bytes)"


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(download, spec) for spec in FILES]
        for future in as_completed(futures):
            print(future.result(), flush=True)
