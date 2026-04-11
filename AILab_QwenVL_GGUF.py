# ComfyUI-QwenVL (GGUF)
# GGUF nodes powered by llama.cpp for Qwen-VL models, including Qwen3-VL and Qwen2.5-VL.
# Provides vision-capable GGUF inference and prompt execution.
#
# Models are loaded via llama-cpp-python and configured through gguf_models.json.
# This integration script follows GPL-3.0 License.
# When using or modifying this code, please respect both the original model licenses
# and this integration's license terms.
#
# Source: https://github.com/1038lab/ComfyUI-QwenVL

import base64
import gc
import io
import inspect
import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

import folder_paths
from AILab_OutputCleaner import OutputCleanConfig, clean_model_output

NODE_DIR = Path(__file__).parent
CONFIG_PATH = NODE_DIR / "hf_models.json"
SYSTEM_PROMPTS_PATH = NODE_DIR / "AILab_System_Prompts.json"
GGUF_CONFIG_PATH = NODE_DIR / "gguf_models.json"


def _load_prompt_config():
    preset_prompts = ["🖼️ Detailed Description"]
    system_prompts: dict[str, str] = {}

    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh) or {}
        preset_prompts = data.get("_preset_prompts") or preset_prompts
        system_prompts = data.get("_system_prompts") or system_prompts
    except Exception as exc:
        print(f"[QwenVL] Config load failed: {exc}")

    try:
        with open(SYSTEM_PROMPTS_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh) or {}
        qwenvl_prompts = data.get("qwenvl") or {}
        preset_override = data.get("_preset_prompts") or []
        if isinstance(qwenvl_prompts, dict) and qwenvl_prompts:
            system_prompts = qwenvl_prompts
        if isinstance(preset_override, list) and preset_override:
            preset_prompts = preset_override
    except FileNotFoundError:
        pass
    except Exception as exc:
        print(f"[QwenVL] System prompts load failed: {exc}")

    return preset_prompts, system_prompts


PRESET_PROMPTS, SYSTEM_PROMPTS = _load_prompt_config()


@dataclass(frozen=True)
class GGUFVLResolved:
    display_name: str
    repo_id: str | None
    alt_repo_ids: list[str]
    author: str | None
    repo_dirname: str
    model_filename: str
    mmproj_filename: str | None
    context_length: int
    image_max_tokens: int
    image_min_tokens: int
    n_batch: int
    gpu_layers: int
    top_k: int
    pool_size: int


def _resolve_base_dir(base_dir_value: str) -> Path:
    base_dir = Path(base_dir_value)
    if base_dir.is_absolute():
        return base_dir
    # Check extra_model_paths.yaml via folder_paths
    folder_key = base_dir.parts[0] if base_dir.parts else base_dir_value
    sub_path = Path(*base_dir.parts[1:]) if len(base_dir.parts) > 1 else Path()
    if folder_key in folder_paths.folder_names_and_paths:
        paths = folder_paths.get_folder_paths(folder_key)
        if paths:
            return Path(paths[0]) / sub_path
    return Path(folder_paths.models_dir) / base_dir


def _safe_dirname(value: str) -> str:
    value = (value or "").strip()
    if not value:
        return "unknown"
    return "".join(ch for ch in value if ch.isalnum() or ch in "._- ").strip() or "unknown"


def _model_name_to_filename_candidates(model_name: str) -> set[str]:
    raw = (model_name or "").strip()
    if not raw:
        return set()
    candidates = {raw, f"{raw}.gguf"}
    if " / " in raw:
        tail = raw.split(" / ", 1)[1].strip()
        candidates.update({tail, f"{tail}.gguf"})
    if "/" in raw:
        tail = raw.rsplit("/", 1)[-1].strip()
        candidates.update({tail, f"{tail}.gguf"})
    return candidates


def _load_gguf_vl_catalog():
    if not GGUF_CONFIG_PATH.exists():
        return {"base_dir": "LLM", "models": {}}
    try:
        with open(GGUF_CONFIG_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh) or {}
    except Exception as exc:
        print(f"[QwenVL] gguf_models.json load failed: {exc}")
        return {"base_dir": "LLM", "models": {}}

    base_dir = data.get("base_dir") or "LLM"

    flattened: dict[str, dict] = {}

    repos = data.get("qwenVL_model") or data.get("vl_repos") or data.get("repos") or {}
    seen_display_names: set[str] = set()
    for repo_key, repo in repos.items():
        if not isinstance(repo, dict):
            continue
        author = repo.get("author") or repo.get("publisher")
        repo_name = repo.get("repo_name") or repo.get("repo_name_override") or repo_key
        repo_id = repo.get("repo_id") or (f"{author}/{repo_name}" if author and repo_name else None)
        alt_repo_ids = repo.get("alt_repo_ids") or []

        defaults = repo.get("defaults") or {}
        mmproj_file = repo.get("mmproj_file")
        model_files = repo.get("model_files") or []

        for model_file in model_files:
            display = Path(model_file).name
            if display in seen_display_names:
                display = f"{display} ({repo_key})"
            seen_display_names.add(display)
            flattened[display] = {
                **defaults,
                "author": author,
                "repo_dirname": repo_name,
                "repo_id": repo_id,
                "alt_repo_ids": alt_repo_ids,
                "filename": model_file,
                "mmproj_filename": mmproj_file,
            }

    legacy_models = data.get("models") or {}
    for name, entry in legacy_models.items():
        if isinstance(entry, dict):
            flattened[name] = entry

    return {"base_dir": base_dir, "models": flattened}


GGUF_VL_CATALOG = _load_gguf_vl_catalog()

LOCAL_PREFIX = "[local] "


def _is_gemma_model_name(name: str) -> bool:
    """Detect Gemma models by filename substring (covers catalog keys and [local] paths)."""
    return "gemma" in (name or "").lower()


def _scan_local_gguf_files() -> dict[str, Path]:
    """Scan base_dir for .gguf files not already in the catalog."""
    base_dir = _resolve_base_dir(GGUF_VL_CATALOG.get("base_dir") or "LLM")
    if not base_dir.is_dir():
        return {}
    catalog_filenames: set[str] = set()
    for entry in (GGUF_VL_CATALOG.get("models") or {}).values():
        fn = (entry or {}).get("filename")
        if fn:
            catalog_filenames.add(Path(fn).name)
    found: dict[str, Path] = {}
    for p in base_dir.rglob("*.gguf"):
        if not p.is_file() or "mmproj" in p.name.lower():
            continue
        if p.name in catalog_filenames:
            continue
        try:
            rel = p.relative_to(base_dir)
        except ValueError:
            rel = Path(p.name)
        found[f"{LOCAL_PREFIX}{rel}"] = p
    return found


LOCAL_GGUF_FILES: dict[str, Path] = _scan_local_gguf_files()


def _filter_kwargs_for_callable(fn, kwargs: dict) -> dict:
    try:
        sig = inspect.signature(fn)
    except Exception:
        return dict(kwargs)

    params = list(sig.parameters.values())
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params):
        return dict(kwargs)

    allowed: set[str] = set()
    for p in params:
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
            allowed.add(p.name)
    return {k: v for k, v in kwargs.items() if k in allowed}


import math


def _estimate_image_tokens(width: int, height: int) -> int:
    """Estimate Qwen2VL image tokens: ceil(H/28) * ceil(W/28)."""
    return math.ceil(height / 28) * math.ceil(width / 28)


def _resize_image_to_token_budget(pil_img: Image.Image, max_tokens: int) -> Image.Image:
    """Shrink image so its estimated token count fits within *max_tokens*."""
    w, h = pil_img.size
    cur_tokens = _estimate_image_tokens(w, h)
    if cur_tokens <= max_tokens:
        return pil_img
    scale = math.sqrt(max_tokens / cur_tokens)
    new_w = max(int(w * scale) // 28 * 28, 28)
    new_h = max(int(h * scale) // 28 * 28, 28)
    print(f"[QwenVL] Auto-resizing image from {w}x{h} ({cur_tokens} tokens) "
          f"to {new_w}x{new_h} ({_estimate_image_tokens(new_w, new_h)} tokens) to fit ctx budget")
    return pil_img.resize((new_w, new_h), Image.LANCZOS)


def _tensor_to_pil(tensor) -> Image.Image | None:
    """Convert a ComfyUI IMAGE tensor to a PIL Image."""
    if tensor is None:
        return None
    if tensor.ndim == 4:
        tensor = tensor[0]
    array = (tensor * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
    return Image.fromarray(array, mode="RGB")


def _pil_to_base64_png(pil_img: Image.Image) -> str:
    """Encode a PIL Image as base64 PNG string."""
    buf = io.BytesIO()
    try:
        pil_img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    finally:
        buf.close()


def _tensor_to_base64_png(tensor) -> str | None:
    pil_img = _tensor_to_pil(tensor)
    if pil_img is None:
        return None
    return _pil_to_base64_png(pil_img)


def _sample_video_frames(video, frame_count: int):
    if video is None:
        return []
    if video.ndim != 4:
        return [video]
    total = int(video.shape[0])
    frame_count = max(int(frame_count), 1)
    if total <= frame_count:
        return [video[i] for i in range(total)]
    idx = np.linspace(0, total - 1, frame_count, dtype=int)
    return [video[i] for i in idx]


def _pick_device(device_choice: str) -> str:
    if device_choice == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if device_choice.startswith("cuda") and torch.cuda.is_available():
        return "cuda"
    if device_choice == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _download_single_file(repo_ids: list[str], filename: str, target_path: Path):
    if target_path.exists():
        print(f"[QwenVL] Using cached file: {target_path}")
        return

    target_path.parent.mkdir(parents=True, exist_ok=True)

    last_exc: Exception | None = None
    for repo_id in repo_ids:
        print(f"[QwenVL] Downloading {filename} from {repo_id} -> {target_path}")
        try:
            downloaded = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="model",
                local_dir=str(target_path.parent),
                local_dir_use_symlinks=False,
            )
            downloaded_path = Path(downloaded)
            if downloaded_path.exists() and downloaded_path.resolve() != target_path.resolve():
                downloaded_path.replace(target_path)
            if target_path.exists():
                print(f"[QwenVL] Download complete: {target_path}")
            break
        except Exception as exc:
            last_exc = exc
            print(f"[QwenVL] hf_hub_download failed from {repo_id}: {exc}")
    else:
        raise FileNotFoundError(f"[QwenVL] Download failed for {filename}: {last_exc}")

    if not target_path.exists():
        raise FileNotFoundError(f"[QwenVL] File not found after download: {target_path}")


_downloading_files: set[str] = set()
_download_lock = threading.Lock()


def _background_download(repo_ids: list[str], filename: str, target_path: Path):
    try:
        _download_single_file(repo_ids, filename, target_path)
    except Exception as exc:
        print(f"[QwenVL] Background download failed: {exc}")
        print("[QwenVL] Download flag has been auto-cleared. Re-run to retry. / ダウンロードフラグは自動解除されました。再実行で再試行します。")
        print("[QwenVL] Please check your network connection and available storage. / ネットワーク接続やストレージ空き容量を確認してください。")
    finally:
        with _download_lock:
            _downloading_files.discard(str(target_path))


def _resolve_model_entry(model_name: str) -> GGUFVLResolved:
    # --- Local file ---
    if model_name.startswith(LOCAL_PREFIX):
        local_path = LOCAL_GGUF_FILES.get(model_name)
        if local_path is None or not local_path.is_file():
            raise FileNotFoundError(f"[QwenVL] Local GGUF not found: {model_name}")
        # Auto-detect mmproj in the same directory (matches both "mmproj-*.gguf" and "*.mmproj-*.gguf")
        mmproj = None
        for candidate in sorted(local_path.parent.glob("*mmproj*.gguf")):
            if candidate.is_file():
                mmproj = candidate.name
                break
        return GGUFVLResolved(
            display_name=model_name,
            repo_id=None,
            alt_repo_ids=[],
            author=None,
            repo_dirname=local_path.parent.name,
            model_filename=local_path.name,
            mmproj_filename=mmproj,
            context_length=8192,
            image_max_tokens=8192,
            image_min_tokens=1024,
            n_batch=8192,
            gpu_layers=-1,
            top_k=0,
            pool_size=4194304,
        )

    # --- Catalog lookup ---
    all_models = GGUF_VL_CATALOG.get("models") or {}
    entry = all_models.get(model_name) or {}
    if not entry:
        wanted = _model_name_to_filename_candidates(model_name)
        for candidate in all_models.values():
            filename = candidate.get("filename")
            if filename and Path(filename).name in wanted:
                entry = candidate
                break

    # --- Fallback: try as local file if not found in catalog ---
    if not entry or not entry.get("filename"):
        local_key = f"{LOCAL_PREFIX}{model_name}"
        if local_key in LOCAL_GGUF_FILES:
            return _resolve_model_entry(local_key)

    repo_id = entry.get("repo_id")
    alt_repo_ids = entry.get("alt_repo_ids") or []

    author = entry.get("author") or entry.get("publisher")
    repo_dirname = entry.get("repo_dirname") or (repo_id.split("/")[-1] if isinstance(repo_id, str) and "/" in repo_id else model_name)

    model_filename = entry.get("filename")
    mmproj_filename = entry.get("mmproj_filename")

    if not model_filename:
        raise ValueError(f"[QwenVL] gguf_vl_models.json entry missing 'filename' for: {model_name}")

    def _int(name: str, default: int) -> int:
        value = entry.get(name, default)
        try:
            return int(value)
        except Exception:
            return default

    return GGUFVLResolved(
        display_name=model_name,
        repo_id=repo_id,
        alt_repo_ids=[str(x) for x in alt_repo_ids if x],
        author=str(author) if author else None,
        repo_dirname=_safe_dirname(str(repo_dirname)),
        model_filename=str(model_filename),
        mmproj_filename=str(mmproj_filename) if mmproj_filename else None,
        context_length=_int("context_length", 8192),
        image_max_tokens=_int("image_max_tokens", 8192),
        image_min_tokens=_int("image_min_tokens", 1024),
        n_batch=_int("n_batch", 8192),
        gpu_layers=_int("gpu_layers", -1),
        top_k=_int("top_k", 0),
        pool_size=_int("pool_size", 4194304),
    )


class QwenVLGGUFBase:
    def __init__(self):
        self.llm = None
        self.chat_handler = None
        self.current_signature = None
        self._is_gemma = False

    def clear(self):
        self.llm = None
        self.chat_handler = None
        self.current_signature = None
        self._is_gemma = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load_backend(self):
        try:
            from llama_cpp import Llama  # noqa: F401
        except Exception as exc:
            raise RuntimeError(
                "[QwenVL] llama_cpp is not available. Install the GGUF vision dependency first. See docs/GGUF_MANUAL_INSTALL.md"
            ) from exc

    def _load_model(
        self,
        model_name: str,
        device: str,
        ctx: int | None,
        n_batch: int | None,
        gpu_layers: int | None,
        image_max_tokens: int | None,
        image_min_tokens: int | None,
        top_k: int | None,
        pool_size: int | None,
        enable_thinking: bool = False,
    ):
        self._load_backend()

        resolved = _resolve_model_entry(model_name)

        if resolved.repo_id is None:
            # Local file: path comes directly from scan results
            local_key = model_name if model_name.startswith(LOCAL_PREFIX) else f"{LOCAL_PREFIX}{model_name}"
            local_path = LOCAL_GGUF_FILES.get(local_key)
            if local_path is None or not local_path.is_file():
                raise FileNotFoundError(f"[QwenVL] Local GGUF not found: {model_name}")
            model_path = local_path
            mmproj_path = local_path.parent / resolved.mmproj_filename if resolved.mmproj_filename else None
            if mmproj_path is not None and not mmproj_path.is_file():
                print(f"[QwenVL] Warning: auto-detected mmproj not found: {mmproj_path}")
                mmproj_path = None
        else:
            # Catalog model: resolve paths and download if needed
            base_dir = _resolve_base_dir(GGUF_VL_CATALOG.get("base_dir") or "LLM")
            author_dir = _safe_dirname(resolved.author or "")
            repo_dir = _safe_dirname(resolved.repo_dirname)
            target_dir = base_dir / author_dir / repo_dir

            model_path = target_dir / Path(resolved.model_filename).name
            mmproj_path = target_dir / Path(resolved.mmproj_filename).name if resolved.mmproj_filename else None

            repo_ids: list[str] = []
            if resolved.repo_id:
                repo_ids.append(resolved.repo_id)
            repo_ids.extend(resolved.alt_repo_ids)

            needs_download = []
            if not model_path.exists():
                if not repo_ids:
                    raise FileNotFoundError(f"[QwenVL] GGUF model not found locally and no repo_id provided: {model_path}")
                needs_download.append((repo_ids, resolved.model_filename, model_path))
            if mmproj_path is not None and not mmproj_path.exists():
                if not repo_ids:
                    raise FileNotFoundError(f"[QwenVL] mmproj not found locally and no repo_id provided: {mmproj_path}")
                needs_download.append((repo_ids, resolved.mmproj_filename, mmproj_path))

            if needs_download:
                started = []
                with _download_lock:
                    for repos, filename, path in needs_download:
                        key = str(path)
                        if key not in _downloading_files:
                            _downloading_files.add(key)
                            started.append((repos, filename, path))

                for repos, filename, path in started:
                    t = threading.Thread(target=_background_download, args=(repos, filename, path), daemon=True)
                    t.start()

                all_files = ", ".join(f for _, f, _ in needs_download)
                if started:
                    raise RuntimeError(
                        f"[QwenVL] Model download started / モデルのダウンロードを開始しました: {all_files}\n"
                        "Please re-run after download completes. Check console for progress.\n"
                        "ダウンロード完了後に再実行してください。進捗はコンソールで確認できます。"
                    )
                else:
                    raise RuntimeError(
                        f"[QwenVL] Model is downloading / モデルをダウンロード中です: {all_files}\n"
                        "Please re-run after download completes. Check console for progress.\n"
                        "ダウンロード完了後に再実行してください。進捗はコンソールで確認できます。"
                    )

        device_kind = _pick_device(device)

        n_ctx = int(ctx) if ctx is not None else resolved.context_length
        n_batch_val = int(n_batch) if n_batch is not None else resolved.n_batch
        top_k_val = int(top_k) if top_k is not None else resolved.top_k
        pool_size_val = int(pool_size) if pool_size is not None else resolved.pool_size

        if device_kind == "cuda":
            n_gpu_layers = int(gpu_layers) if gpu_layers is not None else resolved.gpu_layers
        else:
            n_gpu_layers = 0

        img_max = int(image_max_tokens) if image_max_tokens is not None else resolved.image_max_tokens
        img_min = int(image_min_tokens) if image_min_tokens is not None else resolved.image_min_tokens

        has_mmproj = mmproj_path is not None and mmproj_path.exists()
        is_gemma = _is_gemma_model_name(model_path.name) or _is_gemma_model_name(model_name)

        signature = (
            str(model_path),
            str(mmproj_path) if has_mmproj else "",
            n_ctx,
            n_batch_val,
            n_gpu_layers,
            img_max,
            img_min,
            top_k_val,
            pool_size_val,
        )
        if self.llm is not None and self.current_signature == signature:
            return

        self.clear()

        from llama_cpp import Llama

        self.chat_handler = None
        if has_mmproj:
            handler_cls = None
            if is_gemma:
                try:
                    from llama_cpp.llama_chat_format import Gemma4ChatHandler

                    handler_cls = Gemma4ChatHandler
                except ImportError:
                    raise RuntimeError(
                        "[QwenVL] Gemma 4 requires llama-cpp-python v0.3.35+ with Gemma4ChatHandler "
                        "(JamePeng fork). Update your llama_cpp install. See docs/GGUF_MANUAL_INSTALL.md"
                    )
            else:
                try:
                    from llama_cpp.llama_chat_format import Qwen3VLChatHandler

                    handler_cls = Qwen3VLChatHandler
                except ImportError:
                    try:
                        from llama_cpp.llama_chat_format import Qwen25VLChatHandler

                        handler_cls = Qwen25VLChatHandler
                    except ImportError:
                        raise RuntimeError(
                            "[QwenVL] Missing Qwen VL chat handler in llama_cpp. Install the correct fork/wheel. See docs/GGUF_MANUAL_INSTALL.md"
                        )

            # Build handler kwargs per family. Gemma4ChatHandler validates kwargs in
            # its parent __init__ at runtime (not via signature), so _filter_kwargs_for_callable
            # cannot protect us — pass only keys we know each handler accepts.
            mmproj_kwargs = {
                "clip_model_path": str(mmproj_path),
                "image_max_tokens": img_max,
                "verbose": False,
            }
            if not is_gemma:
                mmproj_kwargs["force_reasoning"] = False
            mmproj_kwargs = _filter_kwargs_for_callable(getattr(handler_cls, "__init__", handler_cls), mmproj_kwargs)
            if "image_max_tokens" not in mmproj_kwargs:
                print(
                    "[QwenVL] Warning: installed llama_cpp chat handler does not support image_max_tokens; "
                    "image token budget will be controlled by ctx only."
                )
            self.chat_handler = handler_cls(**mmproj_kwargs)

        llm_kwargs = {
            "model_path": str(model_path),
            "n_ctx": n_ctx,
            "n_gpu_layers": n_gpu_layers,
            "n_batch": n_batch_val,
            "swa_full": True,
            "verbose": False,
            "pool_size": pool_size_val,
            "top_k": top_k_val,
        }
        if has_mmproj and self.chat_handler is not None:
            llm_kwargs["chat_handler"] = self.chat_handler
            llm_kwargs["image_min_tokens"] = img_min
            llm_kwargs["image_max_tokens"] = img_max

        print(f"[QwenVL] Loading GGUF: {model_path.name} (device={device_kind}, gpu_layers={n_gpu_layers}, ctx={n_ctx})")
        llm_kwargs_filtered = _filter_kwargs_for_callable(getattr(Llama, "__init__", Llama), llm_kwargs)
        if has_mmproj and self.chat_handler is not None and "chat_handler" not in llm_kwargs_filtered:
            print(
                "[QwenVL] Warning: installed llama_cpp Llama() does not accept chat_handler; images will be ignored. "
                "Update llama-cpp-python to a multimodal-capable build."
            )
        if device_kind == "cuda" and n_gpu_layers == 0:
            print("[QwenVL] Warning: device=cuda selected but n_gpu_layers=0; model will run on CPU.")
        try:
            self.llm = Llama(**llm_kwargs_filtered)
        except Exception:
            self.chat_handler = None
            raise
        self._is_gemma = is_gemma
        self.current_signature = signature

    def _invoke(
        self,
        system_prompt: str,
        user_prompt: str,
        images_b64: list[str],
        max_tokens: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
        seed: int,
        stop_words: list[str] | None = None,
    ) -> str:
        if images_b64:
            content = [{"type": "text", "text": user_prompt}]
            for img in images_b64:
                if not img:
                    continue
                content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}})
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ]
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

        if self._is_gemma:
            default_stop = ["<|turn>", "<|channel>", "<end_of_turn>", "<start_of_turn>"]
        else:
            default_stop = ["<|im_end|>", "<|im_start|>"]

        start = time.perf_counter()
        result = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=int(max_tokens),
            temperature=float(temperature),
            top_p=float(top_p),
            repeat_penalty=float(repetition_penalty),
            seed=int(seed),
            stop=default_stop + (stop_words or []),
        )
        elapsed = max(time.perf_counter() - start, 1e-6)

        usage = result.get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        if isinstance(completion_tokens, int) and completion_tokens > 0:
            tok_s = completion_tokens / elapsed
            if isinstance(prompt_tokens, int) and prompt_tokens >= 0:
                print(
                    f"[QwenVL] Tokens: prompt={prompt_tokens}, completion={completion_tokens}, "
                    f"time={elapsed:.2f}s, speed={tok_s:.2f} tok/s"
                )
            else:
                print(f"[QwenVL] Tokens: completion={completion_tokens}, time={elapsed:.2f}s, speed={tok_s:.2f} tok/s")

        content = (result.get("choices") or [{}])[0].get("message", {}).get("content", "")
        cleaned = clean_model_output(str(content or ""), OutputCleanConfig(mode="text"))
        return cleaned.strip()

    def run(
        self,
        model_name: str,
        preset_prompt: str,
        custom_prompt: str,
        image,
        video,
        frame_count: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
        seed: int,
        keep_model_loaded: bool,
        device: str,
        ctx: int | None,
        n_batch: int | None,
        gpu_layers: int | None,
        image_max_tokens: int | None,
        image_min_tokens: int | None,
        top_k: int | None,
        pool_size: int | None,
        enable_thinking: bool = False,
        stop_words: list[str] | None = None,
        image2=None,
        image3=None,
    ):
        torch.manual_seed(int(seed))

        prompt = SYSTEM_PROMPTS.get(preset_prompt, preset_prompt)
        if custom_prompt and custom_prompt.strip():
            prompt = custom_prompt.strip()

        is_gemma = _is_gemma_model_name(model_name)
        if not is_gemma:
            # Qwen uses inline /think /no_think tokens; Gemma 4 uses the handler's enable_thinking flag.
            think_prefix = "/think" if enable_thinking else "/no_think"
            prompt = f"{think_prefix}\n{prompt}"

        # Collect all PIL images first (static images + video frames)
        pil_images: list[Image.Image] = []
        for img_tensor in (image, image2, image3):
            pil = _tensor_to_pil(img_tensor)
            if pil is not None:
                pil_images.append(pil)
        if video is not None:
            for frame in _sample_video_frames(video, int(frame_count)):
                pil = _tensor_to_pil(frame)
                if pil is not None:
                    pil_images.append(pil)

        # Auto-resize images to fit within ctx budget (prevent MROPE seq_add crash on Qwen2VL).
        # Gemma 4 does not use MROPE, so we skip this Qwen-specific guard.
        if pil_images and ctx is not None and not is_gemma:
            text_token_overhead = 256  # system prompt + user prompt + formatting
            token_budget_for_images = max(ctx - max_tokens - text_token_overhead, 0)
            if token_budget_for_images == 0:
                print(f"[QwenVL] Warning: ctx={ctx} is too small for max_tokens={max_tokens}; images may cause a crash")
            else:
                per_image_budget = token_budget_for_images // len(pil_images)
                effective_cap = min(per_image_budget, image_max_tokens or per_image_budget)
                pil_images = [_resize_image_to_token_budget(img, effective_cap) for img in pil_images]

        images_b64: list[str] = [_pil_to_base64_png(img) for img in pil_images]
        del pil_images

        try:
            self._load_model(
                model_name=model_name,
                device=device,
                ctx=ctx,
                n_batch=n_batch,
                gpu_layers=gpu_layers,
                image_max_tokens=image_max_tokens,
                image_min_tokens=image_min_tokens,
                top_k=top_k,
                pool_size=pool_size,
                enable_thinking=enable_thinking,
            )
            if images_b64 and self.chat_handler is None:
                print("[QwenVL] Warning: images provided but this model entry has no mmproj_file; images will be ignored")
            text = self._invoke(
                system_prompt="You are a helpful vision-language assistant.",
                user_prompt=prompt,
                images_b64=images_b64 if self.chat_handler is not None else [],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                seed=seed,
                stop_words=stop_words,
            )
            return (text,)
        finally:
            if not keep_model_loaded:
                self.clear()


class AILab_QwenVL_GGUF(QwenVLGGUFBase):
    @classmethod
    def INPUT_TYPES(cls):
        all_models = GGUF_VL_CATALOG.get("models") or {}
        model_keys = sorted([key for key, entry in all_models.items() if (entry or {}).get("mmproj_filename")])
        local_keys = sorted(LOCAL_GGUF_FILES.keys())
        model_keys = (model_keys + local_keys) or ["(edit gguf_models.json)"]
        default_model = model_keys[0]

        prompts = PRESET_PROMPTS or ["🖼️ Detailed Description"]
        preferred_prompt = "🖼️ Detailed Description"
        default_prompt = preferred_prompt if preferred_prompt in prompts else prompts[0]

        return {
            "required": {
                "model_name": (model_keys, {"default": default_model}),
                "preset_prompt": (prompts, {"default": default_prompt}),
                "custom_prompt": ("STRING", {"default": "", "multiline": True}),
                "max_tokens": ("INT", {"default": 512, "min": 64, "max": 32768}),
                "enable_thinking": ("BOOLEAN", {"default": False}),
                "keep_model_loaded": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 1, "min": 1, "max": 2**32 - 1}),
            },
            "optional": {
                "image": ("IMAGE",),
                "video": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("RESPONSE",)
    FUNCTION = "process"
    CATEGORY = "QwenVL-F"

    def process(
        self,
        model_name,
        preset_prompt,
        custom_prompt,
        max_tokens,
        enable_thinking,
        keep_model_loaded,
        seed,
        image=None,
        video=None,
    ):
        return self.run(
            model_name=model_name,
            preset_prompt=preset_prompt,
            custom_prompt=custom_prompt,
            image=image,
            video=video,
            frame_count=16,
            max_tokens=max_tokens,
            temperature=0.6,
            top_p=0.9,
            repetition_penalty=1.2,
            seed=seed,
            keep_model_loaded=keep_model_loaded,
            device="auto",
            ctx=None,
            n_batch=None,
            gpu_layers=None,
            image_max_tokens=None,
            image_min_tokens=None,
            top_k=None,
            pool_size=None,
            enable_thinking=enable_thinking,
        )


class AILab_QwenVL_GGUF_Advanced(QwenVLGGUFBase):
    @classmethod
    def INPUT_TYPES(cls):
        all_models = GGUF_VL_CATALOG.get("models") or {}
        model_keys = sorted([key for key, entry in all_models.items() if (entry or {}).get("mmproj_filename")])
        local_keys = sorted(LOCAL_GGUF_FILES.keys())
        model_keys = (model_keys + local_keys) or ["(edit gguf_models.json)"]
        default_model = model_keys[0]

        prompts = PRESET_PROMPTS or ["🖼️ Detailed Description"]
        preferred_prompt = "🖼️ Detailed Description"
        default_prompt = preferred_prompt if preferred_prompt in prompts else prompts[0]

        num_gpus = torch.cuda.device_count()
        gpu_list = [f"cuda:{i}" for i in range(num_gpus)]
        device_options = ["auto", "cpu", "mps"] + gpu_list

        return {
            "required": {
                "model_name": (model_keys, {"default": default_model}),
                "device": (device_options, {"default": "auto"}),
                "preset_prompt": (prompts, {"default": default_prompt}),
                "custom_prompt": ("STRING", {"default": "", "multiline": True}),
                "max_tokens": ("INT", {"default": 512, "min": 64, "max": 32768}),
                "temperature": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 2.0}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0}),
                "repetition_penalty": ("FLOAT", {"default": 1.2, "min": 0.5, "max": 2.0}),
                "frame_count": ("INT", {"default": 16, "min": 1, "max": 64}),
                "ctx": ("INT", {"default": 32768, "min": 1024, "max": 262144, "step": 512}),
                "n_batch": ("INT", {"default": 8192, "min": 64, "max": 32768, "step": 64}),
                "gpu_layers": ("INT", {"default": -1, "min": -1, "max": 200}),
                "image_max_tokens": ("INT", {"default": 8192, "min": 256, "max": 1024000, "step": 256}),
                "image_min_tokens": ("INT", {"default": 1024, "min": 64, "max": 1024000, "step": 64}),
                "top_k": ("INT", {"default": 0, "min": 0, "max": 32768}),
                "pool_size": ("INT", {"default": 4194304, "min": 1048576, "max": 10485760, "step": 524288}),
                "enable_thinking": ("BOOLEAN", {"default": False}),
                "stop_words": ("STRING", {"default": ""}),
                "keep_model_loaded": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 1, "min": 1, "max": 2**32 - 1}),
            },
            "optional": {
                "image": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "video": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("RESPONSE",)
    FUNCTION = "process"
    CATEGORY = "QwenVL-F"

    def process(
        self,
        model_name,
        device,
        preset_prompt,
        custom_prompt,
        max_tokens,
        temperature,
        top_p,
        repetition_penalty,
        frame_count,
        ctx,
        n_batch,
        gpu_layers,
        image_max_tokens,
        image_min_tokens,
        top_k,
        pool_size,
        enable_thinking,
        stop_words,
        keep_model_loaded,
        seed,
        image=None,
        image2=None,
        image3=None,
        video=None,
    ):
        parsed = [w.strip() for w in stop_words.split(",") if w.strip()] if stop_words else None
        return self.run(
            model_name=model_name,
            preset_prompt=preset_prompt,
            custom_prompt=custom_prompt,
            image=image,
            video=video,
            frame_count=frame_count,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            seed=seed,
            keep_model_loaded=keep_model_loaded,
            device=device,
            ctx=ctx,
            n_batch=n_batch,
            gpu_layers=gpu_layers,
            image_max_tokens=image_max_tokens,
            image_min_tokens=image_min_tokens,
            top_k=top_k,
            pool_size=pool_size,
            enable_thinking=enable_thinking,
            stop_words=parsed,
            image2=image2,
            image3=image3,
        )


NODE_CLASS_MAPPINGS = {
    "QwenVL-F_GGUF": AILab_QwenVL_GGUF,
    "QwenVL-F_GGUF_Advanced": AILab_QwenVL_GGUF_Advanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenVL-F_GGUF": "QwenVL-F (GGUF)",
    "QwenVL-F_GGUF_Advanced": "QwenVL-F Advanced (GGUF)",
}
