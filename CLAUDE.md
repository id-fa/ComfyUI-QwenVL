# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ComfyUI-QwenVL is a ComfyUI custom node plugin providing multimodal AI capabilities via Alibaba's Qwen-VL vision-language models. It supports two backends: HuggingFace Transformers and GGUF (llama-cpp-python), with 6 ComfyUI nodes total.

## Setup & Installation

```bash
pip install -r requirements.txt
# GGUF backend (optional): pip install -r gguf_requirements.txt
# SageAttention (optional): pip install sageattention
```

No test suite or linter is configured. Publishing to the ComfyUI registry is handled automatically via `.github/workflows/publish.yml` when `pyproject.toml` changes on main.

## Architecture

### Node Registration

`__init__.py` dynamically scans all `.py` files, imports them, and collects `NODE_CLASS_MAPPINGS` / `NODE_DISPLAY_NAME_MAPPINGS` for ComfyUI. Each node module exports these two dicts.

### Two Backend Hierarchies

**Transformers backend** (`AILab_QwenVL.py`):
- `QwenVLBase` — base class handling model download (`ensure_model` via `snapshot_download`), loading (`load_model`), device/VRAM management, quantization (4-bit/8-bit/FP16 via BitsAndBytes), and attention backend selection.
- `AILab_QwenVL` (simple) and `AILab_QwenVL_Advanced` (full control) inherit from it.
- Advancedノードは `image`, `image2`, `image3` の3つのoptional画像入力を持ち、複数画像の同時参照が可能。Simpleノードは `image` のみ。

**GGUF backend** (`AILab_QwenVL_GGUF.py`):
- `QwenVLGGUFBase` — base class for llama-cpp-python models with output cleaning.
- `AILab_QwenVL_GGUF` and `AILab_QwenVL_GGUF_Advanced` inherit from it.
- Advancedノードは `image`, `image2`, `image3` の3つのoptional画像入力を持ち、複数画像の同時参照が可能。Simpleノードは `image` のみ。
- **Chat handler自動選択**: モデル名（ファイル名または `[local] ...` キー）に `gemma` が含まれる場合は `Gemma4ChatHandler` を使用（JamePeng fork v0.3.35+ 必須）。それ以外は `Qwen3VLChatHandler` → `Qwen25VLChatHandler` の順でフォールバック。判定は `_is_gemma_model_name()` によるファイル名部分一致。

**Prompt enhancers** (text-only, no vision):
- `AILab_QwenVL_PromptEnhancer.py` — Transformers-based。`keep_model_loaded` スイッチあり（HF text model用の `_invoke_text` パスと、VLモデル流用の `_invoke_qwen` パスの両方でアンロード対応）。
- `AILab_QwenVL_GGUF_PromptEnhancer.py` — GGUF-based。`keep_model_loaded` スイッチあり（`process()` 完了後に `self.clear()` でアンロード）。モデル名に `gemma` を含む場合は `chat_format="qwen"` を渡さず、GGUFに埋め込まれた chat template を llama_cpp に使わせる（それ以外は従来通り `chat_format="qwen"` を強制）。

**Output cleaning** (`AILab_OutputCleaner.py`):
- `OutputCleanConfig` dataclass and utilities to strip thinking tags and leaked tokens from model output.
- HF側 (`generate()`) でも GGUF側 (`_invoke()`) でも `clean_model_output()` を通して出力をクリーニングする。

### Key Subsystems

- **Attention resolution**: `resolve_attention_mode()` auto-selects SageAttention → Flash-Attn → SDPA with GPU architecture-aware kernel selection (`set_sage_attention`, `get_sage_attention_config`).
- **Memory management**: `enforce_memory()` auto-downgrades quantization if VRAM is insufficient; `clear()` releases models and clears CUDA cache。各バックエンドでメモリリーク対策を実施済み：
  - **中間テンソル解放**: HF側 `generate()` と `_invoke_text()` で `processed`, `model_inputs`, `outputs`, `inputs` を使用後に `del`。
  - **PIL/BytesIO解放**: GGUF側 `_pil_to_base64_png()` で BytesIO を `finally` で close。`run()` で `pil_images` をエンコード後に `del`。
  - **モデルパス切替時の解放**: HF PromptEnhancer の `process()` で textモデル⇔QwenVLパス切替時に他方のモデルを解放。`_load_text_model()` でモデル入替時に `gc.collect()` / `empty_cache()` を実行。
  - **GGUF例外安全**: `_load_model()` で Llama 初期化失敗時に `chat_handler` をクリーンアップ。
  - **GGUF PromptEnhancer**: `clear()` に `gc.collect()` / `torch.cuda.empty_cache()` を追加（VLベースクラスと同等）。
- **Device handling**: `get_device_info()` detects CUDA/MPS/CPU; `normalize_device_choice()` validates device strings.
- **バックグラウンドダウンロード**: HF・GGUF両バックエンドで、モデル未ダウンロード時はバックグラウンドスレッドでダウンロードを開始し、即座にエラーをraiseしてユーザーに通知する。モジュールレベルの `_downloading_files` (set) と `_download_lock` でダウンロード中の重複実行を防止。ダウンロード完了/失敗時にフラグは自動解除される。ローカルファイル（`[local]`プレフィクス）はダウンロード不要のためスキップ。

### Model Path Resolution (カスタム改造)

HF・GGUF両バックエンドで `base_dir` と `extra_model_paths.yaml` の両対応に改造済み。

**共通ロジック** — 相対パスの `base_dir` は以下の優先順で解決される：
1. `folder_paths.get_folder_paths()` で先頭セグメント（例: `"LLM"`）を検索 → `extra_model_paths.yaml` 設定があればそのパスを使用
2. フォールバック → `{ComfyUI}/models/{base_dir}`

**HF側** (`AILab_QwenVL.py`):
- `hf_models.json` の `"base_dir"` を読む（デフォルト `"LLM"`）。グローバル変数 `HF_BASE_DIR` に格納。
- `_resolve_hf_base_dir()` でパス解決後、`{解決先}/{repo名}` にモデルを `snapshot_download` する。
- 元コードにあった `Qwen-VL/` サブディレクトリの付加は削除済み。

**GGUF側** (`AILab_QwenVL_GGUF.py`, `AILab_QwenVL_GGUF_PromptEnhancer.py`):
- `_resolve_base_dir()` を改造。HF側と同じロジックで `folder_paths` → `{ComfyUI}/models/{base_dir}` にフォールバック。

### Local GGUF File Scanning

`base_dir`（デフォルト `models/LLM`）以下を再帰スキャンし、カタログ (`gguf_models.json`) 未登録の `.gguf` ファイルをドロップダウンに `[local] 相対パス` として追加する。`mmproj*.gguf` ファイルはスキップされる（モデル本体ではないため）。

- **VLノード** (`AILab_QwenVL_GGUF.py`): モジュール読み込み時に `_scan_local_gguf_files()` → `LOCAL_GGUF_FILES` にキャッシュ。`_resolve_model_entry()` と `_load_model()` で `LOCAL_PREFIX` 判定して直接パスを使用。同ディレクトリの `*mmproj*.gguf`（中間一致）を自動検出してVision対応。
- **PromptEnhancer** (`AILab_QwenVL_GGUF_PromptEnhancer.py`): `INPUT_TYPES` 呼び出し時に `_scan_local_gguf_files(catalog)` でスキャン。`_resolve_model_path()` で `LOCAL_PREFIX` 判定して直接パスを返却。mmproj不要（テキスト専用）。
- **カタログ未登録フォールバック**: APIから `[local]` プレフィクスなしのモデル名が渡され、カタログにも該当がない場合、自動的に `[local] {model_name}` で `LOCAL_GGUF_FILES` を再検索する。VLノード側は `_resolve_model_entry()` 内で再帰呼び出し、PromptEnhancer側は `_resolve_model_path()` 内でローカルスキャン結果を照合する。`_load_model()` のローカルファイル判定は `resolved.repo_id is None` で行う。
- ローカルファイルは `repo_id=None` のためダウンロード処理はスキップされる。パラメータのデフォルト値（ctx=32768, gpu_layers=-1 等）はAdvancedノードのUIで上書き可能。

### Configuration Files

- `hf_models.json` — HuggingFace model repo IDs, VRAM requirements, preset/system prompts, `base_dir`（モデル保存先）
- `gguf_models.json` — GGUF model paths, context lengths, GPU layer defaults, `base_dir`（モデル保存先）
- `custom_models_example.json` — template for user-added models (merged at runtime if `custom_models.json` exists)
- `AILab_System_Prompts.json` — system prompt presets

### Max Tokens 設定

`max_tokens` の上限値は全ノード（Simple/Advanced × HF/GGUF）で **32,768** に設定。Qwen3.5 の推奨 max_new_tokens（標準タスク 32K、複雑タスク 80K）に基づく。Thinkingモデルでは `<think>思考</think>回答` の全体がこの予算内で生成されるため、低い値では思考部分で打ち切られる。

GGUF側のデフォルト値:
- `ctx`: 32768（画像トークン + テキスト + 生成出力を収容するため。テキスト専用なら8192でも可）
- `n_batch`: 8192（`image_max_tokens` 以上が必要な制約あり）
- `image_max_tokens`: 8192（高解像度画像対応）
- `image_min_tokens`: 1024（グラウンディング/OCRタスクの最低推奨値）

### 画像自動リサイズ（GGUF側）

GGUF側の `run()` で、入力画像のトークン数が `ctx` 予算を超える場合に自動縮小する仕組みを持つ。Qwen2VLのMROPE（n_pos_per_embd=3）ではKVキャッシュの `seq_add` がサポートされないため、コンテキストシフトが発生するとプロセスが強制終了する。これを防ぐためのガード。**Gemma 4 はMROPEを使わないためこの自動リサイズをスキップする**（`is_gemma=True` の場合 `run()` 内で分岐）。

- **トークン推定**: `_estimate_image_tokens(w, h)` = `ceil(H/28) * ceil(W/28)`（14pxパッチ × 2x2マージ）
- **予算計算**: `ctx - max_tokens - テキストオーバーヘッド(256)` を画像枚数で均等分割し、`image_max_tokens` とのmin値を各画像の上限とする
- **リサイズ**: `_resize_image_to_token_budget()` が28px単位にアラインしつつLANCZOS縮小。リサイズ時はコンソールにログ出力

### Thinking モード制御

全ノード（Simple/Advanced × HF/GGUF）に `enable_thinking` (BOOLEAN, default=False) ウィジェットがある。Qwen3-VL-*-Thinking モデル向け。

**HF Transformers側** (`AILab_QwenVL.py`):
- `apply_chat_template` に `chat_template_kwargs={"enable_thinking": enable_thinking}` を渡す。テンプレート側が `enable_thinking=False` のとき `/no_think` トークンを自動挿入する。
- `enable_thinking=None`（非Thinkingモデル）の場合は `chat_template_kwargs` 自体を送らず、テンプレート側のデフォルト動作に任せる。

**GGUF側** (`AILab_QwenVL_GGUF.py`):
- **Qwen系**: llama-cpp-python にはテンプレート制御フラグがないため、ユーザープロンプト先頭に `/think` または `/no_think` トークンを直接挿入する。ハンドラ init には `force_reasoning=False` を渡す。
- **Gemma 4系**: プレフィックス注入はスキップし、`Gemma4ChatHandler` にも thinking 関連 kwargs を渡さない（`force_reasoning` / `enable_thinking` はフォーク側 v0.3.35 の実装で親クラスが `TypeError` を投げるため）。現状 `enable_thinking` トグルは Gemma 経路では無効。将来フォーク側が安定したkwarg名を公開したら追加予定。Gemma 4 のthinkingはフォーク側の仕様で 31B / 26BA4B バリアントのみ対応（E2B / E4B は非対応）。

> **注意**: `Gemma4ChatHandler.__init__` は `**kwargs` を受け取り親クラスで runtime 検証するため、`inspect.signature` ベースの `_filter_kwargs_for_callable` ではフィルタできない。ハンドラ毎に必要最小限の kwargs だけ明示的に構築すること。

**出力側**（共通）:
- `AILab_OutputCleaner.py` の `clean_model_output()` が `<think>...</think>` ブロック、不完全な `<think>` / `</think>` タグを正規表現で除去する。
- 閉じタグのない `<think>`（`max_tokens` 不足で途中打ち切り時に発生）は `<think>` 以降のテキストをすべて除去する。
- **Gemma 4 チャンネル除去**: Gemma 4 は reasoning を `<|channel|>thought ... <|channel|>` で囲むため、`_GEMMA_CHANNEL_BLOCK_RE` でペアマッチしてブロックごと除去する。パイプの位置が揺れるレンダリング（`<|channel>`, `<channel|>`, `<|channel|>`）すべてにマッチ。ペア除去後に残った単独マーカーは、先頭にある場合は途中打ち切りの opener とみなして以降を全削除、そうでなければ closer とみなして最終マーカー以降のテキストを残す。`<|turn|>` / `<|start_of_turn|>` / `<|end_of_turn|>` 系リークトークンは `_GEMMA_TURN_TOKEN_RE` で個別に除去。

### Stop Words

Advancedノード（HF/GGUF両方）に `stop_words` (STRING) ウィジェットがある。カンマ区切りで複数のストップシーケンスを指定可能。空欄ならデフォルト動作。

- **HF側**: 各ストップワードをトークナイズし、末尾トークンIDを `eos_token_id` リストに追加。
- **GGUF側**: `create_chat_completion` の `stop` リストにそのまま文字列として追加。デフォルト値はハンドラ別に切替: Qwen系は `["<|im_end|>", "<|im_start|>"]`、Gemma 4系は `["<|turn>", "<|channel>", "<end_of_turn>", "<start_of_turn>"]`。

### UI

`web/js/appearance.js` registers a ComfyUI extension for custom node colors and sizing.

## Conventions

- All node classes follow ComfyUI's `INPUT_TYPES`, `RETURN_TYPES`, `FUNCTION`, `CATEGORY` class-variable pattern.
- Model files のデフォルト保存先は `models/LLM/`（HF・GGUF共通）。`base_dir` や `extra_model_paths.yaml` で変更可能。
- License: GPL-3.0.
