"""Phase 0 共通エンジン：デバイス検出・モデル読込・生成・計測を1か所に集約（再利用用）。

OS やモデルを変えても使い回せるよう、以下を集約する：
  - device 検出（§3：cuda→mps→cpu を可用性で判定）        ← OS変更に効く
  - device 別 dtype（§4：mps/cpu=float32, cuda=float16）   ← OS変更に効く
  - device 別メモリ計測（§4.1：mps_* / cuda_* / rss）       ← OS変更に効く
  - モデル読込（＋SAO最終ステップ guard）／生成／prompts.yaml 読込

今は SAO 1.0 / MPS 専用。
  - **OS 変更**：`pick_device` / `device_dtype` / `measure_memory` / `env_info` が吸収（汎用）。
  - **モデル追加**：`load_pipeline`（読込）と `generate`（pipe呼び出し引数・出力の取り出し）の**両方が
    SAO 固有**なので、両者を含む「アダプタ」を1組新たに足す必要がある。本格的な複数モデル抽象は
    Phase 1 / FF-D003 の仕事で、Phase 0 はこの最小版に留める（§3）。

実行は必ずサンドボックスの外で（MPS。memory: mps-blocked-by-sandbox）。
"""
from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # torch import より前

import platform
import sys
import time
from pathlib import Path

import torch
import yaml

HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[3]                        # src/spikes/phase0/ から3つ上＝リポジトリ root
PROMPTS_YAML = HERE.parent / "prompts.yaml"


# ── デバイス / dtype（§3・§4）──────────────────────────────────────────────
def pick_device() -> str:
    """使える計算バックエンドを優先順（cuda→mps→cpu）で返す。OS名でなく可用性で判定（§3）。"""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def device_dtype(device: str) -> torch.dtype:
    """device 別の既定 dtype（cuda=float16／mps・cpu=float32。§4 で確認）。"""
    return torch.float16 if device == "cuda" else torch.float32


def env_info(device: str) -> dict:
    """EnvMeta 用の環境情報（再現性のため版を記録・§5.4(5)）。"""
    import diffusers

    os_str = f"macOS {platform.mac_ver()[0]}" if platform.system() == "Darwin" else platform.platform()
    return {
        "device": device,
        "os": os_str,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "diffusers": diffusers.__version__,
    }


# ── プロンプト（prompts.yaml）──────────────────────────────────────────────
def load_prompts() -> list[dict]:
    """prompts.yaml の全プロンプトを返す。"""
    return yaml.safe_load(PROMPTS_YAML.read_text(encoding="utf-8"))["prompts"]


def load_prompt(prompt_id: str) -> dict:
    """指定 id のプロンプト1件を返す。"""
    for p in load_prompts():
        if p["id"] == prompt_id:
            return p
    raise SystemExit(f"prompt id が見つかりません: {prompt_id}")


# ── SAO 最終ステップ guard（research/debugging 参照）──────────────────────
def apply_final_step_noise_guard() -> None:
    """SAO × diffusers の最終ステップ SDEノイズ（torchsde）が NaN／無限再帰を起こす問題を回避。

    最終ステップ（区間が退化/範囲外）では SDEノイズを 0 にする（denoising 完了時の SDE 項は ~0）。
    既定 `final_sigmas_type="zero"`（クリーンな出力）のまま完走できる。通常ステップは無改変。
    詳細: research/debugging/stable-audio-final-step-nan-recursion.md
    """
    from diffusers.schedulers import scheduling_dpmsolver_sde as sde

    if getattr(sde.BrownianTreeNoiseSampler, "_ff_guarded", False):
        return
    _orig = sde.BrownianTreeNoiseSampler.__call__

    def _guarded(self, sigma, sigma_next):
        t0 = self.transform(torch.as_tensor(sigma))
        t1 = self.transform(torch.as_tensor(sigma_next))
        if float((t1 - t0).abs()) < 1e-9 or float(t1) <= 0.0 or float(t0) <= 0.0:
            ref = self.tree(t0.clamp(min=1e-3), t0.clamp(min=1e-3) + 1e-3)
            return torch.zeros_like(ref)
        return _orig(self, sigma, sigma_next)

    sde.BrownianTreeNoiseSampler.__call__ = _guarded
    sde.BrownianTreeNoiseSampler._ff_guarded = True


# ── SAO アダプタ：モデル読込 / 生成（★モデル固有。別モデル追加時はこの1組を新規に）──────
def model_dir(model_name: str) -> Path:
    """モデル名 → ローカル配置パス（src/models/<name>・FF-D004/D011）。"""
    return REPO_ROOT / "src" / "models" / model_name


def load_pipeline(model_name: str, device: str, dtype: torch.dtype):
    """ローカルのモデルを読み込み、guard を当てて device へ載せる。(pipe, load_sec) を返す。

    将来モデルを増やすときは、ここに `model_name` 分岐（適切な Pipeline クラス）を足す。
    """
    from diffusers import StableAudioPipeline

    apply_final_step_noise_guard()
    t = time.perf_counter()
    pipe = StableAudioPipeline.from_pretrained(
        model_dir(model_name).as_posix(), torch_dtype=dtype, local_files_only=True
    ).to(device)
    return pipe, time.perf_counter() - t


def generate(pipe, *, prompt: str, negative_prompt: str, steps: int, cfg: float, duration: float, seed: int):
    """1本生成し (audio[(samples,channels) ndarray], gen_sec) を返す。

    初期ノイズは CPU generator で固定＝再現性（diffusers 推奨）。
    audios[0] は (channels, samples) なので転置して soundfile 用に (samples, channels) で返す。

    ★モデル固有：pipe(...) のキーワード（audio_end_in_s・num_waveforms_per_prompt）と出力の
      取り出し（res.audios[0].T）は StableAudioPipeline 前提。別モデルでは引数名・出力形が異なるため、
      その場合は別アダプタ（この generate の対応版）を用意する。
    """
    g = torch.Generator("cpu").manual_seed(seed)
    t = time.perf_counter()
    res = pipe(
        prompt=prompt, negative_prompt=negative_prompt, num_inference_steps=steps,
        audio_end_in_s=duration, num_waveforms_per_prompt=1, guidance_scale=cfg, generator=g,
    )
    gen_sec = time.perf_counter() - t
    audio = res.audios[0].T.float().cpu().numpy()
    return audio, gen_sec


# ── メモリ計測（§4.1）──────────────────────────────────────────────────────
def measure_memory(device: str) -> dict:
    """device 別メモリを MemoryMeta のキーに合わせた dict で返す（該当しない側は None・§4.1）。"""
    import psutil

    m = {
        "mps_current_allocated_bytes": None,
        "mps_driver_allocated_bytes": None,
        "mps_recommended_max_bytes": None,
        "cuda_max_allocated_bytes": None,
        "nvidia_smi_used_bytes": None,
        "rss_bytes": int(psutil.Process().memory_info().rss),
    }
    if device == "mps":
        try:
            m["mps_current_allocated_bytes"] = int(torch.mps.current_allocated_memory())
            m["mps_driver_allocated_bytes"] = int(torch.mps.driver_allocated_memory())
            m["mps_recommended_max_bytes"] = int(torch.mps.recommended_max_memory())
        except Exception:
            pass
    elif device == "cuda":
        try:
            m["cuda_max_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
        except Exception:
            pass
    return m
