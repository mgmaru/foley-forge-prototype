"""Phase 0 速度スイープ：num_inference_steps × (速度・健全性) を測り、本番グリッドが回る step を探る。

雨の森プロンプト1本を複数の step 数で生成し、各々の **warm 生成時間・RTF・L0 DSP健全性** を
測って wav 保存する。耳で「方向性が保てる最小 step」を判定するための材料を作る（Q2＝速度）。

工夫：
  - モデルは1回だけロード。最初に短い warmup を回して MPS シェーダをコンパイル → 以降は warm。
  - duration は短め(5s)＝per-step を抑え、サーマルの影響と総時間を圧縮（方向性は5sでも判定可）。
  - 生成条件は smoke と同一（guard／float32／CPU generator）を **再利用**（smoke_generate から import）。

実行は必ずサンドボックスの外で（MPS。memory: mps-blocked-by-sandbox）。
  使い方:  .venv/bin/python src/spikes/phase0/speed_sweep.py
"""
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # torch import より前（smoke と同じ）

import math
import time

import numpy as np
import soundfile as sf
import torch
from diffusers import StableAudioPipeline

# smoke と同じ部品を再利用（guard・プロンプト読込・パス）
from smoke_generate import MODEL_DIR, OUT_DIR, apply_final_step_noise_guard, load_prompt

# --- スイープ設定 ---
PROMPT_ID = "env_rain_forest"
STEP_LIST = [100, 50, 25, 10]   # 多い→少ない（100=smokeで方向性OKの基準）
DURATION = 5.0                  # 速度比較用に短め（per-step を抑える）
GUIDANCE = 7.0
SEED = 0
SWEEP_DIR = OUT_DIR.parent / "speed_sweep"   # src/outputs/phase0/speed_sweep（gitignore対象）


def l0_check(audio: np.ndarray) -> dict:
    """L0 DSP 健全性（無音/クリップ/NaN）。peak/rms は dBFS でも返す。"""
    x = np.asarray(audio, dtype=np.float64)
    peak, rms = float(np.max(np.abs(x))), float(np.sqrt(np.mean(x**2)))
    to_db = lambda v: (20 * math.log10(v) if v > 0 else float("-inf"))
    return {
        "peak_db": to_db(peak),
        "rms_db": to_db(rms),
        "clip_pct": float(np.mean(np.abs(x) >= 0.999)) * 100,
        "nan": bool(np.isnan(x).any()),
    }


def generate(pipe, prompt: str, negative: str, steps: int) -> tuple[float, np.ndarray]:
    """1本生成し (経過秒, 波形 (samples,channels)) を返す。"""
    gen = torch.Generator("cpu").manual_seed(SEED)
    t = time.perf_counter()
    res = pipe(
        prompt=prompt, negative_prompt=negative, num_inference_steps=steps,
        audio_end_in_s=DURATION, num_waveforms_per_prompt=1, guidance_scale=GUIDANCE, generator=gen,
    )
    dt = time.perf_counter() - t
    return dt, res.audios[0].T.float().cpu().numpy()


def main() -> None:
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS が使えません。サンドボックスの外で実行していますか？（memory: mps-blocked-by-sandbox）")
    apply_final_step_noise_guard()

    p = load_prompt(PROMPT_ID)
    pipe = StableAudioPipeline.from_pretrained(
        MODEL_DIR.as_posix(), torch_dtype=torch.float32, local_files_only=True
    ).to("mps")
    sr = pipe.vae.sampling_rate
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[prompt] {p['scene_ja']} / duration={DURATION}s / sr={sr}")

    # warmup（計測しない）：MPSシェーダをコンパイルさせ、以降の計測を warm にする
    print("[warmup] 5 steps（計測対象外）...", flush=True)
    generate(pipe, p["prompt"], p["negative_prompt"], 5)

    rows = []
    for steps in STEP_LIST:
        dt, audio = generate(pipe, p["prompt"], p["negative_prompt"], steps)
        out = SWEEP_DIR / f"{PROMPT_ID}_steps{steps:03d}.wav"
        sf.write(out.as_posix(), audio, sr)
        m = l0_check(audio)
        rtf = dt / DURATION
        rows.append((steps, dt, rtf, m))
        print(
            f"[steps {steps:3d}] {dt:6.1f}s  RTF {rtf:5.2f}  "
            f"peak {m['peak_db']:6.1f}dB  rms {m['rms_db']:6.1f}dB  "
            f"clip {m['clip_pct']:.2f}%  nan {m['nan']}  -> {out.name}",
            flush=True,
        )

    # サマリ表
    print(f"\n==== サマリ（warm・duration={DURATION}s）====")
    print("steps | gen_s | RTF  | peak dB | rms dB | clip% | nan")
    for steps, dt, rtf, m in rows:
        print(f"{steps:5d} | {dt:5.1f} | {rtf:4.2f} | {m['peak_db']:7.1f} | {m['rms_db']:6.1f} | {m['clip_pct']:4.2f} | {m['nan']}")
    print(f"\n出力: {SWEEP_DIR}")
    print(">>> 各 wav を聴き比べ、雨/森の方向性が保てる最小 step を判定してください")
    print(">>> 速度は cold/サーマルで変動。RTF×実duration が本番グリッド1本の目安")


if __name__ == "__main__":
    main()
