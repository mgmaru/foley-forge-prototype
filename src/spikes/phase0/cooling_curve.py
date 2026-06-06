"""Phase 0 冷却カーブ：休憩を「最小限」にすると総スループットが勝つか検証する。

thermal_test の続き。あの 120秒は恣意値だった。ここでは複数の休憩カデンスを比較し、
「短い休憩で素の速度を保てるか／総時間（生成＋休憩）で連続実行に勝てるか」を測る。

各フェーズ：同一条件(25step・5s・seed0)で GENS 本を、各生成の前に rest 秒の休憩を挟んで生成。
  - continuous(rest=0) … 連続＝熱が溜まりスロットリング（基準）
  - rest_15s / rest_30s … 短い休憩を挟む
判定指標 = **1本あたり総時間（gen + rest）の定常値**（各フェーズ末尾2本の平均）。
  rest を入れて gen が十分速くなり、(gen + rest) < continuous の gen なら → 短い休憩が“勝つ”。

おまけ：rms を出して決定論（温度に依らず音は同じ）も再確認。
実行はサンドボックス外（MPS）。  使い方:  .venv/bin/python src/spikes/phase0/cooling_curve.py
"""
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import time

import numpy as np
import torch
from diffusers import StableAudioPipeline

from smoke_generate import MODEL_DIR, apply_final_step_noise_guard, load_prompt

PROMPT_ID = "env_rain_forest"
STEPS = 25
DURATION = 5.0
GUIDANCE = 7.0
SEED = 0
GENS = 4                              # 各カデンスの本数
PHASES = [("continuous", 0), ("rest_15s", 15), ("rest_30s", 30)]
INTER_PHASE_COOL = 60                 # フェーズ間のリセット用冷却


def generate(pipe, prompt: str, negative: str) -> tuple[float, float]:
    gen = torch.Generator("cpu").manual_seed(SEED)
    t = time.perf_counter()
    res = pipe(
        prompt=prompt, negative_prompt=negative, num_inference_steps=STEPS,
        audio_end_in_s=DURATION, num_waveforms_per_prompt=1, guidance_scale=GUIDANCE, generator=gen,
    )
    dt = time.perf_counter() - t
    a = np.asarray(res.audios[0].float().cpu().numpy(), dtype=np.float64)
    return dt, float(np.sqrt(np.mean(a**2)))


def main() -> None:
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS が使えません。サンドボックスの外で実行していますか？（memory: mps-blocked-by-sandbox）")
    apply_final_step_noise_guard()
    p = load_prompt(PROMPT_ID)
    pipe = StableAudioPipeline.from_pretrained(
        MODEL_DIR.as_posix(), torch_dtype=torch.float32, local_files_only=True
    ).to("mps")
    print(f"[cfg] {STEPS}step / {DURATION}s / seed{SEED}  GENS={GENS}/phase  inter-cool={INTER_PHASE_COOL}s")

    print("[warmup] ...", flush=True)
    generate(pipe, p["prompt"], p["negative_prompt"])

    results = {}
    for pi, (name, rest) in enumerate(PHASES):
        if pi > 0:
            print(f"\n--- フェーズ間冷却 {INTER_PHASE_COOL}s ---", flush=True)
            time.sleep(INTER_PHASE_COOL)
        print(f"\n=== {name}（各生成前に {rest}s 休憩） ===", flush=True)
        gens = []
        for i in range(GENS):
            if rest > 0:
                time.sleep(rest)
            dt, rms = generate(pipe, p["prompt"], p["negative_prompt"])
            gens.append(dt)
            print(f"  {name} {i + 1}: gen {dt:6.1f}s  (+rest {rest}s = {dt + rest:6.1f}s/本)  rms={rms:.6f}", flush=True)
        results[name] = (rest, gens)

    # まとめ：定常（末尾2本平均）の「1本あたり総時間」で比較
    print("\n==== まとめ（定常＝末尾2本平均） ====")
    print("phase       | rest | gen定常 | 総/本(gen+rest)")
    best = None
    for name, (rest, gens) in results.items():
        steady = sum(gens[-2:]) / 2
        total = steady + rest
        print(f"{name:11} | {rest:4d} | {steady:6.1f}s | {total:6.1f}s")
        if best is None or total < best[1]:
            best = (name, total)
    print(f"\n>>> 1本あたり総時間が最小＝最良カデンス: {best[0]}（{best[1]:.1f}s/本）")
    print(">>> 短い休憩(15/30s)の総/本 < continuous の総/本 なら → 休憩を入れた方が速い")


if __name__ == "__main__":
    main()
