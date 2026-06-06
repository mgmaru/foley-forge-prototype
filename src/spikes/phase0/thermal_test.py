"""Phase 0 サーマル/速度テスト：連続生成 vs 冷却ありで「生成時間」がどう変わるかを測る。

同一条件(25step・5s・seed0)で2フェーズ比較：
  - BURST : 冷却なしで連続生成 → 生成時間が回を追って増えるか（= サーマルスロットリング）
  - COOLED: 各生成前に冷却sleepを挟む → 生成時間が低く保たれるか
差が大きければ「熱で遅くなっている（冷却が効く）」、小さければ「熱の影響は小さい」。

おまけ：各生成の rms を出力。全回で同一なら「温度に依らず音は同じ＝決定論」も同時に確認できる
（音声に“熱ノイズ”は乗らない。サーマルは速度だけの話）。

実行は必ずサンドボックスの外で（MPS。memory: mps-blocked-by-sandbox）。
  使い方:  .venv/bin/python src/spikes/phase0/thermal_test.py
"""
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import time

import numpy as np
import torch
from diffusers import StableAudioPipeline

from smoke_generate import MODEL_DIR, apply_final_step_noise_guard, load_prompt

# --- 測定条件 ---
PROMPT_ID = "env_rain_forest"
STEPS = 25            # 作業点候補（聴き比べで実用下限）
DURATION = 5.0
GUIDANCE = 7.0
SEED = 0
N_BURST = 4          # 連続（冷却なし）回数
N_COOLED = 2         # 冷却ありの回数
COOLDOWN_S = 120     # 各冷却生成の前に待つ秒数（ファンレスなので長め）


def generate(pipe, prompt: str, negative: str) -> tuple[float, float]:
    """1本生成し (経過秒, rms) を返す。rms は決定論確認用（同条件なら毎回同じはず）。"""
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
    print(f"[cfg] {STEPS}step / {DURATION}s / seed{SEED}  (burst={N_BURST}, cooled={N_COOLED}, cooldown={COOLDOWN_S}s)")

    # warmup（計測対象外・シェーダコンパイル）
    print("[warmup] ...", flush=True)
    generate(pipe, p["prompt"], p["negative_prompt"])

    # BURST：冷却なしで連続
    burst = []
    print("\n=== BURST（冷却なし・連続） ===", flush=True)
    for i in range(N_BURST):
        dt, rms = generate(pipe, p["prompt"], p["negative_prompt"])
        burst.append(dt)
        print(f"  burst {i + 1}: {dt:6.1f}s  rms={rms:.6f}", flush=True)

    # COOLED：各生成前に冷却
    cooled = []
    print(f"\n=== COOLED（各生成前に {COOLDOWN_S}s 冷却） ===", flush=True)
    for i in range(N_COOLED):
        print(f"  cooling {COOLDOWN_S}s ...", flush=True)
        time.sleep(COOLDOWN_S)
        dt, rms = generate(pipe, p["prompt"], p["negative_prompt"])
        cooled.append(dt)
        print(f"  cooled {i + 1}: {dt:6.1f}s  rms={rms:.6f}", flush=True)

    # まとめ
    b_avg = sum(burst) / len(burst)
    c_avg = sum(cooled) / len(cooled)
    print("\n==== まとめ ====")
    print(f"BURST  : {[round(x, 1) for x in burst]}  avg={b_avg:.1f}s")
    print(f"COOLED : {[round(x, 1) for x in cooled]}  avg={c_avg:.1f}s")
    print(f"差（burst最後 - cooled平均）: {burst[-1] - c_avg:+.1f}s")
    print(">>> burst が回毎に増え cooled が低く一定なら＝スロットリング（冷却が効く）")
    print(">>> 差が小さければ＝熱の影響は小さい（冷却しても変わらない）")
    print(">>> rms が全回同一なら＝温度に依らず音は同じ（決定論・“熱ノイズ”は乗らない）")


if __name__ == "__main__":
    main()
