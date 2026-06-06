"""Phase 0 スモークテスト：MPS 上で SAO 1.0 が実際に音を生成できるかだけを確かめる使い捨てコード。

目的（risk-first）：本番の生成スクリプト（schema.py でメタデータ保存＋CFGスイープ）を書く前に、
存在的リスクの最後の関門「MPS で拡散サンプリングが回って“音が出る”か」を最小コードで潰す。

このスクリプトは意図的に最小：
  - プロンプトは prompts.yaml の①「雨の森」を1本だけ。
  - メタデータ保存・集計はしない（schema.py を使うのは本番スクリプト）。wav を1本吐くだけ。
  - 設定は速度優先で、本番グリッド（§9.1）の値ではない。
  - 最初の1本は cold（モデルロード＋MPSシェーダのコンパイル）で遅い＝想定どおり。

実行は必ずサンドボックスの外で（殻の中だと MPS が使えない。memory: mps-blocked-by-sandbox）。
  使い方:  .venv/bin/python src/spikes/phase0/smoke_generate.py
"""
from __future__ import annotations

# MPS 未対応 op があっても落とさず CPU へ退避させ、生成を完走させる（torch import より前に設定）。
# 退避が起きると stderr に警告が出る＝§4「fallback の有無」を観察するシグナルになる。
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import time
from pathlib import Path

import soundfile as sf
import torch
import yaml
from diffusers import StableAudioPipeline

# --- パス（cwd に依存しないよう、このファイルの位置から解決）---
HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[3]                       # src/spikes/phase0/ から3つ上＝リポジトリ root
PROMPTS_YAML = HERE.parent / "prompts.yaml"
MODEL_DIR = REPO_ROOT / "src" / "models" / "stable-audio-open-1.0"
OUT_DIR = REPO_ROOT / "src" / "outputs" / "phase0" / "smoke"   # gitignore対象（FF-D011）

# --- スモーク設定（速度優先・本番グリッドの値ではない）---
PROMPT_ID = "env_rain_forest"   # prompts.yaml の①「雨の森」
NUM_STEPS = 100                 # 推論ステップ。少ないほど速い（遅ければ 50 等へ下げてよい）
GUIDANCE = 7.0                  # CFG。SAO の既定値
SEED = 0
DTYPE = torch.float32           # MPS は float32（float16/64 でなく。§4 で確認済み）


def apply_final_step_noise_guard() -> None:
    """SAO × この diffusers の既知バグ回避：拡散の **最終ステップ**で SDE ノイズ生成器
    (torchsde BrownianTree) が境界を踏み、NaN／無限再帰を起こす問題を、最終ステップの
    SDE ノイズだけ 0 にして回避する。

    根拠：denoising 完了時の SDE 確率項は物理的に ~0 なので 0 で正しい。これにより
    SAO 既定の `final_sigmas_type="zero"`（クリーンな出力）のまま完走できる。
    通常ステップ（区間が正の範囲内）は元の挙動のまま。詳細は research/debugging/ に記録予定。
    """
    from diffusers.schedulers import scheduling_dpmsolver_sde as sde

    if getattr(sde.BrownianTreeNoiseSampler, "_ff_guarded", False):
        return  # 二重適用を防ぐ
    _orig = sde.BrownianTreeNoiseSampler.__call__

    def _guarded(self, sigma, sigma_next):
        t0 = self.transform(torch.as_tensor(sigma))
        t1 = self.transform(torch.as_tensor(sigma_next))
        # 退化(|t1-t0|≈0)／境界外(<=0)＝最終ステップ → SDEノイズ0
        if float((t1 - t0).abs()) < 1e-9 or float(t1) <= 0.0 or float(t0) <= 0.0:
            ref = self.tree(t0.clamp(min=1e-3), t0.clamp(min=1e-3) + 1e-3)
            return torch.zeros_like(ref)
        return _orig(self, sigma, sigma_next)

    sde.BrownianTreeNoiseSampler.__call__ = _guarded
    sde.BrownianTreeNoiseSampler._ff_guarded = True


def load_prompt(prompt_id: str) -> dict:
    """prompts.yaml から指定 id のプロンプト1件を返す。"""
    data = yaml.safe_load(PROMPTS_YAML.read_text(encoding="utf-8"))
    for p in data["prompts"]:
        if p["id"] == prompt_id:
            return p
    raise SystemExit(f"prompt id が見つかりません: {prompt_id}")


def main() -> None:
    # 0) デバイス確認（殻の外で動いているか）
    if not torch.backends.mps.is_available():
        raise SystemExit(
            "MPS が使えません。サンドボックスの外で実行していますか？（memory: mps-blocked-by-sandbox）"
        )

    apply_final_step_noise_guard()  # SAO×diffusers の最終ステップ NaN/再帰バグ回避（上の関数参照）

    p = load_prompt(PROMPT_ID)
    duration = float(p["duration"])               # 「雨の森」は 10s（prompts.yaml の duration）
    print(f"[prompt] {p['scene_ja']} / {duration}s")
    print(f"[prompt] {p['prompt']!r}")

    # 1) モデル読み込み → MPS（最初の1本は cold＝ここが遅い）
    t0 = time.perf_counter()
    pipe = StableAudioPipeline.from_pretrained(
        MODEL_DIR.as_posix(), torch_dtype=DTYPE, local_files_only=True
    )
    pipe = pipe.to("mps")
    load_sec = time.perf_counter() - t0
    sr = pipe.vae.sampling_rate                   # SAO 1.0 は 44100
    print(f"[load] {load_sec:.1f}s / sample_rate={sr}")

    # 2) 生成（再現性のため初期ノイズは CPU generator で固定＝diffusers 推奨）
    gen = torch.Generator("cpu").manual_seed(SEED)
    t1 = time.perf_counter()
    result = pipe(
        prompt=p["prompt"],
        negative_prompt=p["negative_prompt"],
        num_inference_steps=NUM_STEPS,
        audio_end_in_s=duration,
        num_waveforms_per_prompt=1,
        guidance_scale=GUIDANCE,
        generator=gen,
    )
    gen_sec = time.perf_counter() - t1

    # 3) 保存（audios[0] は (channels, samples) → soundfile 用に転置して (samples, channels)）
    audio = result.audios[0].T.float().cpu().numpy()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_wav = OUT_DIR / f"{PROMPT_ID}_seed{SEED}.wav"
    sf.write(out_wav.as_posix(), audio, sr)

    # 4) サマリ（cold を含む値であることに注意。本当の速度は warm の2本目以降）
    rtf = gen_sec / duration                      # RTF＝生成時間 ÷ 音長（<1 で実時間より速い・§13）
    print("---- summary ----")
    print(f"gen_time : {gen_sec:.1f}s  (cold 含む)")
    print(f"rtf      : {rtf:.2f}      (<1 なら実時間より速い)")
    print(f"output   : {out_wav}")
    print(">>> この wav を再生して、雨／森系の“方向”に聞こえるか確認してください")
    print(">>> 併せて stderr に 'fallback' 警告が出ていないかも確認（多発＝MPS非対応opのサイン）")


if __name__ == "__main__":
    main()
