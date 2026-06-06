"""Phase 0 グリッド生成：prompts.yaml × CFG × seed を生成し、schema.py で run/clip を保存。

Phase 0 の「捕獲」基盤。engine.py（device検出・読込・生成・計測）と schema.py（出力契約）を使う。
出力：src/outputs/phase0/<run_id>/ に run.json ＋ audio/clip_NNNN.wav ＋ metadata/clip_NNNN.json。

2段階運用：
  - 既定（--subset 相当）：小サブセットでメタデータ機構を検証（数本・数分）
  - --full              ：本番グリッド（全prompt × CFG3 × seed3）

生成条件：25step（実測の作業点）・CFG{3.5,5.0,7.0}・seed{0,1,2}・連続実行（Q2確定）・guard。
実行は必ずサンドボックスの外で（MPS。memory: mps-blocked-by-sandbox）。
  使い方:  .venv/bin/python src/spikes/phase0/generate_grid.py          # サブセット検証
           .venv/bin/python src/spikes/phase0/generate_grid.py --full   # 本番グリッド
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime

import soundfile as sf

import engine
from schema import ClipMeta, EnvMeta, GridMeta, MemoryMeta, MpsFallback, RunMeta, to_dict

MODEL_NAME = "stable-audio-open-1.0"
STEPS = 25                          # 実測の作業点（§9.1 の「モデル既定」を更新）
CFG_SET = [3.5, 5.0, 7.0]           # §9.1 スイープ
SEED_SET = [0, 1, 2]                # §9.1

# 小サブセット（検証用）：長短2プロンプト × CFG2 × seed1 ＝ 4本
SUBSET_PROMPT_IDS = ["env_rain_forest", "foley_cloth_turn"]
SUBSET_CFG = [3.5, 7.0]
SUBSET_SEED = [0]


def write_json(path, dataclass_obj) -> None:
    path.write_text(json.dumps(to_dict(dataclass_obj), ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="本番グリッド（全prompt×CFG3×seed3）。未指定なら小サブセット検証")
    full = ap.parse_args().full   # True=本番グリッド全件 ／ False(既定)=小サブセット検証

    device = engine.pick_device()
    if device == "cpu":
        # cpu は GPU フォールバック＝非常に遅い。mps/cuda はどちらも GPU なので警告しない
        print("[warn] device=cpu は非常に遅い。GPU(mps/cuda)・サンドボックス外での実行を推奨")
    dtype = engine.device_dtype(device)

    all_prompts = engine.load_prompts()
    if full:   # 本番：全prompt × CFG3 × seed3
        prompts, cfgs, seeds = all_prompts, CFG_SET, SEED_SET
    else:      # 検証：少数のサブセット
        prompts = [p for p in all_prompts if p["id"] in SUBSET_PROMPT_IDS]
        cfgs, seeds = SUBSET_CFG, SUBSET_SEED
    tasks = [(p, cfg, seed) for p in prompts for cfg in cfgs for seed in seeds]
    print(f"[mode] {'FULL' if full else 'SUBSET'} / {len(prompts)}prompt × {len(cfgs)}cfg × {len(seeds)}seed = {len(tasks)}本")

    pipe, load_sec = engine.load_pipeline(MODEL_NAME, device, dtype)
    sr = int(pipe.vae.sampling_rate)
    print(f"[load] {load_sec:.1f}s / device={device} / dtype={str(dtype).split('.')[-1]} / sr={sr}")

    run_id = datetime.now().strftime("%Y%m%dT%H%M") + f"_sao10_{device}"
    out_dir = engine.REPO_ROOT / "src" / "outputs" / "phase0" / run_id
    (out_dir / "audio").mkdir(parents=True, exist_ok=True)
    (out_dir / "metadata").mkdir(parents=True, exist_ok=True)
    print(f"[out] {out_dir}")

    run_written = False
    t_all = 0.0
    for i, (p, cfg, seed) in enumerate(tasks, start=1):
        duration = float(p["duration"])
        audio, gen_sec = engine.generate(
            pipe, prompt=p["prompt"], negative_prompt=p["negative_prompt"],
            steps=STEPS, cfg=cfg, duration=duration, seed=seed,
        )
        t_all += gen_sec
        channels = int(audio.shape[1]) if audio.ndim > 1 else 1

        # run.json は最初のクリップ後に1回だけ（channels が確定するため）
        if not run_written:
            run = RunMeta(
                run_id=run_id, created_at=datetime.now().isoformat(timespec="seconds"),
                model=MODEL_NAME, model_revision="(local)",
                env=EnvMeta(**engine.env_info(device)),
                grid=GridMeta(cfg_set=cfgs, seed_set=seeds, steps=STEPS),
                dtype=str(dtype).split(".")[-1], sample_rate=sr, channels=channels,
                model_load_sec=load_sec,
            )
            write_json(out_dir / "run.json", run)
            run_written = True

        clip_id = f"clip_{i:04d}"
        audio_rel = f"audio/{clip_id}.wav"
        sf.write((out_dir / audio_rel).as_posix(), audio, sr)

        clip = ClipMeta(
            clip_id=clip_id, run_id=run_id,
            prompt_id=p["id"], tier=p["tier"], source=p["source"], temporal=p["temporal"],
            prompt=p["prompt"], negative_prompt=p["negative_prompt"],
            cfg=cfg, seed=seed, duration_sec=duration, audio_file=audio_rel,
            gen_time_sec=round(gen_sec, 2), is_cold=(i == 1), rtf=round(gen_sec / duration, 2),
            memory=MemoryMeta(**engine.measure_memory(device)),
            mps_cpu_fallback=MpsFallback(),
        )
        write_json(out_dir / "metadata" / f"{clip_id}.json", clip)
        print(f"  [{i:3d}/{len(tasks)}] {clip_id} {p['id']:<20} cfg{cfg} seed{seed} {duration:.0f}s -> {gen_sec:5.1f}s (rtf {gen_sec/duration:.1f})", flush=True)

    print(f"\n==== 完了 ====  {len(tasks)}本 / 生成合計 {t_all/60:.1f}分")
    print(f"出力: {out_dir}")
    print(">>> run.json ＋ metadata/clip_*.json ＋ audio/clip_*.wav を確認")


if __name__ == "__main__":
    main()
