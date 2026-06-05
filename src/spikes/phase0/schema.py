"""Phase 0 スパイクの出力スキーマ（metadata の確定契約）。

生成スクリプト（README §6 手順3）が依存する「出力の形」を dataclass で定義する。
用語: run = 1モデル×1マシンで1回まわす生成ひとまとまり（多数の clip の親）／
      clip = 音1本（1回の生成で出来る音声ファイル1つ）。
- prompts.yaml = 入力スキーマ（既存）／本モジュール = 出力スキーマ。
- 1 run = 1 ディレクトリ。run.json（run単位）＋ metadata/clip_NNNN.json（クリップ単位）の 2 層。
  run 固定の項目（model/env/dtype/sample_rate/channels/grid）は run.json に寄せ、clip.json には
  クリップごとに変わる値（cfg/seed/duration＋プロンプト＋実測）だけを持たせる（重複を避ける）。
- audio/ と metadata/ を分離し、§5.4 のブラインド判定（判定者は中立名 wav だけ聴く）を成立させる。
- 判定用 CSV（mapping/judgments）は手順7（判定）で追加するため、ここには含めない（v0）。

ディレクトリ構成（保存先 src/outputs/phase0/<run_id>/・gitignore・FF-D011）::

    <run_id>/
      run.json                  # run単位（model/env/grid/dtype/model_load_sec）を 1 回
      audio/clip_0001.wav  …    # 判定者が聴くのはここだけ（中立連番名）
      metadata/clip_0001.json … # 1クリップの全条件＋実測（model名を含む＝判定中は開かない）
      （手順7で追加）mapping.csv / judgments.csv

参照: docs/phases/phase0/README.md §9.2（残す項目）/§4.1（device別メモリAPI）/
      §9.1（スイープ条件）/§5.4(4)（ブラインドJOINキー clip_id）。
スキーマはドラフト（v0）。最初の生成が現実（native SR・steps の露出・fallback の出方など）を
返したら較正する（prompts.yaml と同じ「ドラフト→spike で調整」運用）。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

SCHEMA_VERSION = "0.1"


@dataclass
class EnvMeta:
    """実行環境（再現性のため版を記録・§5.4(5)）。"""

    device: str       # "mps" / "cuda" / "cpu"（§3 のデバイス分岐結果）
    os: str           # 例 "macOS 26.5"
    python: str       # 例 "3.12.7"
    torch: str        # 例 "2.5.1"
    diffusers: str    # 例 "0.31.0"


@dataclass
class GridMeta:
    """生成条件の grid（§9.1）。run 全体で固定の「振り方」を残す。"""

    cfg_set: list[float]                              # スイープする唯一の軸（§9.1）
    seed_set: list[int]                               # 全モデル・全CFGで共用（同一乱数＝公平）
    steps: int                                        # モデル既定（共通固定にしない・§9.1）
    prompts_file: str = "src/spikes/phase0/prompts.yaml"


@dataclass
class RunMeta:
    """run 単位のメタデータ（run.json に 1 回だけ書く）。

    cold の主成分（モデルロード）は run 単位の事象なので、ここに model_load_sec として置く。
    各クリップの cold/warm 区別は ClipMeta.is_cold で持つ（cold = model_load_sec ＋ 最初の
    クリップの gen_time_sec(is_cold=True) で再構成できる）。
    """

    run_id: str            # 人間可読＋一意（例 "20260605T1430_sao10_mps"）
    created_at: str        # ISO8601（スクリプト付与）
    model: str             # 例 "stable-audio-open-1.0"
    model_revision: str    # HF commit/revision（再現用）
    env: EnvMeta
    grid: GridMeta
    dtype: str             # 実行精度（env固定・§3/§4。例 "float32"）
    sample_rate: int       # モデル native（記録のみ・§9.1）
    channels: int          # モデル native（記録のみ）
    model_load_sec: float | None = None   # ロード時間＝cold の主成分（§4）
    schema_version: str = SCHEMA_VERSION
    notes: str = ""


@dataclass
class MemoryMeta:
    """ピークメモリ（§4.1）。device 別に API の切り口が違う。該当しない側は None。"""

    # Mac (MPS) — torch.mps.* ＋ psutil
    mps_current_allocated_bytes: int | None = None    # テンソル占有（キャッシュ除く）
    mps_driver_allocated_bytes: int | None = None     # ドライバ確保総量（キャッシュ込み＝実フットプリント寄り）
    mps_recommended_max_bytes: int | None = None      # 推奨上限ワーキングセット（予算の目安）
    # Windows (CUDA) — torch.cuda.* ＋ nvidia-smi
    cuda_max_allocated_bytes: int | None = None       # テンソル占有のピーク
    nvidia_smi_used_bytes: int | None = None          # プロセス/GPU の VRAM 使用（OS視点）
    # 両OS共通（プロセス全体・OS視点）
    rss_bytes: int | None = None                      # psutil RSS（ユニファイドメモリで有用）


@dataclass
class MpsFallback:
    """MPS 未対応 op の CPU フォールバック（Mac のみ・実用性のシグナル・§4）。"""

    occurred: bool = False
    ops: list[str] = field(default_factory=list)      # 該当 op 名（多発＝激遅＝実用不可のサイン）


@dataclass
class ClipMeta:
    """1生成（1クリップ）の全条件＋実測。metadata/clip_NNNN.json に保存。

    run 固定の項目（model/env/dtype/sample_rate/channels/steps）は持たず、run_id で run.json を参照する。
    """

    clip_id: str           # 中立連番＝ブラインドJOINキー（§5.4。例 "clip_0001"）
    run_id: str            # run.json への参照
    # 何を生成したか（prompts.yaml 由来）
    prompt_id: str
    tier: str              # "body" / "diagnostic"（カバー率母数の区別・§8.1）
    source: str            # 例 "環境音"（分析用）
    temporal: str          # 例 "環境"（分析用）
    prompt: str
    negative_prompt: str
    # 生成パラメータ（§9.1 スイープ条件・このクリップの具体値）
    cfg: float             # grid.cfg_set の 1 つ
    seed: int              # grid.seed_set の 1 つ
    duration_sec: float    # prompt の temporal 由来（単発3 / 継続6 / 環境10・§9.1）
    # 出力
    audio_file: str        # run ディレクトリからの相対（例 "audio/clip_0001.wav"）
    # 速度（§4）。cold/warm は is_cold で区別（§9.2 の cold/warm 2フィールドを畳んだ形）
    gen_time_sec: float | None = None
    is_cold: bool = False
    rtf: float | None = None                          # gen_time_sec ÷ duration_sec（§13）
    # メモリ・互換
    memory: MemoryMeta = field(default_factory=MemoryMeta)
    mps_cpu_fallback: MpsFallback = field(default_factory=MpsFallback)
    schema_version: str = SCHEMA_VERSION
    notes: str = ""


def to_dict(meta: Any) -> dict[str, Any]:
    """dataclass を JSON 直列化用の dict へ（``json.dump(to_dict(meta), f)`` で使う）。

    asdict が入れ子の dataclass（EnvMeta/GridMeta/MemoryMeta/MpsFallback）も再帰展開する。
    """
    return asdict(meta)
