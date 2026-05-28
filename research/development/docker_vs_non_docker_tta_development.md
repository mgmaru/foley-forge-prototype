# Docker環境と非Docker環境の比較ドキュメント

## Text-to-Audioローカル推論Webアプリにおける開発・デバッグ・配布方針

## 1. このドキュメントの目的

このドキュメントは、Text-to-Audio（TTA）ローカル推論Webアプリにおいて、**Docker環境で開発・実行する場合** と **Dockerを使わずに開発・実行する場合** の違いを整理するための開発メモです。

今回のアプリケーションでは、単なるバックエンドの処理速度改善ではなく、**拡散モデルが出力する生成物の品質に関するパラメータ最適化** が重要になります。

そのため、以下の観点を重視して比較します。

- 生成品質のパラメータ調整のしやすさ
- バックエンド各層のデバッグのしやすさ
- モデル出力結果の確認のしやすさ
- 開発環境の再現性
- 将来的な配布のしやすさ
- 一般ユーザー向け配布との相性

---

## 2. 前提となるアプリケーション構成

今回想定しているアプリケーションは、クラウドサーバー上ではなく、ユーザーPC内で動作する **ローカル推論Webアプリ** です。

```text
Browser UI
  ↓
Local FastAPI
  ↓
Python / PyTorch Inference Engine
  ↓
Diffusion-based Text-to-Audio Model
  ↓
Local outputs/
```

想定構成は以下です。

```text
tta-local-web/
  ├─ frontend/
  │   └─ React / Vite
  │
  ├─ backend/
  │   └─ FastAPI
  │
  ├─ inference/
  │   └─ 音声生成モデル実行処理
  │
  ├─ experiments/
  │   └─ 品質調整・パラメータ探索用スクリプト
  │
  ├─ models/
  │   └─ 音声生成モデル
  │
  ├─ outputs/
  │   └─ 生成音声
  │
  ├─ configs/
  │   └─ モデル設定・プリセット
  │
  └─ logs/
```

---

## 3. 今回の「最適化」の意味

一般的なバックエンド最適化では、以下のようなものが対象になります。

```text
APIレスポンス速度
DBクエリ速度
メモリ使用量
スループット
同時接続数
```

しかし、今回重視する最適化はこれとは異なります。

今回の最適化対象は、**拡散モデルの生成品質** です。

具体的には、以下のようなパラメータを調整して、出力音声の品質を比較します。

```text
prompt
negative prompt
seed
steps
guidance scale
scheduler
duration
sampler
temperature
top-k / top-p
cfg scale
model variant
post-processing
normalization
fade in / fade out
sample rate
bit depth
```

そのため、今回の開発では以下が重要になります。

```text
パラメータをすぐ変更できる
生成結果をすぐ聴ける
生成条件と出力を紐づけて保存できる
バックエンド各層を細かくデバッグできる
失敗した生成結果も分析できる
良い設定をプリセット化できる
```

---

## 4. Docker環境と非Docker環境の大まかな違い

### 4.1 Docker環境

Docker環境では、Python、PyTorch、FastAPI、ffmpegなどの実行環境をコンテナ内にまとめます。

```text
Docker Container
  ├─ Python
  ├─ PyTorch
  ├─ FastAPI
  ├─ ffmpeg
  ├─ 推論コード
  └─ 依存ライブラリ

Host PC
  ├─ models/
  ├─ outputs/
  ├─ configs/
  └─ logs/
```

モデルや生成物は、通常Dockerイメージに含めず、ホスト側のディレクトリをマウントします。

```text
Host ./models   →  Container /app/models
Host ./outputs  →  Container /app/outputs
Host ./configs  →  Container /app/configs
Host ./logs     →  Container /app/logs
```

### 4.2 非Docker環境

非Docker環境では、開発者のPC上に直接Python環境を作ります。

```text
Host PC
  ├─ .venv / conda
  ├─ Python
  ├─ PyTorch
  ├─ FastAPI
  ├─ ffmpeg
  ├─ models/
  ├─ outputs/
  └─ configs/
```

起動例は以下です。

```bash
# backend
cd backend
uvicorn app.main:app --reload

# experiment
python experiments/run_single.py --preset stable_audio_default
```

---

## 5. 全体比較表

| 観点 | Dockerあり | Dockerなし |
|---|---|---|
| 環境再現性 | 高い | 低〜中 |
| 初期構築の簡単さ | やや難しい | 比較的簡単 |
| 品質パラメータ調整 | やや手間が増える | やりやすい |
| ブレークポイントデバッグ | 設定が必要 | やりやすい |
| 生成音声の確認 | volume設定次第 | 直接確認しやすい |
| GPU設定 | コンテナ設定が必要 | ホストから直接使いやすい |
| 他PCへの移行 | しやすい | 環境差分が出やすい |
| 技術者向け配布 | 向いている | やや不向き |
| 一般ユーザー向け配布 | 直接は不向き | 直接配布も難しい |
| 実験・研究開発 | やや重い | 向いている |
| 本番相当の再現確認 | 向いている | やや弱い |
| 長期運用 | 向いている | 環境が壊れやすい |

---

## 6. 開発フェーズ別の向き・不向き

| フェーズ | Dockerあり | Dockerなし | 推奨 |
|---|---:|---:|---|
| モデル単体の動作確認 | △ | ◎ | Dockerなし |
| パラメータ探索 | ○ | ◎ | Dockerなし |
| バックエンド層のデバッグ | ○ | ◎ | Dockerなし |
| FastAPI統合 | ○ | ◎ | Dockerなし〜併用 |
| React UI連携 | ○ | ○ | どちらでも可 |
| 再現性確認 | ◎ | △ | Dockerあり |
| 技術者向け共有 | ◎ | △ | Dockerあり |
| 一般ユーザー配布準備 | ○ | ○ | 起動スクリプト・ランチャー検討 |
| 本番相当の検証 | ◎ | △ | Dockerあり |

---

## 7. Dockerがデバッグしづらく感じる理由

### 7.1 コード変更の反映が一手間になる

Docker環境では、設定によってはコード変更後に再ビルドが必要になります。

```text
コード変更
  ↓
docker build
  ↓
docker compose up
  ↓
ログ確認
```

volume mountを使えばコード変更を即時反映できます。

```yaml
volumes:
  - ./backend:/app/backend
  - ./inference:/app/inference
```

ただし、生成品質の調整では、細かいパラメータ変更を頻繁に行うため、試行錯誤の初期段階ではDockerがやや重く感じられる可能性があります。

---

### 7.2 ファイルパスが分かりにくくなる

Dockerでは、ホスト側とコンテナ側でファイルパスが異なります。

```text
ホスト側:
./outputs/generated/sample.wav

コンテナ側:
/app/outputs/generated/sample.wav
```

生成音声の品質比較では、出力ファイルを頻繁に確認します。

その際に、以下のような混乱が起きる可能性があります。

```text
音声ファイルはどこに保存されたのか
ホスト側から見えているのか
volume mountされているのか
コンテナ内にだけ保存されていないか
```

---

### 7.3 デバッガ接続の準備が必要

Dockerなしであれば、VS Codeなどから直接Pythonをデバッグできます。

```text
ブレークポイントを置く
  ↓
FastAPIをdebug起動
  ↓
変数や中間データを見る
```

一方、Docker内でデバッグする場合は、以下の準備が必要になることがあります。

```text
debugpyを入れる
デバッグ用ポートを開ける
VS Codeからattachする
ホストパスとコンテナパスを対応させる
```

設定すれば可能ですが、生成品質の探索段階ではやや手間になります。

---

### 7.4 GPUまわりの確認ポイントが増える

Docker環境でGPUを使う場合、ホストPCだけでなくコンテナ側でもGPUが見えている必要があります。

よくある確認ポイントは以下です。

```text
ホストではGPUが見えるか
コンテナ内でnvidia-smiが使えるか
PyTorchがCUDAを認識するか
torch.cuda.is_available() が true になるか
CUDA版PyTorchが入っているか
```

問題例:

```text
ホストではGPUが見えているが、コンテナ内では見えない
コンテナ内のnvidia-smiは動くが、PyTorchがCUDAを認識しない
CUDA / PyTorch / Driver の組み合わせが合っていない
```

品質調整以前に環境確認で時間を取られる可能性があります。

---

## 8. Dockerのメリット

Dockerにはデバッグ上の手間はありますが、大きなメリットもあります。

### 8.1 環境を再現しやすい

Dockerfileに依存環境を書いておけば、同じ環境を再現しやすくなります。

```text
Python version
PyTorch version
CUDA関連
ffmpeg
OSパッケージ
依存ライブラリ
```

非Docker環境では、後から以下のようになりがちです。

```text
どのPythonバージョンで動いていたのか分からない
どのtorchを入れたのか分からない
pip installしたライブラリを記録し忘れた
ffmpegをどう入れたか忘れた
```

---

### 8.2 他人・別PC・GPUクラウドで動かしやすい

Docker化しておくと、開発環境を別PCやGPUクラウドへ移しやすくなります。

```text
ローカルPC
  ↓
Docker Image
  ↓
別PC / GPUクラウド / VPS
  ↓
同じ環境で起動
```

これは、技術者向け配布や検証環境の共有に向いています。

---

### 8.3 依存関係を分離できる

TTAモデルや拡散モデルは、依存関係が重く、壊れやすいです。

```text
Model A は torch 2.1 が安定
Model B は torch 2.4 が必要
Model C は特殊なライブラリが必要
```

Dockerを使うと、モデルや用途ごとに環境を分けられます。

```text
stable-audio-worker
musicgen-worker
audioldm-worker
```

---

### 8.4 配布・再現フェーズに強い

品質調整が終わった後、良い設定や依存環境を固定したい場合、Dockerは有効です。

```text
調整済みパラメータ
  ↓
プリセット化
  ↓
依存環境を固定
  ↓
Docker化
  ↓
再現性のある配布物にする
```

---

## 9. 非Docker環境のメリット

### 9.1 生成品質の調整がしやすい

Dockerなしの場合、Pythonスクリプトを直接実行できます。

```bash
python experiments/run_single.py --preset stable_audio_default
```

また、パラメータを変更してすぐ再実行できます。

```text
設定を変える
  ↓
実行する
  ↓
音声を聴く
  ↓
metadataを見る
  ↓
また調整する
```

この流れが非常に軽くなります。

---

### 9.2 デバッグしやすい

VS Codeなどでそのままブレークポイントを置きやすいです。

```text
API Layer
Application Service
Inference Service
Model Engine
Post Process
Output Store
```

各層を直接追いやすく、変数や中間データも見やすいです。

---

### 9.3 出力ファイルを直接確認しやすい

非Docker環境では、生成音声がそのままローカルの `outputs/` に保存されます。

```text
outputs/experiments/
  ├─ 001.wav
  ├─ 001.metadata.json
  ├─ 002.wav
  └─ 002.metadata.json
```

音声をすぐ再生し、メタデータと照合できます。

---

### 9.4 GPU確認が単純

Dockerを挟まないため、Pythonから直接GPU認識を確認できます。

```python
import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

ホスト側で動けば、そのまま推論コードでも使える可能性が高いです。

---

## 10. 非Docker環境のデメリット

### 10.1 環境が壊れやすい

直接 `pip install` や `conda install` を繰り返すと、依存関係が崩れる可能性があります。

```text
昨日まで動いていたのに動かない
torchを更新したら別モデルが動かない
CUDA版ではなくCPU版のtorchが入っていた
```

---

### 10.2 他PCへの移行が大変

自分のPCでは動いても、別PCでは同じように動かないことがあります。

```text
Pythonバージョンが違う
CUDAバージョンが違う
ffmpegが入っていない
OSパッケージが足りない
```

---

### 10.3 長期的な再現性が弱い

研究・実験フェーズでは便利ですが、数か月後に同じ環境を再現するのは難しくなりがちです。

そのため、動いた環境は必ず記録します。

```text
requirements.txt
environment.yml
README
実行ログ
モデルバージョン
プリセットファイル
```

---

## 11. 推奨する使い分け

結論として、Dockerと非Dockerはどちらか一方に固定するのではなく、開発フェーズによって使い分けるのが望ましいです。

```text
品質調整・研究開発
  → Dockerなし

アプリ統合
  → Dockerなし or Docker併用

再現性確認
  → Dockerあり

技術者向け配布
  → Dockerあり

一般ユーザー向け
  → Dockerを直接触らせない
```

図にすると以下です。

```text
[Phase 1] モデル単体検証
    Dockerなし
        ↓
[Phase 2] パラメータ探索
    Dockerなし
        ↓
[Phase 3] FastAPI統合
    Dockerなし中心
        ↓
[Phase 4] React UI連携
    Dockerなし / Docker併用
        ↓
[Phase 5] 再現性確認
    Dockerあり
        ↓
[Phase 6] 技術者向け配布
    Docker Compose
        ↓
[Phase 7] 一般ユーザー向け配布
    Dockerを隠蔽
```

---

## 12. 推奨ディレクトリ構成

Dockerあり・なしの両方に対応しやすい構成にします。

```text
tta-local-web/
  ├─ frontend/
  │   └─ React / Vite
  │
  ├─ backend/
  │   ├─ app/
  │   │   ├─ main.py
  │   │   ├─ api/
  │   │   ├─ services/
  │   │   └─ model_manager.py
  │   │
  │   ├─ requirements.txt
  │   └─ Dockerfile
  │
  ├─ inference/
  │   ├─ engines/
  │   │   └─ stable_audio_engine.py
  │   ├─ schemas/
  │   │   └─ generation_params.py
  │   └─ postprocess/
  │       └─ audio_normalize.py
  │
  ├─ experiments/
  │   ├─ run_single.py
  │   ├─ run_batch.py
  │   ├─ compare_params.py
  │   └─ presets/
  │       ├─ stable_audio_default.yaml
  │       └─ stable_audio_high_quality.yaml
  │
  ├─ models/
  │   └─ stable-audio-open-small/
  │
  ├─ outputs/
  │   ├─ experiments/
  │   └─ generated/
  │
  ├─ configs/
  │   └─ models.yaml
  │
  ├─ logs/
  │
  ├─ docker-compose.yml
  ├─ .gitignore
  └─ README.md
```

---

## 13. 品質調整用のレイヤー設計

品質調整では、APIを通さずに推論処理だけを直接呼べる構造が重要です。

```text
[Frontend]
  ↓
[API Layer]
  FastAPI endpoint
  ↓
[Application Service]
  ジョブ管理・バリデーション
  ↓
[Inference Service]
  モデル呼び出し
  ↓
[Model Engine]
  PyTorch / Diffusion model
  ↓
[Post Process]
  音量調整 / フェード / フォーマット変換
  ↓
[Output Store]
  wav / mp3 / metadata
```

デバッグ対象は主に以下です。

| レイヤー | デバッグ内容 |
|---|---|
| API Layer | リクエスト形式、バリデーション |
| Application Service | ジョブ管理、プリセット適用 |
| Inference Service | モデル呼び出し、パラメータ変換 |
| Model Engine | steps, cfg, seed, schedulerなど |
| Post Process | 音量、ノイズ、フェード、形式変換 |
| Output Store | 保存先、メタデータ、ファイル名 |

---

## 14. experiments/ を用意する理由

品質調整では、API経由だけでなく、推論スクリプトを直接実行できることが重要です。

```text
experiments/
  ├─ run_single.py
  ├─ run_batch.py
  ├─ compare_params.py
  └─ presets/
```

### 14.1 単発生成

```bash
python experiments/run_single.py --preset stable_audio_default
```

### 14.2 バッチ比較

```bash
python experiments/run_batch.py --config experiments/presets/quality_search.yaml
```

### 14.3 パラメータ比較

```text
steps: 30, 50, 70
guidance_scale: 5.0, 7.5, 10.0
seed: 123, 456, 789
scheduler: default, dpm_solver
```

このように複数条件を一括で試し、出力結果を比較できます。

---

## 15. 生成結果とmetadataの管理

品質調整では、音声ファイルだけでなく、生成条件も必ず保存します。

推奨形式:

```text
outputs/experiments/2026-05-28_001/
  ├─ audio.wav
  ├─ metadata.json
  └─ prompt.txt
```

metadata例:

```json
{
  "model": "stable-audio-small",
  "model_version": "1.0",
  "prompt": "soft rain in a quiet forest",
  "negative_prompt": "noise, distortion, clipping",
  "seed": 12345,
  "duration_sec": 10,
  "steps": 50,
  "guidance_scale": 7.5,
  "scheduler": "dpm_solver",
  "sample_rate": 44100,
  "created_at": "2026-05-28T00:00:00+09:00",
  "output_file": "audio.wav",
  "notes": "雨音は自然だが、後半にノイズあり"
}
```

これにより、あとから以下を確認できます。

```text
どの音声が良かったか
どのパラメータが良かったか
どのseedで再現できるか
どのモデルバージョンで生成したか
```

---

## 16. Dockerを使う場合の推奨方針

Dockerを使う場合は、**開発用Docker** と **配布用Docker** を分けます。

### 16.1 開発用Docker

開発用Dockerでは、デバッグしやすさを優先します。

特徴:

```text
ソースコードをvolume mountする
outputs/ もvolume mountする
reloadを有効にする
debugpyなどを利用できるようにする
ログを標準出力に出す
コンテナ内に入れるようにする
```

例:

```yaml
services:
  backend:
    build: ./backend
    volumes:
      - ./backend:/app/backend
      - ./inference:/app/inference
      - ./models:/app/models
      - ./outputs:/app/outputs
      - ./configs:/app/configs
      - ./logs:/app/logs
    ports:
      - "8000:8000"
      - "5678:5678"
    command: uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 16.2 配布用Docker

配布用Dockerでは、再現性と安定性を優先します。

特徴:

```text
ソースコードをimageに含める
不要なdebug機能は入れない
依存バージョンを固定する
起動方法を簡単にする
モデル本体は基本的にvolume mountする
ログ出力を整理する
```

配布用では、環境を固定することが目的になります。

---

## 17. 非Docker環境での推奨方針

非Docker環境では、venvまたはcondaを使い、Python環境を分離します。

### 17.1 venv構成例

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
```

Windowsの場合:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
```

### 17.2 起動例

```bash
uvicorn backend.app.main:app --reload
```

### 17.3 実験スクリプト例

```bash
python experiments/run_single.py --preset stable_audio_default
python experiments/run_batch.py --config experiments/presets/quality_search.yaml
```

---

## 18. .gitignoreの考え方

モデル本体や生成結果はGit管理しません。

```gitignore
# Python
.venv/
__pycache__/
*.pyc

# Node
node_modules/
dist/

# Model files
models/

# Generated outputs
outputs/

# Logs and cache
logs/
.cache/

# Environment variables
.env
```

ただし、以下はGit管理します。

```text
configs/models.yaml
experiments/presets/*.yaml
requirements.txt
docker-compose.yml
Dockerfile
README.md
```

---

## 19. 推奨ワークフロー

### 19.1 開発初期

```text
Dockerなし
  ↓
モデル単体を動かす
  ↓
experiments/ で品質調整
  ↓
metadataを残す
```

### 19.2 アプリ統合

```text
Dockerなし中心
  ↓
FastAPIから推論を呼ぶ
  ↓
React UIから生成する
  ↓
outputs/に保存する
```

### 19.3 再現性確認

```text
Dockerあり
  ↓
Docker Composeで起動
  ↓
同じパラメータで同じ結果が得られるか確認
```

### 19.4 配布準備

```text
技術者向け:
  Docker Compose

一般ユーザー向け:
  起動スクリプト / ランチャー
  Dockerを直接触らせない
```

---

## 20. 判断まとめ

### 20.1 Dockerなしを優先すべき場面

```text
モデル単体を試す
生成品質を調整する
パラメータを頻繁に変える
ブレークポイントで細かく見る
出力音声をすぐ確認する
実験スクリプトを頻繁に回す
```

### 20.2 Dockerありを優先すべき場面

```text
環境を固定したい
別PCで再現したい
技術者向けに配布したい
依存関係の衝突を避けたい
本番相当の環境で検証したい
Docker Composeで複数サービスを管理したい
```

---

## 21. 最終結論

今回のアプリケーションでは、Dockerを最初から全面採用するよりも、開発フェーズに応じて使い分ける方針が適しています。

特に、拡散モデルの生成品質を調整するフェーズでは、Dockerなしの方がデバッグしやすく、試行錯誤も速くなります。

一方で、品質調整がある程度固まり、依存関係や実行環境を固定したい段階では、Dockerを導入する価値があります。

最終方針は以下です。

```text
品質調整・研究開発:
  Dockerなし

アプリ統合:
  Dockerなし中心、必要に応じてDocker併用

再現性確認:
  Dockerあり

技術者向け配布:
  Docker Compose

一般ユーザー向け:
  Dockerを直接触らせず、起動スクリプトまたはランチャーで隠蔽
```

この方針により、以下を両立します。

```text
開発初期のデバッグしやすさ
生成品質の調整しやすさ
将来的な再現性
技術者向け配布のしやすさ
一般ユーザー向けの扱いやすさ
```

---

## 22. 一言でまとめる

```text
Dockerは「最初から閉じ込めるため」ではなく、
品質調整が固まったあとに「動く環境を固定・再現・配布するため」に使う。
```
