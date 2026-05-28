# Text-to-Audio ローカル推論Webアプリ 開発方針ドキュメント

## 1. このドキュメントの目的

このドキュメントは、Text-to-Audio（以下 TTA）アプリケーションの動作環境・配布方針・アーキテクチャについて、現時点で決定した内容を整理するための開発メモです。

今回の方針では、クラウドサーバー上で音声生成を行うのではなく、ユーザー自身のPC上で音声生成モデルを実行します。

そのうえで、UIはデスクトップアプリとして作るのではなく、ブラウザで操作できるローカルWebアプリとして構成します。

---

## 2. 基本方針

### 2.1 採用する方針

本アプリは、以下の方針で開発します。

```text
ユーザーPC内で完結する
  ↓
ブラウザでUIを表示する
  ↓
ローカルAPI / 推論エンジンが動く
  ↓
models/ から音声生成モデルを読み込む
  ↓
outputs/ に生成音声を保存する
```

つまり、アプリの形式としては次のようになります。

```text
ローカル推論Webアプリ
localhost型アプリ
browser-based local app
self-hosted local web app
```

このドキュメントでは、以降 **ローカル推論Webアプリ** と呼びます。

---

## 3. なぜこの構成にするのか

### 3.1 サーバー費用を抑えたい

TTAのような音声生成モデルは、GPUリソースを必要とする場合があります。

クラウドGPUや専用サーバーを利用すると、以下のようなコストが発生します。

- GPUサーバー利用料
- ストレージ費用
- 通信量
- モデル実行コスト
- 常時稼働コスト

そのため、初期段階ではサーバーを借りず、ユーザーのPC上で音声生成を行う方針とします。

### 3.2 オフラインでも使えるようにしたい

モデルとアプリをユーザーPCに配置することで、インターネット接続がない状態でも利用できる構成を目指します。

ただし、モデルの初回ダウンロード方式を採用する場合は、初回のみインターネット接続が必要になります。

### 3.3 デスクトップUIのOS依存を避けたい

Windows / macOS / Linux それぞれにネイティブUIを作ると、OS依存が発生しやすくなります。

一方、ブラウザUIであれば、UI部分は比較的クロスプラットフォームに作れます。

```text
React / HTML / CSS / JavaScript
  ↓
ブラウザ上で動作
  ↓
OS依存を比較的抑えられる
```

ただし、推論エンジン側は依然としてOSやGPU環境に依存します。

---

## 4. 採用アーキテクチャ

### 4.1 全体構成

```text
[Browser UI]
  React / Vite / Next.js など
        ↓ HTTP / WebSocket
[Local FastAPI Server]
  API / ジョブ管理 / ファイル管理
        ↓
[Python Inference Engine]
  PyTorch / CUDA / CPU / ffmpeg
        ↓
[Local Files]
  models/
  outputs/
  configs/
  logs/
```

### 4.2 役割分担

| 領域 | 役割 |
|---|---|
| frontend | ブラウザUI、入力フォーム、生成履歴、音声再生 |
| backend | API、ジョブ管理、ファイル管理、モデル管理 |
| inference | 音声生成モデルの読み込み、推論、音声ファイル出力 |
| models | 音声生成モデルの保存場所 |
| outputs | 生成された音声ファイルの保存場所 |
| configs | モデル定義やアプリ設定 |
| logs | エラーや実行ログ |

---

## 5. 推奨ディレクトリ構成

```text
tta-local-web/
  ├─ frontend/
  │   ├─ package.json
  │   ├─ src/
  │   └─ vite.config.ts
  │
  ├─ backend/
  │   ├─ app/
  │   │   ├─ main.py
  │   │   ├─ api/
  │   │   ├─ services/
  │   │   └─ model_manager.py
  │   ├─ inference/
  │   │   └─ tta_engine.py
  │   └─ requirements.txt
  │
  ├─ models/
  │   └─ stable-audio-open-small/
  │
  ├─ outputs/
  │   └─ generated/
  │
  ├─ configs/
  │   └─ models.yaml
  │
  ├─ logs/
  │
  ├─ scripts/
  │   ├─ start_windows.bat
  │   ├─ start_mac.sh
  │   └─ start_linux.sh
  │
  ├─ docker-compose.yml
  ├─ .gitignore
  └─ README.md
```

---

## 6. 各コンポーネントの説明

### 6.1 frontend

ブラウザで表示するUIです。

主な機能は以下です。

- プロンプト入力
- 生成ボタン
- 生成中表示
- モデル選択
- 生成履歴
- 音声再生
- 音声ファイルの保存
- 設定画面

候補技術は以下です。

| 候補 | 備考 |
|---|---|
| React | 既存スキルを活かしやすい |
| Vite | ローカルWebアプリの開発に向いている |
| Next.js | 将来的にクラウド版へ移行する場合に有力 |
| Tailwind CSS | UIを素早く構築しやすい |

初期開発では、React + Vite を想定します。

### 6.2 backend

ローカルPC上で動作するAPIサーバーです。

主な役割は以下です。

- frontend からの生成リクエストを受け取る
- 推論エンジンを呼び出す
- 生成ジョブを管理する
- 生成結果を outputs/ に保存する
- 生成音声を frontend に返す
- models/ の状態を管理する

候補技術は以下です。

| 候補 | 備考 |
|---|---|
| FastAPI | Python推論処理と相性が良い |
| Flask | 軽量だが、API設計ではFastAPIが扱いやすい |
| Node.js | UI側との統一感はあるが、PyTorch推論とは分離が必要 |

初期開発では、FastAPI を想定します。

### 6.3 inference

音声生成モデルを実行する部分です。

主な役割は以下です。

- モデルの読み込み
- プロンプトの受け取り
- 音声生成
- wav / mp3 などの出力
- GPU / CPU の利用判定
- エラー処理

候補技術は以下です。

| 項目 | 候補 |
|---|---|
| 推論基盤 | PyTorch |
| GPU | CUDA / NVIDIA GPU |
| CPU実行 | 可能だが遅い可能性あり |
| 音声変換 | ffmpeg |

---

## 7. モデル管理方針

### 7.1 モデルはローカルに保存する

ユーザーPCで音声生成を行うため、音声生成モデルはユーザーのPC内に保存します。

```text
models/
  stable-audio-open-small/
  audioldm2/
  musicgen-small/
```

### 7.2 モデル本体はGit管理しない

音声生成モデルはサイズが大きいため、Git管理しません。

`.gitignore` には以下を含めます。

```gitignore
models/
outputs/
logs/
.cache/
```

### 7.3 models.yaml でモデル情報を管理する

モデル名、保存先、状態、必要VRAMなどは、設定ファイルで管理します。

例:

```yaml
models:
  stable-audio-small:
    display_name: "Stable Audio Small"
    type: "text-to-audio"
    local_path: "models/stable-audio-open-small"
    installed: true
    version: "1.0"
    required_vram_gb: 8
    max_duration_sec: 30

  audioldm2:
    display_name: "AudioLDM 2"
    type: "text-to-audio"
    local_path: "models/audioldm2"
    installed: false
    version: null
    required_vram_gb: 10
    max_duration_sec: 10
```

### 7.4 モデル配布方式

モデル配布には以下の選択肢があります。

| 方式 | 内容 | 一般ユーザー向け |
|---|---|---|
| モデル同梱型 | アプリにモデルを含める | 使いやすいが容量が大きい |
| 初回ダウンロード型 | 初回起動時にモデルを取得 | バランスが良い |
| 手動配置型 | ユーザーがmodels/に配置 | 開発者向け |

初期開発では手動配置でもよいですが、一般ユーザー向けには **初回ダウンロード型** が望ましいです。

---

## 8. Dockerの位置づけ

### 8.1 Dockerは開発・検証・技術者向け配布に有効

Dockerを使うと、Python / PyTorch / ffmpeg などの実行環境をまとめて管理できます。

```text
Dockerで管理するもの:
  - Python
  - PyTorch
  - FastAPI
  - ffmpeg
  - 推論コード
  - 依存ライブラリ

Dockerで管理しないもの:
  - モデル本体
  - 生成された音声
  - 個人用メモ
  - 大きなキャッシュ
```

### 8.2 一般ユーザーにDockerを直接触らせるのは避ける

Dockerは技術者向けには便利ですが、一般ユーザーには以下の負担があります。

- Docker Desktop のインストール
- コマンド操作
- GPU設定
- models/ の配置
- localhost へのアクセス
- エラー時のログ確認

そのため、一般ユーザー向けには、Dockerを直接使わせるのではなく、起動スクリプトやランチャーで隠蔽する方針が望ましいです。

### 8.3 開発環境としてのDocker Compose

開発者向けには、以下のようなDocker Compose構成が有効です。

```text
docker-compose.yml
  - frontend
  - backend
  - redis
  - worker
```

初期段階では frontend / backend だけで十分です。

---

## 9. 起動方式

### 9.1 開発者向け

開発者向けには、以下のような手動起動を想定します。

```bash
# backend
cd backend
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

# frontend
cd frontend
npm run dev
```

### 9.2 一般ユーザー向け

一般ユーザー向けには、起動スクリプトまたは薄いランチャーを用意します。

```text
ユーザーが起動
  ↓
backend を起動
  ↓
frontend を起動、または静的ファイルを配信
  ↓
ブラウザを自動で開く
```

例:

```text
scripts/
  start_windows.bat
  start_mac.sh
  start_linux.sh
```

将来的には、以下のような薄いランチャーも検討できます。

```text
Launcher
  - ポート確認
  - backend起動
  - ブラウザ起動
  - 終了時にプロセス停止
```

---

## 10. API設計の初期案

### 10.1 最小API

| Method | Path | 役割 |
|---|---|---|
| GET | /health | backendの起動確認 |
| GET | /models | 利用可能モデル一覧 |
| POST | /generate | 音声生成リクエスト |
| GET | /jobs/{job_id} | 生成ジョブの状態確認 |
| GET | /outputs/{filename} | 生成音声ファイル取得 |

### 10.2 generate API のリクエスト例

```json
{
  "model": "stable-audio-small",
  "prompt": "soft rain in a quiet forest",
  "duration_sec": 10,
  "format": "wav"
}
```

### 10.3 generate API のレスポンス例

```json
{
  "job_id": "job_20260528_001",
  "status": "queued"
}
```

生成完了後は、以下のようなレスポンスを想定します。

```json
{
  "job_id": "job_20260528_001",
  "status": "completed",
  "output_file": "outputs/generated/job_20260528_001.wav"
}
```

---

## 11. 非同期処理の考え方

音声生成は時間がかかるため、最終的には非同期ジョブ方式が望ましいです。

```text
ユーザーが生成ボタンを押す
  ↓
backend が job を作成
  ↓
推論処理を実行
  ↓
frontend は進捗を確認
  ↓
完了後に音声を再生
```

初期段階では同期処理でもよいですが、生成時間が長くなる場合は非同期化します。

候補は以下です。

| 方式 | 備考 |
|---|---|
| FastAPI BackgroundTasks | 軽量な初期実装向け |
| Celery + Redis | 本格的なジョブ管理向け |
| RQ + Redis | 比較的シンプル |
| 独自キュー | 小規模なら可能 |

---

## 12. 最小実装のゴール

最初のゴールは、以下を満たすことです。

```text
ブラウザでテキストを入力
  ↓
生成ボタンを押す
  ↓
ローカルPC内のモデルで音声生成
  ↓
outputs/ に保存
  ↓
ブラウザ上で音声再生
```

最初から複数モデル対応や高度なモデル管理を入れず、まずは1モデルで動作確認します。

---

## 13. 開発ロードマップ

### Phase 1: モデル単体の動作確認

目的:

- ローカルPCでTTAモデルが動くか確認する
- GPU / CPU の動作を確認する
- 生成時間と必要メモリを把握する

成果物:

- 推論スクリプト
- 入力プロンプトから音声生成できる状態
- outputs/ に音声ファイルを保存できる状態

### Phase 2: FastAPI化

目的:

- 推論処理をAPIから呼び出せるようにする

成果物:

- POST /generate
- GET /outputs/{filename}
- GET /health

### Phase 3: React UI作成

目的:

- ブラウザから生成操作できるようにする

成果物:

- プロンプト入力画面
- 生成ボタン
- 生成中表示
- 音声再生UI

### Phase 4: モデル管理の導入

目的:

- models.yaml によりモデル情報を管理する

成果物:

- GET /models
- モデル選択UI
- インストール済みモデルの表示

### Phase 5: 起動スクリプト作成

目的:

- ユーザーが簡単に起動できるようにする

成果物:

- start_windows.bat
- start_mac.sh
- start_linux.sh

### Phase 6: 配布形態の整備

目的:

- 技術者向け・一般ユーザー向けの配布方法を整理する

成果物:

- README
- セットアップ手順
- 必要環境
- モデル配置手順
- トラブルシューティング

---

## 14. 想定される課題

### 14.1 OS依存は完全には消えない

UIはブラウザでクロスプラットフォーム化できますが、推論エンジン側にはOS依存が残ります。

例:

- CUDA / NVIDIA GPU
- macOS の MPS
- Linux のGPUドライバ
- ffmpeg
- ファイルパス
- 起動スクリプト

### 14.2 ユーザーPCの性能に依存する

ローカル推論型のため、生成速度や実行可否はユーザーのPC性能に依存します。

確認すべき項目:

- GPUの有無
- VRAM容量
- CPU性能
- メモリ容量
- 空きストレージ
- ffmpegの有無

### 14.3 モデルライセンスの確認が必要

モデルを配布・同梱・自動ダウンロードする場合は、ライセンス確認が必要です。

確認項目:

- 商用利用可能か
- 再配布可能か
- モデル同梱可能か
- 生成物の利用制限
- クレジット表記の要否

### 14.4 初回セットアップが重い可能性

モデルサイズが大きい場合、初回セットアップには時間がかかります。

必要なUX:

- ダウンロード進捗表示
- 空き容量チェック
- 失敗時のリトライ
- 保存先変更
- モデル削除

---

## 15. 現時点の決定事項まとめ

| 項目 | 決定内容 |
|---|---|
| アプリ形式 | ローカル推論Webアプリ |
| UI | ブラウザUI |
| フロントエンド | React / Vite を第一候補 |
| バックエンド | FastAPI を第一候補 |
| 推論 | Python / PyTorch |
| モデル保存 | ユーザーPCの models/ |
| 生成音声保存 | ユーザーPCの outputs/ |
| サーバー利用 | 基本的に利用しない |
| オフライン利用 | 目指す |
| Docker | 開発・技術者向け配布では有効 |
| 一般ユーザー向けDocker | 直接触らせない方針 |
| 初期実装 | 1モデルで最小構成から開始 |
| 将来拡張 | モデル管理・非同期ジョブ・ランチャー |

---

## 16. 現時点の推奨構成

```text
React / Vite
  ↓
Local FastAPI
  ↓
Python / PyTorch TTA model
  ↓
models/ & outputs/
```

最初の到達目標:

```text
ブラウザでプロンプト入力
  ↓
生成ボタン
  ↓
ローカルPCで音声生成
  ↓
ブラウザで再生
```

---

## 17. 今後検討すること

今後、以下を追加で検討します。

- 採用するTTAモデル
- モデルのライセンス
- モデルサイズと必要VRAM
- Windows / macOS / Linux のサポート範囲
- CPU実行を許容するか
- 初回モデルダウンロード機能
- 生成履歴管理
- ジョブキュー方式
- Docker Compose構成
- 起動スクリプト
- ランチャーの要否
- WebGPU / ONNX Runtime Web によるブラウザ内推論の可能性

---

## 18. 補足: ブラウザ完全完結型について

将来的な理想形として、ブラウザだけで推論まで完結する方式も考えられます。

```text
Browser
  ├─ React UI
  ├─ WebGPU / WASM
  ├─ モデル
  └─ 音声生成
```

ただし、TTAの拡散モデルはサイズや計算量が大きく、現時点では初期実装としては難易度が高いです。

そのため、まずは以下の構成を優先します。

```text
ブラウザUI + ローカルFastAPI + PyTorch推論
```

---

## 19. まとめ

本アプリは、クラウドサーバー上で音声生成を行うのではなく、ユーザーPC内でモデルを実行するローカル推論型アプリとして開発します。

ただし、UIはデスクトップUIではなくブラウザUIとし、Web技術を活用してクロスプラットフォーム性を高めます。

最終的な基本構成は以下です。

```text
Browser UI
  ↓
Local API Server
  ↓
Inference Engine
  ↓
Local Model Store
  ↓
Local Output Store
```

この構成により、以下を両立することを目指します。

- サーバー費用を抑える
- オフライン利用に対応する
- Web技術でUIを作る
- ユーザーPC上で音声生成を行う
- 将来的にクラウド版やDocker配布にも展開しやすくする
