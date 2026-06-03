# models — モデルの配置先（リポジトリには同梱しない）

T2A・評価モデルの本体はリポジトリに含めない（[FF-D004](../../docs/decisions.md)）。
ユーザが各自ダウンロードしてここに配置する。中身は `.gitignore` 済み（この README を除く）。
コードからは `models.yaml` のパス設定経由で参照する（dev §3.2／Step5）。
