# backend — 生成アルゴリズム本体（Python + FastAPI）

このアプリの本質的価値はここにある（dev §2.1）。Step 1〜8 のパイプラインを実装する。

- `pipeline/` … Step 1〜8（入力→構造化→プロンプト→生成→評価→選別→提示）
- `engines/` … 生成エンジン抽象化（初期 diffusers 直叩き → 将来 ComfyUI/自前に差替。dev §2.3）
- `eval/` … CLAP / PAM 評価（決定論・LLM 不使用。FF-D005）

開発順序は [prototype-roadmap.md](../../docs/prototype-roadmap.md) を参照。
