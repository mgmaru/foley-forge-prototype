# phases — phase 別の探索ログ（ラボノート）

[prototype-roadmap.md](../prototype-roadmap.md) の各 Phase で**調べたこと・分かったこと・実測値・没案・試行ログ**を、phase ごとに雑多に置く場所。形式は自由（Markdown・メモ・画像・データ断片など）。

## これは「生ログ」であって「真実」ではない

ここは **捕獲（capture）** の場。確定したら、正しい置き場へ **昇格（promote）** させる:

| 生ノートが… | 昇格先 |
|---|---|
| 決定になった | [decisions.md](../decisions.md)（FF-Dxxx） |
| 設計を変える | 正典docs（[foley-forge-dev.md](../foley-forge-dev.md) / [observation-and-evaluation-design.md](../observation-and-evaluation-design.md) 等） |
| 再利用価値ある知見になった | [research/](../../research/)（topic 別の整理済み解説） |

→ dev-workflow.md §1「情報の3レイヤー」の**手前にある“探索ログ”層**。3レイヤー／research へ流し込む供給源。

## ディレクトリ

各 `phaseN/` に、その phase の roadmap の「問い」への答え・証拠を貯める（例：Phase 0 ＝ モデル実現性の実測 time/VRAM）。
Phase 5–6（プロトタイプ後）のディレクトリは、到達時に `mkdir` で足す。
