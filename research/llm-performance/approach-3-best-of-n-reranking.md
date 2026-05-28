# アプローチ3: Best-of-N + リランキング

> FoleyForge 品質向上戦略の解説ドキュメント（3/4）
> 対象読者: このテーマを初めて学ぶ人

---

## 1. 一言でいうと

**「同じプロンプトで音をたくさん作り、その中から一番良いものを選ぶ」** という手法です。

非常にシンプルですが、実装の効果が高く、FoleyForgeの中核を担う手法です。

- **Best-of-N**: N個（例えば10個）の候補を生成すること
- **リランキング**: 生成された候補を、品質指標で評価して並び替え・選別すること

---

## 2. なぜこれが効くのか

### 2.1 拡散モデルの出力は「運」に左右される

拡散モデルは、同じプロンプトでも、シード（乱数の種）が違えば違う音を生成します。そして、その質にはバラつきがあります。

- 1回生成して、たまたま良い音が出ることもある
- たまたまイマイチな音が出ることもある

つまり、1回の生成は「ガチャ」のようなものです。1回引いて当たりを期待するより、**10回引いて一番良いものを選ぶ**ほうが、良い結果を得やすいのは直感的に分かります。

### 2.2 「推論時スケーリング」という考え方

この手法は、学術的には「Inference-Time Scaling（推論時スケーリング）」と呼ばれます。SCOREの研究では次のように説明されています。

> これに対処するため、我々はInference-Time Scalingを採用する。これは訓練不要の手法で、推論時の計算量を増やすことで性能を向上させる。[引用1]

ポイントは「**訓練不要**」という点です。モデルを訓練し直すのではなく、「たくさん生成する」という計算量を投入することで質を上げます。FoleyForgeのファインチューニングをしない方針と完全に合致します。

---

## 3. リランキングの主役: CLAP

### 3.1 「良い音」をどう機械的に判断するか

N個の候補から「一番良いもの」を選ぶには、機械が「良さ」を判断する必要があります。ここで登場するのが **CLAP（Contrastive Language-Audio Pretraining）** です。

CLAPは、**音声とテキストがどれだけ一致しているかを数値で測る**モデルです。

```
プロンプト: "fast footsteps on wet asphalt"
   +
生成された音声
   ↓
CLAP
   ↓
一致度スコア（例: 0.42）
```

このスコアが高いほど、「プロンプト通りの音が生成できている」と判断できます。N個の候補それぞれにCLAPスコアをつけて、高いものを選ぶ。これがCLAPによるリランキングです。

### 3.2 実際のSOTAシステムでも使われている

CLAPによるリランキングは、研究上の理論だけでなく、実際に高性能なシステムで使われています。Foley音（効果音）生成のコンペティションでも、上位システムがこの手法を採用しています。

---

## 4. 最重要の落とし穴: 単一指標の罠

ここがこのアプローチで**最も注意すべき点**です。FoleyForgeの設計判断に直結します。

### 4.1 CLAPだけで選ぶと何が起きるか

「CLAPスコアが高いものを選べば良い」と思うかもしれません。しかし、CLAPスコアだけを追い求めると、**「指示には合っているが、音として汚い」** ものを選んでしまう危険があります。

これを「verifier hacking（評価器ハッキング）」または「reward hacking（報酬ハッキング）」と呼びます。

SCOREの研究は、この現象を実際のデータで示しています。

> 単一の報酬で推論をガイドすると「verifier hacking」を引き起こし、生成が報酬に過剰適応して、全体的な高品質を達成できなくなる。[引用2]

さらに具体的な証拠として、

> PQ報酬を使ったBest-of-Nは、より高いCLAPスコアを持つにもかかわらず、単純なサンプリングよりも低いAQAScoreを達成した。これはverifier hackingの証拠である。[引用2]

つまり、**ある指標を最大化しようとすると、その指標の「抜け穴」を突いた、見かけ上スコアは高いが実際は質の低いものが選ばれてしまう**のです。

### 4.2 Best-of-Nは増やせば良いわけでもない

もう一つの落とし穴があります。「Nを大きくすればするほど良くなる」わけではない、という点です。

Best-of-Nの理論を分析した研究は、次のように警告しています。

> Best-of-Nサンプリングのような技術で計算量を単純に増やすと、reward hacking（報酬ハッキング）によって性能が劣化することがある。[引用3]

別の研究でも、

> Best-of-Nデコーディングは、Nが大きいとreward hackingに苦しむ。これは、下流タスクの結果を必ずしも改善しない、より好都合な代理報酬を見つけてしまうことにつながる。[引用5]

これは重要です。「100個生成して一番CLAPスコアが高いものを選ぶ」と、CLAPの抜け穴を突いた変な音が選ばれるリスクが、むしろ高まる可能性があるのです。

---

## 5. 解決策: 複数指標を組み合わせる（SCORE）

### 5.1 CLAPとPQの両方で評価する

単一指標の罠を避けるには、**複数の指標を組み合わせる**ことです。SCOREの研究が、この方法を体系化しています。

SCOREは、2つの指標を使います。

| 指標 | 測るもの |
|------|---------|
| CLAP | 音声とテキストの一致度（指示追従性） |
| PQ（Audiobox-Aestheticsより） | 音そのものの品質（音響的美しさ） |

> 我々は、audio-textアライメントの報酬としてCLAPスコアを、一般的な音声品質の報酬としてAudiobox-AestheticsのPQを選択する。[引用2]

### 5.2 指標を「正規化」してから combine する

ここに技術的な工夫があります。CLAPとPQは、そのままでは数値の範囲（平均や散らばり）が違うので、単純に足し算できません。

> しかし、複数の報酬モデルを利用することは重大な課題を生む。各シグナルが固有の分布から来ているからだ。[引用2]

そこでSCOREは、両方の指標を「正規化」（平均0、分散1にそろえる処理）してから組み合わせます。これにより、公平に2つの指標を合算できるようになります。

### 5.3 重みで「好み」を調整できる

さらにSCOREは、CLAPとPQの重みを変えることで、出力の傾向をコントロールできます。

> 生成は、品質報酬（PQ）の重みが高いときは音質を優先し、テキストアライメント報酬（CLAP）の重みが高いときはaudio-textアライメントを優先する。[引用4]

これはまさに、FoleyForgeの「指示追従寄り / バランス / 音響美寄り」という3戦略の理論的な裏付けです。重みを変えることで、戦略ごとに異なる特性の候補を選別できます。

---

## 6. FoleyForgeでの応用

これまでの内容を、FoleyForgeの設計に当てはめます。

```
[プロンプト]
   ↓
N個生成（シード違い）  ← Best-of-N
   ↓
各候補を CLAP + PQ で評価  ← 複数指標で評価（verifier hacking回避）
   ↓
正規化して重み付き合算
   ↓
戦略ごとに選別
  - 指示追従寄り: CLAP重み高
  - バランス: CLAP・PQ均等
  - 音響美寄り: PQ重み高
   ↓
異なる特性の候補をユーザに提示
```

### 6.1 設計上の重要な教訓

このアプローチから、FoleyForgeが守るべき設計原則が導かれます。

1. **単一指標で選ばない** — 必ずCLAPとPQの両方を使う（verifier hacking回避）
2. **Nを無闇に増やさない** — 大きすぎるNは逆効果になりうる。適切な値を実験で見つける
3. **正規化を忘れない** — 異なる指標を組み合わせる前に、スケールをそろえる
4. **重みで戦略を表現** — CLAP/PQの重み配分が、そのまま「創作スタイル」になる

---

## 7. まとめ

| ポイント | 内容 |
|---------|------|
| 何をするか | N個生成して、品質指標で一番良いものを選ぶ |
| なぜ効くか | 生成のバラつきを「数撃って選ぶ」ことで吸収する |
| 訓練の要否 | 不要（推論時スケーリング） |
| 主な指標 | CLAP（指示追従性）、PQ（音響品質） |
| 最大の落とし穴 | 単一指標の最大化はverifier hackingを招く |
| 解決策 | 複数指標を正規化して組み合わせる（SCORE方式） |
| FoleyForgeでの意味 | 3戦略の候補生成は、重み配分の異なるリランキングそのもの |

このアプローチは、FoleyForgeの「多様性を耳で選ぶ」という思想を、技術的に実現する手段です。複数指標を組み合わせることで、verifier hackingを避けつつ、異なる特性の良い候補をユーザに届けられます。

---

## 引用元と参考文献

### 説明中で引用した箇所

**[引用1]** SCORE（プロジェクトページ）より（セクション2.2で引用）
- 原文: "To address this, we adopt Inference-Time Scaling, a training-free method that improves performance by increasing inference computation."

**[引用2]** SCORE（論文本体）より（セクション4.1, 5.1, 5.2で引用）
- 原文（4.1・verifier hacking）: "guiding inference with a single reward leads to 'verifier hacking,' causing the generation to over-adapt to the reward and fail to achieve high overall quality"
- 原文（4.1・証拠）: "BON with PQ reward attains a lower AQAScore than naive sampling despite a higher CLAP score, an evidence of verifier hacking."
- 原文（5.1）: "We select the CLAP score as a reward for audio-text alignment, and PQ from Audiobox-Aesthetics for general audio quality."
- 原文（5.2）: "utilizing multiple reward models poses a critical challenge, because each signal originates from a unique distribution."

**[引用3]** Best-of-N理論研究より（セクション4.2で引用）
- 原文: "naively increasing computation in techniques like Best-of-N sampling can lead to performance degradation due to reward hacking."

**[引用4]** SCORE（プロジェクトページ）より（セクション5.3で引用）
- 原文: "The generation favors audio quality when the quality reward (PQ) weight is high, and favors audio-text alignment when the text alignment reward (CLAP) weight is high."

**[引用5]** Reranker研究より（セクション4.2で引用）
- 原文: "Best-of-N decoding can suffer with large N due to reward hacking, which leads to finding a more favorable proxy reward that doesn't necessarily improve results on the downstream task."

### 参考文献リスト

1. **SCORE: Scaling audio generation using Standardized COmposite REwards（プロジェクトページ）**
   https://mm.kaist.ac.kr/projects/score/

2. **SCORE（論文本体・arXiv）**
   https://arxiv.org/html/2509.19831

3. **Is Best-of-N the Best of Them? Coverage, Scaling, and Optimality in Inference-Time Alignment**
   https://arxiv.org/pdf/2503.21878

4. **Revisiting the (Sub)Optimality of Best-of-N for Inference-Time Alignment**
   https://arxiv.org/pdf/2603.05739

5. **Drowning in Documents: Consequences of Scaling Reranker Inference**
   https://arxiv.org/html/2411.11767v2

6. **Guided by the Plan: Enhancing Faithful Autoregressive Text-to-Audio Generation with Guided Decoding**
   https://arxiv.org/html/2601.14304

### 注記

CLAPモデルの具体的な実装としては、LAION-CLAP（オープンソース）が広く使われています。PQスコアについては、Audiobox-Aesthetics（Meta）が論文で使われていますが、実装の手軽さを考えると、初期段階ではCLAPベースの簡易品質指標（PAMスコアなど）から始めることも検討できます。
