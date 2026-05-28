# アプローチ1: LLMによるプロンプト augmentation

> FoleyForge 品質向上戦略の解説ドキュメント（1/4）
> 対象読者: このテーマを初めて学ぶ人

---

## 1. 一言でいうと

**「ユーザが入力した曖昧な指示を、LLMが拡散モデルにとって分かりやすい『良いプロンプト』に書き換える」** という手法です。

augmentation（オーグメンテーション）は「拡張・水増し」という意味です。ここでは「プロンプトを豊かに拡張する」というニュアンスで使われています。

---

## 2. なぜこれが必要なのか

### 2.1 拡散モデルは「言葉の解釈」が得意ではない

Stable Audio Openのような音声生成モデルは、テキストプロンプトを受け取って音を作ります。しかし、これらのモデルは人間のように言葉の裏を読んでくれません。

例えば、ユーザがこう入力したとします。

```
「夜の路地で女の子が走るシーン」
```

人間なら「足音が響くんだろうな」「夜だから静かで、足音が目立つだろう」と想像できます。しかし拡散モデルは、この短い日本語（あるいは英語）から、具体的な音のイメージをうまく構築できません。

### 2.2 プロンプトの質が出力の質を決める

ここが最も重要な点です。研究では、**プロンプト（キャプション）の質が悪いと、生成される音の質も悪くなる**ことが繰り返し示されています。

GRPOの研究では、次のように述べられています。

> データの質はtext-to-audio生成システムの制御性と忠実度において重要な役割を果たし、質の低いキャプションは誤った音声イベント、順序の乱れ、プロンプトの意味的誤解を頻繁に引き起こす。[引用1]

つまり、「ゴミを入れればゴミが出てくる（Garbage In, Garbage Out）」という原則が、ここでも当てはまります。

---

## 3. どうやって解決するのか

### 3.1 LLMを「翻訳者」として使う

解決策はシンプルです。ユーザの入力と拡散モデルの間に、LLM（大規模言語モデル）を「翻訳者」として挟みます。

```
ユーザ入力（曖昧）
   ↓
LLM（翻訳者・解釈者）
   ↓
拡散モデル向けの良いプロンプト（具体的）
   ↓
拡散モデル
```

LLMは言葉の解釈が得意です。「夜の路地で女の子が走る」という入力から、「濡れたアスファルトの上を走る足音、夜の静けさ、遠くの街の音、緊張感のある雰囲気」のように、音に関する具体的な描写を生成できます。

### 3.2 研究での裏付け

GRPOの研究では、この手法の効果が明確に示されています。

> 我々の手法は、まず大規模言語モデル（LLM）を用いて高忠実度で詳細な音声キャプションを生成し、テキストと音声の意味的整合性を大幅に改善する。特に曖昧または不十分なプロンプトに対して効果がある。[引用1]

ここで重要なのは「**特に曖昧または不十分なプロンプトに対して効果がある**」という部分です。FoleyForgeのユーザは、必ずしも音響の専門用語を知っているわけではありません。「なんとなくこういうシーン」という曖昧な入力をしてくる可能性が高いです。だからこそ、LLMによる補完が効きます。

---

## 4. FoleyForgeでの応用: 構造化データという工夫

### 4.1 ただ書き換えるだけではない

FoleyForgeでは、単に「LLMにプロンプトを書き換えさせる」だけではなく、**構造化データという中間表現を経由する**設計を採用しています。

```
ユーザ入力
   ↓
LLM → 構造化データ（JSON形式の整理された情報）
   ↓
プロンプトビルダー → 最終プロンプト
   ↓
拡散モデル
```

### 4.2 構造化データを挟む理由

研究の世界でも、「構造化」がキーワードになっています。GRPOの研究では、LLMから引き出すキャプションについて次のように述べています。

> 豊かで正確、かつ構造的に一貫した音声キャプションをLLMから引き出すために、注意深く設計された一連のプロンプトを用いる。[引用1]

「構造的に一貫した（structurally consistent）」という点がポイントです。LLMに自由に書かせるのではなく、決まった枠組み（スキーマ）に沿って情報を整理させることで、毎回安定した質のプロンプトが得られます。

### 4.3 形容詞・副詞の重要性

興味深い研究として、AudioSetMixがあります。この研究は、既存の音声データセットに「修飾語（形容詞や副詞）」が欠けていることを指摘しています。

> 注目すべきは、我々のデータセットは既存データセットにおける修飾語（形容詞・副詞）の欠如に対処している点である。[引用5]

これは、FoleyForgeのプロンプト生成にも示唆を与えます。「footsteps（足音）」だけでなく、「fast（速い）」「heavy（重い）」「muffled（くぐもった）」といった修飾語を意識的に加えることが、音の質感を伝える鍵になります。構造化データのスキーマに「強度」「質感」といったフィールドを設けることは、まさにこの修飾語を引き出す仕掛けになります。

---

## 5. 関連する発展的な手法

このアプローチには、いくつかの発展形があります。参考までに紹介します。

### 5.1 "audionese"（オーディオ語）への書き換え

ある研究グループは、ユーザのクエリを「audionese」と呼ばれる、生成の忠実度を高めることが知られた「豊かな潜在プロンプト分布」に書き換える手法を提案しています。[引用6]

これは、「拡散モデルが好む言葉づかい」が存在するという考え方です。人間にとって自然な言葉と、モデルにとって効きやすい言葉は違う、という発想です。

### 5.2 Chain-of-Thoughtによる段階的拡張

PPPRという研究では、テキスト記述を言い換えたり段階的に展開したりすることで、言語的多様性と信頼性を高める手法が提案されています。[引用6]

これは、以前の議論で出てきたChain-of-Thought（思考の連鎖）の考え方と繋がります。

### 5.3 CLAPによる品質フィルタリング

EzAudioという研究では、自動キャプションシステムとLLMによる洗練に加えて、「CLAPによる品質ゲーティング」を組み合わせています。[引用6]

これは、生成されたキャプションの質をCLAP（後述のアプローチ3で詳しく扱います）で評価し、質の低いものを除外する仕組みです。プロンプト拡張と品質評価を組み合わせる発想で、FoleyForgeの設計とも親和性があります。

---

## 6. まとめ

| ポイント | 内容 |
|---------|------|
| 何をするか | ユーザの曖昧な入力を、LLMが具体的で良いプロンプトに変換 |
| なぜ効くか | プロンプトの質が出力の質を直接左右するから |
| 特に効く場面 | 曖昧・不十分な入力（FoleyForgeの典型的なユースケース） |
| FoleyForgeの工夫 | 構造化データを中間表現として挟み、安定した質を確保 |
| 関連する鍵 | 修飾語（形容詞・副詞）を意識的に加えること |

このアプローチは、FoleyForgeの「構造化データ」設計の理論的な土台になっています。次のアプローチ（推論時パラメータの最適化）と組み合わせることで、さらに品質を高められます。

---

## 引用元と参考文献

### 説明中で引用した箇所

**[引用1]** GRPO研究より（セクション2.2, 3.2, 4.2で引用）
- 原文（2.2）: "Data quality plays a critical role in the controllability and fidelity of text-to-audio generation systems, where poor-quality captions frequently lead to incorrect audio events, disordered temporal sequencing, or semantic misinterpretations of the input prompt."
- 原文（3.2）: "Our method first employs a large language model (LLM) to generate high-fidelity, richly detailed audio captions, substantially improving text-audio semantic alignment, especially for ambiguous or underspecified prompts."
- 原文（4.2）: "we first design a set of carefully engineered prompts that elicit rich, accurate, and structurally consistent audio captions from large language models"

**[引用5]** AudioSetMix研究より（セクション4.3で引用）
- 原文: "Notably, our dataset addresses the absence of modifiers (adjectives and adverbs) in existing datasets."

**[引用6]** Text-to-Audio Generation総説より（セクション5で引用）
- audionese、PPPR、EzAudioに関する記述

### 参考文献リスト

1. **Investigating Group Relative Policy Optimization for Diffusion Transformer based Text-to-Audio Generation** (GRPO)
   https://arxiv.org/pdf/2603.01565

2. **Make-An-Audio 2: Temporal-Enhanced Text-to-Audio Generation**
   https://arxiv.org/pdf/2305.18474

3. **Text-to-Audio Generation using Instruction-Tuned LLM and Latent Diffusion Model** (TANGO)
   https://arxiv.org/pdf/2304.13731

4. **Performance Improvement of Language-Queried Audio Source Separation Based on Caption Augmentation From LLMs (DCASE 2024 Task 9)**
   https://arxiv.org/pdf/2406.11248

5. **AudioSetMix: Enhancing Audio-Language Datasets with LLM-Assisted Augmentations**
   https://arxiv.org/pdf/2405.11093

6. **Text-to-Audio (T2A) Generation（総説・トピック解説）**
   https://www.emergentmind.com/topics/text-to-audio-t2a-generation

### 注記

引用した論文の一部（特にGRPO研究、arxiv 2603.01565）は、URLの形式から将来公開予定のプレプリントの可能性があります。閲覧できない場合は、arXivで "GRPO text-to-audio" などのキーワードで最新版を検索してください。
