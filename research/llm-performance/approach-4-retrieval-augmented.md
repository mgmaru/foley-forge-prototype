# アプローチ4: 検索拡張（RAG）

> FoleyForge 品質向上戦略の解説ドキュメント（4/4）
> 対象読者: このテーマを初めて学ぶ人

---

## 1. 一言でいうと

**「音を一から生成するのではなく、既存の音素材を『参考資料』として引っ張ってきて、それを手がかりに生成する」** という手法です。

RAGは「Retrieval-Augmented Generation（検索拡張生成）」の略です。「Retrieval（検索）」で関連する素材を探し、それで「Generation（生成）」を補強する、という意味です。

なお、このアプローチはFoleyForgeでは**将来拡張**という位置付けです。MVPには含めませんが、品質向上の有力な選択肢として理解しておく価値があります。

---

## 2. なぜこれが必要なのか: ロングテール問題

### 2.1 拡散モデルは「珍しい音」が苦手

拡散モデルは、訓練データにたくさん含まれる「よくある音」は上手に生成できます。しかし、訓練データにあまり含まれない「珍しい音」は苦手です。

これを「ロングテール問題（long-tailed problem）」と呼びます。Re-AudioLDMの研究は、この問題を明確に指摘しています。

> AudioLDMのような最先端モデルは、AudioCapsのような不均衡なクラス分布を持つデータセットで訓練されており、生成性能に偏りがある。具体的には、よくある音声クラスの生成には優れているが、珍しいものでは性能が低く、全体の生成性能を劣化させている。我々はこの問題をロングテールtext-to-audio生成と呼ぶ。[引用1]

### 2.2 FoleyForgeにとっての意味

アニメSEには、独特で珍しい音が多く含まれます。「特殊な魔法のエフェクト音」「架空の機械の駆動音」など、一般的な音声データセットにはあまり存在しない音です。

こうした珍しい音は、拡散モデルが苦手とする領域です。RAGは、この弱点を補える可能性があります。

---

## 3. どうやって解決するのか

### 3.1 基本の流れ

RAGの仕組みは、次のようになります。

```
[入力プロンプト]
   ↓
CLAPで関連する音素材を検索  ← Retrieval（検索）
   ↓
見つかった素材の特徴を抽出
   ↓
その特徴を「参考情報」として拡散モデルに渡す
   ↓
拡散モデルが生成  ← Generation（生成）
```

ここでもCLAP（アプローチ3で登場した、音声とテキストの一致度を測るモデル）が活躍します。プロンプトに近い音素材を、CLAPを使って大量の素材ライブラリから探し出します。

Re-AudioLDMの研究では、次のように説明されています。

> 入力テキストプロンプトが与えられると、まずCLAPモデルを利用して関連するtext-audioペアを検索する。検索されたaudio-textデータの特徴は、TTAモデルの学習をガイドする追加の条件として使われる。[引用1]

### 3.2 参考素材は「お手本」になる

検索されてきた音素材は、生成の「お手本」のような役割を果たします。モデルは、ゼロから音を想像するのではなく、「こういう音に近いものを作ればいい」というヒントを得られます。

> 検索されたaudio-textペアは、（訓練段階では）低頻度の音声イベントのモデリングを改善する補足情報として機能する。推論段階でも、検索拡張戦略はテキストプロンプトに関連する参照を提供し、より正確で忠実な音声生成結果を保証する。[引用1]

---

## 4. 効果: 珍しい音・未知の音にも対応

RAGの効果は、数値でも示されています。Re-AudioLDMは、AudioCapsデータセットで当時のSOTA（最高性能）を達成しました。

> AudioCapsデータセットにおいて、Re-AudioLDMはFréchet Audio Distance (FAD) 1.37という最先端の性能を達成し、既存手法を大きく上回った。[引用1]

（補足: FADは「生成された音が本物の音にどれだけ近いか」を測る指標で、低いほど良いです。）

さらに重要なのは、訓練データになかった音も生成できるようになる点です。

> Re-AudioLDMは、複雑なシーン、珍しい音声クラス、さらには未知の音声タイプに対しても現実的な音声を生成できる。[引用1]

「未知の音声タイプ」にも対応できるというのは、アニメSEのような独特な音を扱うFoleyForgeにとって、大きな魅力です。

---

## 5. 発展形: ラベルなしデータも使える

RAGには、より新しい発展形があります。AudioRAG+という研究は、**ラベル付けされていない音素材も活用できる**点を強調しています。

> 我々のアプローチはtext-to-audio検索を採用しているため、ラベル付けされた外部データソースを必要としない。したがって、in-the-wild（実世界の雑多な）かつラベルなしの音声データセットからの大規模検索をサポートする。[引用4]

これがFoleyForgeにとって何を意味するか。Hiroakiさんが持っている、あるいは集めた**雑多なSE素材ライブラリ**（ラベル付けされていなくても）を、そのまま検索対象にできる可能性があるということです。

---

## 6. FoleyForgeでの位置付けと注意点

### 6.1 なぜ「将来拡張」なのか

RAGは強力ですが、FoleyForgeのMVPには含めません。理由は以下です。

1. **実装の複雑さ** — 検索用の素材ライブラリの構築、CLAP検索の実装、検索結果を拡散モデルに渡す仕組みなど、追加のコンポーネントが多い
2. **モデル側の対応が必要** — Re-AudioLDMのように検索情報を受け取れるよう、モデルが対応している必要がある。既存のStable Audio Openをそのまま使うだけでは実現しにくい
3. **素材ライブラリの準備** — 検索対象となる音素材のデータベースを用意する必要がある

### 6.2 将来的にどう活きるか

FoleyForgeが成熟した段階で、RAGは以下のような形で活きる可能性があります。

- ユーザが蓄積した「採用した音」のライブラリを検索対象にする
- フリー素材ライブラリ（Freesoundなど）をインデックス化して参照する
- 「この音に似た音を作って」というワークフローを実現する

これは、FoleyForgeの「実験ログを蓄積する」という設計思想とも繋がります。蓄積したデータが、将来のRAGの検索対象になりうるのです。

### 6.3 関連手法: SonicRAG

以前の議論で触れたSonicRAGのように、「LLM + 既存SEデータベース」を組み合わせて、音を検索・再結合・合成するフレームワークもあります。これはRAGの考え方を、より実用的なSE制作ワークフローに応用したものです。

---

## 7. まとめ

| ポイント | 内容 |
|---------|------|
| 何をするか | 既存の音素材を検索し、それを参考に生成する |
| 解決する問題 | ロングテール問題（珍しい音が苦手） |
| 検索の仕組み | CLAPで関連素材を探す |
| 効果 | 珍しい音・未知の音にも対応、忠実度が向上 |
| FoleyForgeでの位置付け | 将来拡張（MVPには含めない） |
| 将来の活用 | 蓄積した採用音やフリー素材ライブラリの参照 |

RAGは、4つのアプローチの中で最も実装コストが高い一方、アニメSE特有の「珍しい音」への対応という、FoleyForgeにとって本質的な価値を持っています。MVPの完成後、品質をさらに引き上げる段階で検討する価値があります。

---

## 引用元と参考文献

### 説明中で引用した箇所

**[引用1]** Re-AudioLDM（Retrieval-Augmented Text-to-Audio Generation）より（セクション2.1, 3.1, 3.2, 4で引用）
- 原文（2.1・ロングテール）: "models, such as AudioLDM, trained on datasets with an imbalanced class distribution, such as AudioCaps, are biased in their generation performance. Specifically, they excel in generating common audio classes while underperforming in the rare ones"
- 原文（3.1）: "given an input text prompt, we first leverage a Contrastive Language Audio Pretraining (CLAP) model to retrieve relevant text-audio pairs. The features of the retrieved audio-text data are then used as additional conditions to guide the learning of TTA models."
- 原文（3.2）: "The retrieved audio-text pairs serve as supplementary information that helps improve the modelling of low-frequency audio events in the training stage."
- 原文（4・FAD）: "Re-AudioLDM achieves a state-of-the-art Frechet Audio Distance (FAD) of 1.37, outperforming the existing approaches by a large margin."
- 原文（4・未知タイプ）: "Re-AudioLDM can generate realistic audio for complex scenes, rare audio classes, and even unseen audio types"

**[引用4]** AudioRAG+ より（セクション5で引用）
- 原文: "Since our approach adopts text-to-audio retrieval, it does not require labeled external data sources, thereby supporting large-scale retrieval from in-the-wild and unlabeled audio datasets."

### 参考文献リスト

1. **Retrieval-Augmented Text-to-Audio Generation (Re-AudioLDM)**
   https://arxiv.org/abs/2309.08051
   （PDF: https://arxiv.org/pdf/2309.08051 ）

2. **Re-AudioLDM（ICASSP 2024版PDF）**
   https://personalpages.surrey.ac.uk/w.wang/papers/Yuan%20et%20al_ICASSP_2024.pdf

3. **A Retrieval Augmented Approach for Text-to-Music Generation**
   https://aclanthology.org/2024.nlp4musa-1.6.pdf

4. **AudioRAG+: Feedback-driven Retrieval-augmented Audio Generation with Large Audio Language Models**
   https://arxiv.org/html/2511.01091

5. **Text-to-Audio (T2A) Generation（総説・RAGの位置付け）**
   https://www.emergentmind.com/topics/text-to-audio-t2a-generation

### 注記

RAGを実際に使うには、検索情報を受け取れるモデル（Re-AudioLDMなど）が必要です。Stable Audio Openをそのまま使う構成では、RAGの完全な実装は難しい点に注意してください。将来的にRAGを取り入れる場合、対応モデルの選定や、検索結果をプロンプトに反映する簡易的な方法（検索した素材の説明をプロンプトに加えるなど）から始めることが考えられます。
