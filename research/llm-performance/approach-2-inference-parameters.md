# アプローチ2: 推論時パラメータの最適化

> FoleyForge 品質向上戦略の解説ドキュメント（2/4）
> 対象読者: このテーマを初めて学ぶ人

---

## 1. 一言でいうと

**「拡散モデルを動かすときの『設定値（パラメータ）』を、用途に合わせて調整することで、生成される音の質を上げる」** という手法です。

モデルを訓練し直すわけではありません。**すでに完成しているモデルを、どう動かすか**の調整です。だから「推論時（inference-time）」と呼びます。推論とは、モデルが実際に音を生成する処理のことです。

---

## 2. 主役は「CFG（Classifier-Free Guidance）」

推論時パラメータの中で、最も重要なのがCFGです。日本語では「分類器なしガイダンス」と訳されます。

### 2.1 CFGとは何か

拡散モデルは、ノイズから少しずつ音（や画像）を作り上げていきます。このとき、「プロンプトをどれだけ強く守るか」を調整するのがCFGです。

直感的には、**CFGは「プロンプトへの忠実さのツマミ」**だと考えてください。

- CFGを上げる → プロンプトに強く従う
- CFGを下げる → プロンプトを緩く解釈する

CFGの解説では、次のように説明されています。

> CFGは、条件付きデノイザーと条件なしデノイザーをガイダンススケールを使って補間する手法であり、プロンプトへの追従性を高める。[引用1]

「条件付き」はプロンプトを考慮した予測、「条件なし」はプロンプトを無視した予測です。この2つを混ぜる比率がCFGスケールです。

### 2.2 なぜCFGの調整が訓練不要で効くのか

CFGの大きな利点は、**モデルを再訓練せずに、推論時に挙動を変えられる**ことです。

> （CFGは）モデルを再訓練することなく、推論時に条件への追従性とサンプルの多様性のトレードオフを制御する、シンプルで直感的な方法を提供する。[引用4]

FoleyForgeにとってこれは重要です。ファインチューニングはしない方針なので、「訓練せずに質を変えられる」CFGの調整は、現実的に使える数少ない手段の一つです。

---

## 3. ここが肝: CFGのトレードオフ

### 3.1 上げればいい、というものではない

「プロンプトに忠実なほうが良いなら、CFGを最大にすればいいのでは?」と思うかもしれません。しかし、ここに罠があります。

**CFGを上げすぎると、音質が劣化します。** これがトレードオフです。

β-CFGの研究が、このトレードオフを非常に分かりやすく説明しています。

> 強いガイダンスを使うと、生成される画像は条件付けられたテキストに完璧に一致するが、その代償として品質が犠牲になる。逆に、弱いガイダンスを使えば高品質な結果を生成できるが、生成された画像はプロンプトに合わない。[引用2]

これは画像生成の文脈ですが、音声生成でも同じことが起きます。つまり。

- **CFG高すぎ**: プロンプトには従うが、音が歪む・ノイズが乗る
- **CFG低すぎ**: 音はきれいだが、プロンプトと違う音になる

### 3.2 高すぎるCFGは「アーティファクト」を生む

「アーティファクト」とは、本来あるべきでない不自然なノイズや歪みのことです。CFGを上げすぎると、これが発生します。

> 高い静的ガイダンスは、アーティファクト、色の歪み、多様性の喪失を悪化させる。[引用5]

音声の場合、「色の歪み」は「周波数特性の歪み」に相当します。本来クリアであるべき音が、ザラついたりこもったりするわけです。

### 3.3 計算コストも上がる

CFGにはもう一つコストがあります。プロンプトを考慮した予測と無視した予測の両方を計算するため、**計算量が約2倍**になります。

> CFGによるサンプリングは、条件なし生成よりも計算コストが高く、通常はデノイジングステップごとにモデルの順伝播を2回必要とする。[引用1]

FoleyForgeはBest-of-Nで大量に生成する設計なので、この計算コストは無視できません。CFGの値は「質」だけでなく「速度」にも影響します。

---

## 4. もう一つのパラメータ: 推論ステップ数

### 4.1 ステップ数とは

拡散モデルは、ノイズから音を作るまでに「何回処理を繰り返すか」を指定できます。これが推論ステップ数です。

- ステップ数が多い → 時間はかかるが、丁寧に生成される
- ステップ数が少ない → 速いが、粗くなる可能性

### 4.2 多ければいいわけでもない

ステップ数も、ある程度を超えると品質の向上が頭打ちになります。「200ステップが100ステップの2倍良い」わけではありません。どこかに「これ以上増やしても無駄」という点があります。

FoleyForgeでは、品質と速度のバランスを見て、実験的に適切なステップ数を決めることになります。

---

## 5. 発展: プロンプトごとに最適なCFGは違う

### 5.1 固定値の限界

ここまで「CFGの調整が大事」と説明しましたが、さらに進んだ研究では、**「すべてのプロンプトに同じCFG値を使うこと自体が最適ではない」**ことが指摘されています。

prompt-aware CFG（プロンプト対応CFG）の研究では、固定スケールの限界を次のように指摘しています。

> ガイダンススケールの選択は十分に検討されてこなかった。固定スケールは（中略、最適ではない）。[引用3]

つまり、「速い足音」と「静かな環境音」では、最適なCFG値が違う可能性がある、ということです。

### 5.2 FoleyForgeへの示唆

この研究の発想は、FoleyForgeの設計と深く繋がります。FoleyForgeでは、生成音タイプ（環境音 / 単発SE / 継続音）や戦略（指示追従寄り / バランス / 音響美寄り）ごとに、異なるCFG値を使い分ける設計を採用しています。これは、prompt-aware CFGの「プロンプトに応じてガイダンスを変える」という考え方の、シンプルな実装版と言えます。

prompt-aware CFGの研究では、評価指標として **AudioBox-Aesthetics**（音響的な美しさを測る指標）を使っている点も注目に値します。[引用3] これは、FoleyForgeで「音響的美しさ」を評価する際の候補になります（詳しくはアプローチ3で扱います）。

---

## 6. まとめ

| パラメータ | 役割 | 注意点 |
|-----------|------|--------|
| CFGスケール | プロンプトへの忠実さを制御 | 高すぎると音質劣化・アーティファクト。計算コスト約2倍 |
| 推論ステップ数 | 生成の丁寧さを制御 | 多すぎても頭打ち。速度とのトレードオフ |

**最重要ポイント**: CFGは「上げれば良い」ものではなく、**スイートスポット（最適点）が存在する**。プロンプトへの忠実さと音質のトレードオフを理解し、用途ごとに適切な値を選ぶことが、このアプローチの本質です。

FoleyForgeでは、このトレードオフを「戦略」として表現し、生成音タイプごとに最適なパラメータを内部で自動選択します。ユーザにはCFGの数値を見せず、バックエンドが最適化します。

---

## 引用元と参考文献

### 説明中で引用した箇所

**[引用1]** Classifier-Free Guidance (CFG) 解説より（セクション2.1, 3.3で引用）
- 原文（2.1）: "Classifier-Free Guidance (CFG) is a method in diffusion models that interpolates between conditional and unconditional denoisers using a guidance scale to enhance prompt adherence."
- 原文（3.3）: "sampling with CFG is computationally more expensive than unconditional generation, typically requiring two forward passes through the diffusion model per denoising step"

**[引用2]** β-CFG（Classifier-free Guidance with Adaptive Scaling）より（セクション3.1で引用）
- 原文: "When we use strong guidance, generated images fit the conditioned text perfectly but at the cost of their quality. Dually, we can use small guidance to generate high-quality results, but the generated images do not suit our prompt."

**[引用3]** Prompt-aware CFG より（セクション5.1, 5.2で引用）
- 固定ガイダンススケールの限界、AudioBox-Aestheticsによる評価に関する記述

**[引用4]** Classifier-Free Guidance 解説（apxml）より（セクション2.2で引用）
- 原文: "provides a simple and intuitive way to control the trade-off between condition adherence and sample diversity at inference time, without needing to retrain the model"

**[引用5]** Classifier-Free Diffusion Guidance (CFG) 解説より（セクション3.2で引用）
- 原文: "High static guidance exacerbates artifacts, color distortion, and diversity loss."

### 参考文献リスト

1. **Classifier-Free Guidance (CFG)（トピック解説）**
   https://www.emergentmind.com/topics/classifier-free-guidance-cfg

2. **Classifier-free Guidance with Adaptive Scaling (β-CFG)**
   https://arxiv.org/pdf/2502.10574

3. **Prompt-aware classifier free guidance for diffusion models**
   https://arxiv.org/html/2509.22728

4. **Classifier-Free Guidance in Diffusion Models（apxml 教材）**
   https://apxml.com/courses/intro-diffusion-models/chapter-6-conditional-generation-diffusion/classifier-free-guidance

5. **Classifier-Free Diffusion Guidance (CFG)（トピック解説）**
   https://www.emergentmind.com/topics/classifier-free-diffusion-guidance-cfg

6. **An overview of classifier-free guidance for diffusion models（AI Summer・入門解説）**
   https://theaisummer.com/classifier-free-guidance/

### 注記

CFGの具体的な最適値（例: CFG=3.5でCLAPがピーク）は、使用するモデルやデータセットによって変わります。FoleyForgeでは、採用するモデルごとに実験して最適値を見つける必要があります。上記の論文は「トレードオフが存在する」という原理を理解するためのもので、具体的な数値はモデル依存である点に注意してください。
