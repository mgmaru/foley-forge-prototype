# 評価モデル（CLAP / PAM）の仕組み

> approach-3（Best-of-N + リランキング）で使われる評価モデル CLAP と PAM が、内部でどのように動くのかを掘り下げたドキュメント。
> FoleyForgeの処理フローでは Step 6（評価とフィルタリング）に相当する。

---

## このドキュメントの位置付け

`docs/foley-forge-dev.md` の全体フロー図で、Step 6（評価とフィルタリング）を「モデル呼び出し」として黄色で強調している。ここで使われる **CLAP** と **PAM**（PQスコアの実装候補）は、Step 2 の LLM や Step 5 の拡散モデルと同じ「黒箱」なのか、それとも別の性質を持つのか、を調査した結果をまとめる。

approach-3 ドキュメントでは「CLAPで指示追従性を測る」「PQスコアで音響品質を測る」とだけ書かれていて、内部の計算方法までは踏み込んでいない。本ドキュメントはその補足である。

---

## 結論（先に）

| 観点 | 結論 |
|------|------|
| Step 6 は LLM を使うか？ | **使わない**。CLAP も PAM も対比学習ベースの埋め込みモデル |
| 出力は確率的か？ | **完全に決定論的**。同じ入力なら必ず同じスコアが返る |
| 「黒箱」性は Step 2/5 と同じか？ | **意味合いが違う**。生成系（Step 2, 5）は出力が確率的、評価系（Step 6）は出力こそ決定論的だが内部の埋め込み空間は学習されたもので解釈できない |
| 開発者が調整できる余地は？ | モデル選択、PAMの対立プロンプト文、スコア閾値、複数指標の重み付け |

---

## 1. CLAP（Contrastive Language-Audio Pretraining）

### 1.1 一言で言うと

「CLIPの音声版」。画像の代わりに音声を扱う、対比学習ベースのマルチモーダル埋め込みモデル。

### 1.2 アーキテクチャ

- **Text encoder**: RoBERTa系（テキストを埋め込みベクトルに変換）
- **Audio encoder**: HTSAT系などの音声トランスフォーマー（音声を埋め込みベクトルに変換）
- 両方を**共有の512次元埋め込み空間**に投影するように対比学習で訓練される
- LAION-CLAPは約460万件の音声-テキストペアで事前学習

### 1.3 CLAPスコアの計算方法

非常にシンプル。プロンプトと音声をそれぞれエンコードし、**コサイン類似度**を取るだけ。

```
1. プロンプト t をtext encoderで埋め込み: u = TextEnc(t)  ∈ ℝ^512
2. 音声 x をaudio encoderで埋め込み:  v = AudioEnc(x) ∈ ℝ^512
3. CLAPスコア = cos(u, v) = (u · v) / (||u|| · ||v||)
```

Stable-Audio-Metricsの実装でも、以下の1行で完結している:

```python
torch.nn.functional.cosine_similarity(
    audio_embeddings,
    text_emb[id].unsqueeze(0),
    dim=1,
    eps=1e-8,
)
```

L2正規化されたベクトル同士のコサイン類似度は単純にドット積に等しい。範囲は -1〜+1 だが、実用上は0付近〜1付近に出る。

### 1.4 決定論性

- 事前学習済みモデルの重みは固定
- サンプリング、温度パラメータ、確率的な処理は**一切ない**
- 同じ音声と同じプロンプトに対して、CLAPスコアは**常に同じ値**を返す

### 1.5 FoleyForgeでの用途（指示追従性の評価）

各候補音声 x に対して、元のプロンプト t との CLAPスコアを計算する。
スコアが高い ＝ プロンプトに忠実な音、スコアが低い ＝ プロンプトから外れた音、と解釈する。

approach-3 で言う「**指示追従性**」の指標がこれにあたる。

---

## 2. PAM（Prompting Audio-Language Models for Audio Quality Assessment）

### 2.1 一言で言うと

**CLAPを流用した「音響品質」評価メトリック**。Deshmukh et al., INTERSPEECH 2024。
PAM自体は新しいモデルを訓練していない。CLAPを「使い方を工夫して」品質評価に転用したのがポイント。

### 2.2 課題と着想

純粋なCLAPは「テキストと音声が一致しているか」しか測れない。「音響的に高品質か」を直接測ることはできない。

そこで PAM では、**対立する2つのプロンプト**を使って、音声がどちら寄りかをCLAPに判定させる。これを **antonym prompt strategy（対立プロンプト戦略）** と呼ぶ。

### 2.3 PAMスコアの計算方法

```
1. 対立する2つのプロンプトを用意する
   - 高品質プロンプト t_high : 例 "the sound is clear and clean"
   - 低品質プロンプト t_low  : 例 "the sound is noisy and with artifacts"

2. 各プロンプトをCLAP text encoderで埋め込み
   - u_high = TextEnc(t_high)
   - u_low  = TextEnc(t_low)

3. 評価対象の音声 x をCLAP audio encoderで埋め込み
   - v = AudioEnc(x)

4. それぞれとのドット積（類似度）を計算
   - z_high = u_high · v
   - z_low  = u_low · v

5. softmaxで「高品質寄り確率」に正規化
   - p_high = exp(z_high) / (exp(z_high) + exp(z_low))

6. PAMスコア = p_high  ∈ [0, 1]
```

スコアが1に近い ＝ 「clear/clean」プロンプト寄りの音 ＝ 高品質、
スコアが0に近い ＝ 「noisy/artifact」プロンプト寄りの音 ＝ 低品質。

### 2.4 なぜ「対立プロンプト」が必要なのか

論文では、単一プロンプト戦略（高品質プロンプト1つだけと比較）には**言語的曖昧性**の問題があると指摘されている。「clear」というプロンプトは「澄んだ」と「分かりやすい」など複数の解釈が可能で、CLAPがどの軸で評価しているかが定まらない。

対立する2つのプロンプトを比較に使うことで、CLAP埋め込み空間内で「品質軸」を明示的に定義できる。

### 2.5 決定論性

- CLAPのモデル重みは固定
- 対立プロンプト文も固定値
- softmaxはあるが、これは確率分布への正規化であって**サンプリングではない**

→ **完全に決定論的**。同じ音声を入れれば必ず同じPAMスコアが返る。

### 2.6 FoleyForgeでの用途（音響品質の評価）

approach-3 で言う「**音響的美しさ / 音響品質**」の指標として、PAMスコアを採用する想定。`docs/foley-forge-dev.md` 9章でも「PQスコアの実装方法（PAMスコアから始める想定）」と明記されている。

---

## 3. LLMとの違い

ここが今回の調査の最大のポイント。Step 6 はLLM呼び出しと**性質が違う**。

| 項目 | LLM（Step 2 のClaude / Ollama） | CLAP / PAM（Step 6） |
|------|--------------------------------|----------------------|
| モデル種別 | 生成モデル（autoregressive） | 埋め込みモデル（contrastive） |
| 訓練目的 | 次トークン予測 | 音声とテキストの埋め込み一致 |
| 推論時の処理 | トークンを1つずつサンプリング生成 | 1回エンコード + ドット積 |
| 確率的要素 | あり（温度・トップP・サンプリング） | なし |
| 出力の再現性 | 同じ入力でも結果が変わりうる | 同じ入力なら必ず同じ |
| 内部の解釈性 | 不透明（attentionは可視化できるが意味は不明） | 不透明（埋め込み空間は学習されたもの） |
| 「黒箱」の意味 | 「出力が予測不能」 | 「内部表現が解釈不能」 |

「LLMで評価しているか」という質問への答えは、**Noの上に「LLMとは別種のモデルである」というのが正確**。

---

## 4. 設計への含意（foley-forge-dev.md との接続）

### 4.1 Step 6 を「黒箱」として強調すべきか？

YES。理由は2つ:

1. **モデル選択で挙動が変わる**: LAION-CLAP の学習データバイアスがそのままスコアに乗る。例えばLAION-CLAPはFreesoundなどの音響データで訓練されているため、特定ジャンルの音には強く、別ジャンルには弱い、といった偏りが必ずある。開発者はこの埋め込み空間の中身を直接修正できない。

2. **「スコアの妥当性」が経験則に依存する**: CLAPスコア 0.6 が「良い音」を意味するかは状況による。閾値はデータを見て決めるしかない。

ただし、Step 2/5 とは「黒箱の性質」が違う、という注記は必要。それが今回のドキュメント更新（凡例の「※生成系は確率的、評価系は決定論的」）の動機。

### 4.2 開発者が調整できる余地

| 調整対象 | 影響 |
|---------|------|
| CLAPモデルの選択（LAION-CLAP, MS-CLAP, ...） | 埋め込み空間が変わる。ドメインに合うものを選ぶ |
| PAMの対立プロンプト文 | 「何を品質と見なすか」の定義が変わる。FoleyForgeなら "cinematic and high-fidelity" vs "muffled and low-quality" 等にカスタマイズ可 |
| スコア閾値 | フィルタリングの厳しさ |
| 複数指標の重み付け | CLAP（指示追従性）と PAM（音響品質）のトレードオフ |
| 戦略ごとの主指標 | approach-3 の3戦略で、それぞれ主指標を変える |

これらは全てコードで決定論的に書ける部分なので、図上は「アルゴリズム」扱い。Step 6 が黒箱なのは「CLAP/PAM の埋め込み計算そのもの」だけ。

### 4.3 verifier hacking との関係

approach-3 ドキュメントで触れられている **verifier hacking**（単一指標を最大化するとその抜け穴を突いた音が選ばれる）は、CLAP/PAM のスコアが**決定論的だからこそ**問題になる。つまり「同じ抜け穴は再現性100%で突かれる」ので、Best-of-N で大量生成すると、運悪く抜け穴を突いた候補が選ばれるリスクが系統的に存在する。

これがFoleyForgeで CLAP + PAM の**複数指標**を組み合わせる理由になっている。決定論性と裏返しのリスクへの対処。

---

## 5. 参考文献

### CLAP関連

- [LAION-AI/CLAP（公式実装）](https://github.com/LAION-AI/CLAP)
- [Stable-Audio-Metrics の clap_score.py（実装例）](https://github.com/Stability-AI/stable-audio-metrics/blob/main/src/clap_score.py)
- [Hugging Face CLAP ドキュメント](https://huggingface.co/docs/transformers/en/model_doc/clap)
- [Audiocraft CLAP Consistency API](https://facebookresearch.github.io/audiocraft/api_docs/audiocraft/metrics/clap_consistency.html)

### PAM関連

- [PAM: Prompting Audio-Language Models for Audio Quality Assessment（arXiv 2402.00282）](https://arxiv.org/abs/2402.00282)
- [PAM Interspeech 2024 PDF](https://www.isca-archive.org/interspeech_2024/deshmukh24b_interspeech.pdf)
- [PAM 公式実装（soham97/pam）](https://github.com/soham97/pam)

---

## 6. まとめ

- CLAPとPAMは**LLMではない**。CLIP系の対比学習埋め込みモデルである。
- スコア計算は**完全に決定論的**。同じ入力なら必ず同じ出力。
- ただし埋め込み空間そのものは学習されたもので解釈できないため、「内部表現の意味では黒箱」。
- 開発者が調整できるのは「モデル選択、対立プロンプト、閾値、重み付け」など外側のパラメータのみ。
- 決定論性ゆえに **verifier hacking** のリスクが系統的にあり、FoleyForgeで複数指標を組み合わせる動機になっている。
