# Stable Audio Open × diffusers：生成が「無限再帰／無音／NaN」になる問題の切り分け

> 作成日: 2026-06-06
> ステータス: 事例ノート（エンジニアリング学習用・デバッグ記録）
> 用途: SAO の生成が落ちる/壊れた時、**原因特定を速くするための参照記録**。Phase 0 スモークテスト（手順3–5）で遭遇。
> 関連: [phases/phase0/README.md](../../docs/phases/phase0/README.md)（手順3–5・§4）/ [mps-unavailable-in-sandbox.md](./mps-unavailable-in-sandbox.md)（先行する**別**問題＝サンドボックス）/ `src/spikes/phase0/smoke_generate.py`（`apply_final_step_noise_guard()`）/ [diffusers #8728](https://github.com/huggingface/diffusers/issues/8728)

---

## 0. TL;DR（30秒で結論）

「雨の森」を1本生成するだけのスモークテストが、**3段階で壊れた**（クラッシュ → 無音 → NaN）。最終的な真因と対処：

| 観点 | 内容 |
|---|---|
| **真因** | **拡散の“最終ステップの SDEノイズ計算”だけ**が壊れる。**拡散本体（DiT）は完全に健全**（latentは綺麗に収束）。スケジューラ最後の sigma 境界で torchsde が破綻する |
| **2つの壊れ方** | `final_sigmas_type="zero"`（SAO既定）→ 最終σ=0 が範囲外 → **無限再帰(RecursionError)**／`"sigma_min"`（途中の誤った手当て）→ 最後の2σが同値 → `÷|t1−t0|=÷0` → **NaN** |
| **対処** | ノイズ生成器をラップし、**最終ステップ（区間が退化/範囲外）では SDEノイズ=0** を返す。denoising 完了時の SDE 確率項は物理的に ~0 なので 0 で正しい。既定 `"zero"` のまま完走（`smoke_generate.py` の `apply_final_step_noise_guard()`） |
| **MPS固有か** | **No**。dtype（float16/32）も無関係。SAO×この diffusers の境界処理バグ。MPS は NaN を 0 に潰すので「無音」に見えただけ |

> 一番の教訓：**「クラッシュせず結果だけ壊れる（無音/NaN）」が一番厄介。** 出力に**客観チェック（無音/NaN検出）を必ずかけ**、**計装で“どこで”壊れるかを局在化**してから直す。

---

## 1. 背景：スモークテストの狙い

[phase0 README](../../docs/phases/phase0/README.md) の手順3–5＝「1モデルを端から端まで薄く通す」縦スライス。雨の森プロンプト1本を MPS で生成し、**wavが出て耳で方向性が分かるか**を確認する最小コード。

> 先行して**別の壁**（MPSがサンドボックス内で無効）があり、それは [mps-unavailable-in-sandbox.md](./mps-unavailable-in-sandbox.md) に別記。本書は**生成そのもの**が壊れた話。

---

## 2. 時系列：何が起きたか（ここが本題）

7手で局在化した。各手で「1つだけ変えて」情報を得ているのがポイント。

| # | 試したこと | 結果 | そこで得た情報 |
|---|---|---|---|
| 1 | 既定設定で生成（float32・100step） | step **99/100** で **RecursionError** | 拡散は99stepまで回る＝**MPSは動く**。torchsde の**最終ステップ**で落ちる |
| 2 | `final_sigmas_type="sigma_min"` に変更（クラッシュ回避狙い） | 完走するが **MPSで全ゼロ＝無音** | クラッシュは消えたが**出力が壊れた**＝「落ちないが間違う」型に転落 |
| 3 | 同設定を **CPUで**実行（device を1つだけ変える） | **NaN**（ゼロでなく！） | **MPS単独でない**＝数値発散。MPSは NaN を 0 に潰していた（同じ根・違う見え方） |
| 4 | dtype を **float16** に（公式CUDA例に合わせる） | 既定で再び Recursion／sigma_min で NaN | **dtypeは無関係**。sigmas は既定 float32 で計算されるため、float16 にしても境界は変わらない |
| 5 | ノイズ生成器を **単体**で境界値で叩く（ユニット試験） | 開始超過(500.00006)は**正常**、最終(σ→0)だけ **RecursionError** | **NaN の出所はノイズ生成器“ではない”**。クラッシュ要因は最終境界だけ |
| 6 | **ステップ毎に latent を計装**（callback） | step0–6 は健全(795→6.4)、**step7(最終)で突然 NaN** | **NaN は最終ステップ限定**。拡散本体は完全に健全と確定 |
| 7 | **ガード**（最終ステップのノイズ=0）＋既定スケジューラ | **実音！** nan=False・信号あり(OK) | 根本原因＝最終ステップの SDEノイズと確定。対処成立 |

> 手2の「クラッシュを消すために `sigma_min` に変えた」は**誤った手当て**だった（NaNに化けた）。**症状を消すだけの対処は、別のもっと静かな症状に化ける**典型。

---

## 3. 根本原因（なぜ最終ステップだけ壊れるか）

スケジューラは各ステップで `noise = noise_sampler(sigmas[i], sigmas[i+1])` を計算。`noise_sampler` の中身は（ソース確認済み）：

```
return self.tree(t0, t1) / (t1 - t0).abs().sqrt()      # tree は [σ_min=0.3, σ_max=500] 上のブラウン木
```

**通常ステップ**は σ が範囲内かつ t0≠t1 なので問題なし。壊れるのは**最終ステップの σ 境界**だけ：

```mermaid
flowchart TD
    D["拡散ループ（DiT・MPS）<br/>latent 795 → … → 6.4 と健全に収束"]:::ok --> F{"最終ステップの<br/>SDEノイズ計算"}
    F -- "final_sigmas='zero'（SAO既定）<br/>最終σ=0 ＜ σ_min=0.3" --> R["範囲外参照<br/>→ torchsde が区間分割を続け<br/>**無限再帰 = RecursionError**"]:::bad
    F -- "final_sigmas='sigma_min'（誤った手当て）<br/>最後の2σが both 0.3" --> N["t1−t0 = 0<br/>→ tree / √0 = **÷0**<br/>→ **NaN**"]:::bad

    subgraph Legend["凡例"]
        direction LR
        LO["緑＝健全"]:::ok
        LB["赤＝破綻"]:::bad
    end
    classDef ok fill:#ecfdf5,stroke:#059669,color:#000
    classDef bad fill:#fee2e2,stroke:#dc2626,color:#000
```

**最重要の事実：拡散本体は完全に健全**（latent が綺麗に収束）。壊れるのは**最後の1ステップの SDE 確率項だけ**。だから「全体が壊れている」と思い込むと迷子になる。

---

## 4. 切り分けの決め手（4つの技）

| 技 | 何をした | 効果 |
|---|---|---|
| **差分テスト**（手3） | 同設定で **CPU vs MPS** | CPU=NaN／MPS=ゼロ → 「MPSのせい」誤認を是正。**同じ根・違う症状**（MPSはNaN→0）と判明 |
| **ユニット単体試験**（手5） | パイプライン全体でなく**ノイズ生成器だけ**を境界値で叩く | 「開始超過は無害／最終境界だけ壊れる」を分離。容疑者を1つに絞る |
| **ステップ計装**（手6） | `callback` で **step毎の latent** を覗く | NaN が**最終stepに局在**と一発特定。「どこで」を掴んでから直す |
| **ソース確認**（手3の後） | `scheduling_dpmsolver_sde.py` を読む | `÷|t1−t0|` を見て「同値→÷0→NaN」を**推測でなく裏取り** |

---

## 5. 対処（ガード）

最終ステップの SDE 確率項は物理的に ~0 でよい（denoising 完了）。なので**区間が退化/範囲外なら noise=0** を返すガードを噛ませる（`smoke_generate.py`）：

```python
def _guarded(self, sigma, sigma_next):
    t0 = self.transform(torch.as_tensor(sigma))
    t1 = self.transform(torch.as_tensor(sigma_next))
    # 退化(|t1-t0|≈0)／境界外(<=0) ＝ 最終ステップ → SDEノイズ0
    if float((t1 - t0).abs()) < 1e-9 or float(t1) <= 0.0 or float(t0) <= 0.0:
        ref = self.tree(t0.clamp(min=1e-3), t0.clamp(min=1e-3) + 1e-3)
        return torch.zeros_like(ref)
    return _orig(self, sigma, sigma_next)   # 通常ステップは無改変
```

- **SAO既定 `final_sigmas_type="zero"`（クリーンな出力）のまま完走**。
- 通常ステップの挙動は変えない（最終ステップだけ作動）。
- ⚠️ **本番の生成スクリプトにも同じガードが要る**（SAO×MPS/この diffusers で生成する限り）。将来は upstream 修正・版固定・別スケジューラも選択肢。

---

## 6. 未解明の点（正直に）

- 検索では「CUDA(float16) では警告は出るが完走する」との情報があるが、**我々の検証では float16 でも同じく破綻**した（dtypeは効かず）。**なぜ一部環境で完走するのか**は未確認（torchsde の版差・seed・乱数経路の違いの可能性）。
- 我々の対処は**破綻機構を直接潰す**ので、その差に依存せず効く。**経験的に直る対処で十分**とし、内部機構の完全解明には深入りしない（コスト対効果。前作 [mps-unavailable-in-sandbox.md](./mps-unavailable-in-sandbox.md) と同じ姿勢）。

---

## 7. 持ち帰る教訓（次に同種が来たら）

1. **「クラッシュせず結果だけ壊れる」型が最悪。** クラッシュは場所を教えるが、無音/NaNは黙る。→ **出力に必ず客観チェック（無音・NaN・クリップ検出＝L0 DSP）をかける**。今回もそれで「完走したのに無音」に気づけた。
2. **差分テストは“同じ根・違う症状”を暴く。** CPU=NaN/MPS=ゼロで「MPSのせい」と誤認しかけた。**1変数(device)だけ**変えると本質（数値発散）が見える。
3. **計装でピンポイント局在化。** step毎の latent 監視で「最終ステップだけ」と一発特定。**闇雲に直す前に「どこで」**を掴む。
4. **ユニットを切り出して叩く。** 全体でなく怪しい部品（ノイズ生成器）を単体で境界値テスト → 容疑者を即絞る。
5. **最後はソースを読む。** `÷|t1−t0|` を見て初めて「同値→÷0」が確定。ライブラリのバグは**ソースで裏取り**。
6. **対処は物理/数学の妥当性で設計。** 最終ステップのSDE項は~0 → 0で埋めるのが“正しい”。場当たりでなく原理ベースだと安心して残せる。
7. **症状を消すだけの手当てに注意。** `sigma_min` でクラッシュは消えたが NaN に化けた。**根本でなく症状を抑えると、より静かな別症状になる**。

---

## 8. 付録：主要な証拠ログ

```text
# 手1：既定 → 最終ステップでクラッシュ（拡散は99stepまで回っている）
 99%|█████████▉| 99/100 ...
RecursionError: maximum recursion depth exceeded   （torchsde brownian_interval._split）

# 手5：ノイズ生成器を単体で叩く
in-bounds              -> nan=False
start-overshoot(>500)  -> nan=False     ← 開始超過は無害
near-min(==0.3)        -> nan=False
final-to-zero(<0.3)    -> CRASH RecursionError   ← 最終境界だけ壊れる

# 手6：step毎の latent（NaNは最終stepに局在）
step 0: nan=False absmax=795.4
 ... 健全に収束 ...
step 6: nan=False absmax=6.354
step 7: nan=True  absmax=0           ← 最終ステップで突然NaN

# 手7：ガード適用 → 実音
step 7: nan=False absmax=5.562
FINAL AUDIO: nan=False | rms=0.0172 | absmax=0.423   ← 無音でもNaNでもない実信号
```
