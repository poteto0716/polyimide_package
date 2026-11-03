# ポリイミドフィルムの Tg・引張弾性率・CTE 予測

学習データ、データ整備と学習のコード、アンサンブルモデルの構築、結果を見るWebアプリをまとめたものです。
学習済みモデル(`.joblib`)は含みません。`run_train.sh` で作成します。

## 学習データの復号(最初に実行)

測定値、文献番号、学習構造を含む学習データ `training_data.tar.gz.gpg`(AES-256で暗号化)は、このリポジトリには含みません。ファイルとパスワードは別途管理者から受け取り、このフォルダの直下に置いてください。

```bash
sh decrypt_training_data.sh   # gpg が必要。パスワードを聞かれます
```

復号すると `polyimide_models/` と `polyimide_tg_modulus/` にファイルが展開されます。学習、データの作り直し、アプリの構築は、復号後でないと実行できません。平文のまま再配布しないでください。

## 環境

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r polyimide_models/requirements.txt
```

動作確認は Python 3.13 で行いました。

## 実行

```bash
sh run_train.sh   # 3モデルの学習、アンサンブル構築、10分割検証
sh run_app.sh     # 仮想ポリマーの予測、アプリ構築、http://localhost:8766 で表示
```

モデルを作らずにアプリだけ見る場合は、復号後に同梱の予測結果を使います。

```bash
python polyimide_virtual_app/build_app.py
python -m http.server 8766 --bind 127.0.0.1 --directory polyimide_virtual_app/dist
```

## ファイルとディレクトリの説明

### 直下

| 名前 | 内容 |
|---|---|
| `README.md` | この説明書 |
| `training_data.tar.gz.gpg` | 暗号化した学習データ(測定値、文献番号、学習構造)。リポジトリには含まず、別途受け取る |
| `decrypt_training_data.sh` | 学習データを復号して各ディレクトリに展開する |
| `run_train.sh` | 3モデルの学習、アンサンブル構築、10分割検証を実行する |
| `run_app.sh` | 仮想ポリマーの予測、アプリ構築、表示を実行する |
| `polyimide_models/` | データ整備、モデル学習、アンサンブル構築 |
| `polyimide_optimizer/` | 多目的ベイズ最適化 |
| `polyimide_virtual_app/` | 仮想ポリマーの予測とWebアプリ |
| `tci_polyimide/` | TCIモノマーの一覧作成とp-SMILES生成 |
| `polyimide_tg_modulus/` | 元データからの試料と測定の抽出 |

### `polyimide_models/`

| 名前 | 内容 |
|---|---|
| `modeling.py` | 特徴量の計算、アンサンブルのクラス `PolymerEnsemble` |
| `train_models.py` | モデル比較、アンサンブル選抜、独立評価、全データでの再学習 |
| `evaluate_oof10.py` | 10分割検証 |
| `requirements.txt` | 必要なPythonライブラリ |
| `prepare_data.py`, `prepare_film_data.py` | 元データから学習データを作る、フィルムに限定する |
| `apply_modulus_floor.py`, `prepare_film_modulus_range.py`, `audit_film_modulus.py`, `prepare_film_modulus_clean.py` | 弾性率データの範囲処理、監査、除外 |
| `prepare_film_cte.py` | CTEの学習データを作る |
| `evaluate_cleansing.py` | 除外ルールの比較に使ったスクリプト |
| `Tg_film/`, `tensile_modulus_film_clean/`, `CTE_film/` | 物性ごとの評価結果(`results.json`, `oof10_results.json`)と、選抜されたモデル構成(`frozen_selection.json`)。学習後は `ensemble_production.joblib` などもここに作られる |

### `polyimide_optimizer/`

| 名前 | 内容 |
|---|---|
| `mobo_copolyimide.py` | 最適化の本体 |
| `acid_candidates.csv`, `diamine_candidates.csv` | 探索候補と略称の一覧。`use=False` の行は候補から外れる |
| `build_monomer_lists.py` | 候補一覧を作り直す。編集した候補一覧は上書きされる |
| `test_mobo.py` | 組成の規則と略称の解決を検査する |
| `region_search.py` | 物性の目標領域(例 Tg 240〜260 ℃、弾性率 1 GPa以下)に入る確率が高く、学習データから離れた組成を探す |
| `soft_segment.py`, `soft_monomers.csv` | 長鎖シロキサンジアミンなどのソフトセグメント。`region_search.py` が使う |
| `compare_with_training.py`, `compare_web/` | 最適化の結果を学習データと重ねて見るWebページを作る |
| `examples/` | 実測ファイルの書式の例(値は架空) |
| `output*/` | 実行すると作られる。`pareto_predicted.csv`(パレート解)、`next_experiments.csv`(次の実験候補)、`run_summary.json`(実行条件)など。比較ページ `compare/` には実測値が入るので、コミットしない |

### `polyimide_virtual_app/`

| 名前 | 内容 |
|---|---|
| `predict_virtual.py` | TCIモノマーの全組み合わせ(ホモポリマー)を予測する |
| `virtual_predictions.tsv` | 上の予測結果 |
| `build_app.py` | Webアプリのデータと構造式を作る |
| `dist/` | Webアプリ本体(`index.html`, `app.js`, `style.css`) |
| `analyze_structure_effects.py` | 構造と物性の関係を集計する |
| `export_excel.py` | 学習データと仮想データをエクセルに出力する |

### `tci_polyimide/`

| 名前 | 内容 |
|---|---|
| `polyimide_psmiles.py` | 酸二無水物とジアミンからp-SMILESを作る |
| `tci_monomer_list.py` | TCIモノマーの一覧をPubChemから作る |
| `supplemental_products.csv` | PubChemで漏れる製品の追加分(製品番号とCAS) |
| `output/` | モノマー一覧(`tci_dianhydrides.csv`, `tci_diamines.csv`)、全組み合わせのp-SMILES(`tci_polyimide_psmiles.csv`)、件数(`summary.json`) |

### `polyimide_tg_modulus/`

| 名前 | 内容 |
|---|---|
| `extract_polyimide.py` | 元データから試料と測定の表を作る(記録用。パッケージの検証では未実行) |

### 復号すると追加されるファイル

| 場所 | 内容 |
|---|---|
| `polyimide_models/datasets.json`, `features.npz` | 学習データと特徴量(学習と最適化の入力) |
| `polyimide_models/*_dataset.tsv` | 同じ学習データの表形式 |
| `polyimide_models/*_audit.tsv`, `review_references.json`, `tensile_modulus_film_range_removed_measurements.tsv` | 測定ごとの採否と、除外の記録 |
| `polyimide_tg_modulus/polyimide_samples.tsv`, `polyimide_measurements.tsv`, `polyimide_components.tsv`, `source_columns.tsv` | 無添加ポリイミドの試料と測定の表、抽出した列の一覧 |

## 学習データ

| 対象 | ターゲット名 | 行数 | 10分割検証 R² | MAE |
|---|---|---:|---:|---:|
| Tg | `Tg_film` | 973 | 0.75 | 23 ℃ |
| 引張弾性率 | `tensile_modulus_film_clean` | 516 | 0.62 | 0.47 GPa |
| CTE | `CTE_film` | 138 | 0.63 | 12 ppm/℃ |

評価は、反復単位を共有する構造を同じ分割に入れる構造ファミリー単位の分割です。

- **弾性率の除外:** 0.01 GPa以下または10 GPa以上、引張試験でない測定、架橋体、応力とひずみの矛盾、単位異常の疑い、要確認文献の低弾性値。測定ごとの判定は `tensile_modulus_film_audit.tsv` にあります。
- **CTEの除外:** `CTE_film_audit.tsv` に全件の採否と理由があります。
- **CTEの温度:** 元データの大半に測定温度の記載がなく、厳密な室温の値ではありません。

## 酸二無水物とジアミンからp-SMILESを作る

`tci_polyimide/polyimide_psmiles.py` は学習データの復号なしで使えます。必要なのは RDKit だけです。

```bash
# 1組(PMDA + ODA)
python tci_polyimide/polyimide_psmiles.py --dianhydride 'O=C1OC(=O)c2cc3c(cc21)C(=O)OC3=O' --diamine 'Nc1ccc(Oc2ccc(N)cc2)cc1'

# 総当たり(入力は smiles 列を持つCSV/TSV)
python tci_polyimide/polyimide_psmiles.py --dianhydrides a.csv --diamines b.csv --output pi.csv
```

両方のモノマーが非対称だと位置異性体が2つできるので、2本目(CSVでは `alt_psmiles`)も出力します。入力に `polyimide_ok=False` の行があると既定では飛ばします。含めるには `--include-flagged` を付けます。TCIモノマーの全組み合わせは `tci_polyimide/output/tci_polyimide_psmiles.csv` にあります。

## 多目的ベイズ最適化で共重合組成を探す

`polyimide_optimizer/mobo_copolyimide.py` は、酸二無水物とジアミンの種類と組成を、遺伝的アルゴリズム(NSGA-II)と期待超体積改善(EHVI)で探します。酸側とジアミン側のmol比の和はそれぞれ厳密に1.0です。`run_train.sh` で作るモデルと、復号した学習データが必要です。

```bash
# 酸2種、ジアミン2種。Tg 300 ℃以上で Tg最大、弾性率最大、CTE最小
python polyimide_optimizer/mobo_copolyimide.py --n-acids 2 --n-diamines 2 --constraint 'Tg>=300'

# PMDAを0.5に固定し、ODAは種類だけ固定。残りの種類と比率を最適化
python polyimide_optimizer/mobo_copolyimide.py --n-acids 2 --n-diamines 2 --fix-acid 'PMDA=0.5' --fix-diamine ODA

# 候補を絞る。比率の範囲指定もできる
python polyimide_optimizer/mobo_copolyimide.py --n-acids 2 --acid-candidates 'PMDA;BPDA;6FDA;ODPA' --fix-acid 'PMDA=0.2-0.6'

# 実測値を反映して次の実験を提案(列: acids, diamines, Tg, modulus, CTE。例 PMDA=0.5;BPDA=0.5)
python polyimide_optimizer/mobo_copolyimide.py --observations measured.csv
```

**探索候補の指定:** 2つの方法があります。

- **実行ごとに指定:** `--acid-candidates 'PMDA;s-BPDA;6FDA' --diamine-candidates 'ODA;p-PDA;TFMB'`
- **リストを編集:** `polyimide_optimizer/acid_candidates.csv` と `diamine_candidates.csv` の `use` 列を `False` にした行は候補から外れます。行を追加して新しいモノマーも入れられます(`smiles` 列は必須)。

名前は略称、TCI名、製品番号、CAS番号、SMILESのどれでも指定できます。略称は一般的な用法に基づくもので、定着した略称がないモノマーは正式名のままです。

**種類数と比率の指定:**

| 指定 | 動き |
|---|---|
| `--n-diamines 3` | 必ず3種類使う。候補が3種類なら比率だけを最適化し、どれも0にならない |
| `--min-diamines 1 --max-diamines 3` | 1〜3種類から選ぶ。選ばれないものは実質0(既定は1〜2種類) |
| `--min-fraction 0.1`, `--step 0.05` | 成分の最小比率と刻み。酸とジアミンの両方に効く(既定は最小比率0.1、刻み0.05) |
| `--fix-diamine ODA` | 種類だけ固定。比率は最適化する |
| `--fix-diamine 'ODA=0.5'`, `--fix-diamine 'ODA=0.2-0.6'` | 比率を固定、または範囲で指定 |

酸二無水物も `--n-acids`, `--max-acids`, `--fix-acid` で同様に指定します。固定した成分は、候補リストになくても使えます。

略称がないモノマーは、TCI名、製品番号、CAS番号、SMILESで指定します。`'` を含む名前はダブルクォートで囲みます(例 `--fix-diamine "4,4'-Diaminobenzophenone"`)。

| 出力 | 内容 |
|---|---|
| `output/pareto_predicted.csv` | 予測平均でのパレート解 |
| `output/next_experiments.csv` | 不確かさも考慮した次の実験候補 |

- **反復単位:** 酸 x_i、ジアミン y_j のとき、反復単位 A_iD_j の比率を x_i y_j とします(ランダム共重合を仮定)。
- **外挿:** 学習データの共重合体は反復単位が3種類までです。4種類以上は `composition_extrapolated=True` とし、混合則との差を不確かさに加えます。
- **不確かさの限界:** 弾性率とCTEは、学習データからの距離で不確かさがほとんど変わりません。
- **確認:** `python polyimide_optimizer/test_mobo.py` で組成の規則を検査できます。
- **実測との比較:** EHVIは実測済みポリマーのパレート前線を基準に計算します(`--front predicted` で予測の前線)。実測済みとほぼ同じ組成(`--remeasure-tolerance`, 既定0.1)は候補から外します。出力の `novelty` 列は、実測済み / 既知の系 / 既知の反復単位 / 新しい反復単位 の区別です。`measured_front.csv` に実測の前線を書き出します。

### 目標領域に入る組成を探す

`region_search.py` は、指定した物性の範囲にすべて入る確率 P(領域) を最大にし、同時に最も近い学習データとの類似度を下げる組成を探します(類似度の上限は `--max-similarity`, 既定0.8)。次の実験候補は、候補どうしの予測誤差の相関を考えて「どれか1つが目標に入る確率」が高くなるように選びます。

```bash
python polyimide_optimizer/region_search.py --region 'Tg>=350' --region 'modulus<=3'
python polyimide_optimizer/region_search.py --region 'Tg=240-260' --region 'modulus<=1.0' \
  --fix-diamine 'MESi3200=0.05-0.3' --observations polyimide_optimizer/examples/observations_siloxane_example.csv \
  --output-dir polyimide_optimizer/output_region_siloxane
```

出力は `region_candidates.csv`(P(領域)と類似度のパレート解)、`region_batch.csv`(次の実験候補)、`run_summary.json` です。

- **ソフトセグメント:** `soft_monomers.csv` のモノマー(例 Mn 3200 のシロキサンジアミン MESi3200)は学習データの範囲外なので、モデルに直接は聞きません。ソフト成分を除いたハード部分の予測に「傾き × ソフト成分の重量分率」を足し、傾きは `--observations` の実測から推定します。実測がないと傾きは事前分布のままで、予測は当てになりません。

### 結果を学習データと比べる

```bash
python polyimide_optimizer/compare_with_training.py --run-dir polyimide_optimizer/output
```

`<run-dir>/compare/index.html` をブラウザで開きます。設計と学習データ(実測値)を1枚の散布図(Tg・弾性率・CTEの組み合わせか構造マップ)に重ね、点を選ぶと予測値、モノマー、反復単位、構造の近い学習データが出ます。学習データのモノマーは反復単位をイミド結合で切って推定します。ページには実測値と文献番号が入るので、外部に共有しないでください。

## 学習データを元データから作り直す

元データ `data_20240322.tsv`(430 MB)は含みません。このフォルダの直下に置いて、次の順に実行します。

```bash
cd polyimide_models
python prepare_data.py
python prepare_film_data.py
python prepare_film_data.py --target Tg
python apply_modulus_floor.py
python prepare_film_modulus_range.py
python audit_film_modulus.py
python prepare_film_modulus_clean.py
python prepare_film_cte.py
```

途中で `Tg` などのフォルダが必要になる場合があるので、先に `mkdir -p Tg tensile_modulus Tg_film tensile_modulus_film` を実行してください。
`evaluate_cleansing.py` は除外ルールの比較に使ったスクリプトです。

`polyimide_tg_modulus/extract_polyimide.py` は元データから試料と測定の表を作ったスクリプトで、記録のために同梱しています。パッケージの検証では実行していません。

## 注意

- **モノマーの範囲:** TCIがPubChemに登録した物質が対象で、未登録品は漏れます。追加するには `tci_polyimide/supplemental_products.csv` に製品番号とCASを書いて `tci_monomer_list.py` を再実行します。
- **販売状況:** 現在の販売は確認していません。
- **予測の範囲:** 構造だけからの予測で、分子量、配向、イミド化条件は考慮していません。類似度が低い構造は外挿です。
- **容量:** アプリを構築すると、構造式SVGが約300 MB作られます。
