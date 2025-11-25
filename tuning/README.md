# ft-llm-2026-tuning

このリポジトリは，2026年のチューニングコンペティションに向けたチューニングのコードを格納しています．

今回のコンペティションモデルである llm-jp-4-8b を対象とした Supervised Fine-tuning (SFT) および Direct Preference Optimization (DPO) のコードが含まれています．

**注意**
llm-jp-4-8b は一般的な LLaMA 形式のモデルであるため，transformers や trl など他ライブラリを使用してチューニングを行うことも可能です


## 準備

1. 以下のコマンドでチューニング用の環境を構築します．ここで環境を構築したディレクトリ（`ENV_DIR`）は後の手順で使用します．

```bash
cd installers/abci
bash run_setup.sh $ENV_DIR
cd ..
```

2. `scripts/abci/common/setup.sh` 内の `ENV_DIR` を上記で構築した環境のディレクトリに置き換えてください．
   - `FIXME` と記載されている箇所です

3. 共有の google drive からチューニング用データセットをダウンロードし，`datasets/` に配置してください．
   - `tuning_data20251105.zip` をダウンロード・解凍し， `datasets/` に配置してください．
     - [llm-jp/extraction-wiki-ja](https://huggingface.co/datasets/llm-jp/extraction-wiki-ja) をサンプルデータとして入れています 

4. `configs/base_template.yaml` をコピーして `configs/base.yaml` を作成し，`FIXME` と記載されている箇所を修正してください．
   - 3 でダウンロードしたデータセットを使用する場合は `data_dir` と `data_version` は修正不要です．
   - `entity` は wandb のアカウント名に修正してください．
   - `project` は wandb のプロジェクト名に修正してください．
   - `work_dir` は実験結果の保存先に修正してください．モデルのチェックポイントが保存されるため，十分なディスク容量がある場所（`/home` 以下は非推奨）を指定してください．

## チェックポイント変換

### Hugging Face -> Nemo

学習前に， Hugging Face チェックポイントを Nemo チェックポイントに変換する必要があります．

```bash
# abci
bash scripts/abci/converter/run_hf_to_nemo_llama.sh ${INPUT_HF_PATH} ${OUTPUT_NEMO_PATH} ${HPARAMS_FILE}
```

`${HPARAMS_FILE}` には `megatron_configs/llmjp4/8b.yaml` を指定してください．

変換後の NEMO チェックポイントには設定が記載された `model_config.yaml` ファイルと重みが保存された `model_weights` ディレクトリが含まれます．

### Nemo -> Hugging Face

学習後は， Nemo チェックポイントを Hugging Face チェックポイントに変換する必要があります．
**注意**: パスは絶対パスで指定してください．

```bash
bash scripts/abci/converter/run_nemo_to_hf_llama.sh ${INPUT_NEMO_PATH} ${OUTPUT_HF_PATH}
```

## Supervised Fine-tuning

```bash
bash scripts/abci/train/run_sft.sh llmjp4_8b ${NUM_NODES} ${INPUT_NEMO_PATH}
```

`{NUM_NODES}` は学習に使用するノード数を指定してください．
`${INPUT_NEMO_PATH}` は Nemo チェックポイントのパスを指定してください．

**注意:**
この学習スクリプトでは，データをまずトークナイズし，キャッシュとして保存します．キャッシュは各 jsonl ファイルごとに作成され，pkl ファイルとして保存されます（ファイル名は同名）．
キャッシュが存在する場合は，トークナイズ処理はスキップされ，キャッシュが直接読み込まれるため，データセットを変更した場合はキャッシュを削除してください．
処理の詳細については `sft_dataset.py` を参照してください．

**注意:**
トークナイズするデータが大量にある場合，キャッシュの作成時にエラーを起こす場合があります．その場合は再実行を試みてください．

**注意:**
データセットを変更する場合，`configs/sft.yaml` の `datasets` セクションを適宜修正してください．

## Direct Preference Optimization

```bash
bash scripts/abci/train/run_dpo.sh llmjp4_8b ${NUM_NODES} ${INPUT_NEMO_PATH}
```
`{NUM_NODES}` は学習に使用するノード数を指定してください．
`${INPUT_NEMO_PATH}` は Nemo チェックポイントのパスを指定してください．

**注意:**
この学習スクリプトでは，データをまずトークナイズし，キャッシュとして保存します．キャッシュは各 jsonl ファイルごとに作成され，pkl ファイルとして保存されます（ファイル名は同名）．
キャッシュが存在する場合は，トークナイズ処理はスキップされ，キャッシュが直接読み込まれるため，データセットを変更した場合はキャッシュを削除してください．
処理の詳細については `dpo_dataset.py` を参照してください．


**注意:**
DPO 学習を Supervised Fine-tuning 直後の Nemo チェックポイントに対して直接実行する場合，DPO 実行前に以下のコマンドを実行する必要があります（対応する Nemo チェックポイントを既に HF 形式に変換している場合はこのステップは不要です）:

```bash
ln -s $(ls -d ${INPUT_NEMO_PATH}/step=*-last) ${INPUT_NEMO_PATH}/model_weights
```

**注意:**
データセットを変更する場合，`configs/dpo.yaml` の `datasets` セクションを適宜修正してください．