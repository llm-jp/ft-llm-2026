# ft-llm-2026-midtraining
このリポジトリは，2026年のチューニングコンペティションに向けた継続事前学習用のコードを格納しています．

## インストール

継続事前学習のための環境構築については
[installer/README.md](installer/README.md) を参照してください．
ここで環境を構築したディレクトリ（`ENV_DIR`）は後の手順で使用します．


## 学習コーパスの準備

コーパスのダウンロードと前処理を行います．`ENV_DIR` を上記で構築した環境のディレクトリに置き換えてください．

```bash
cd corpus
bash run_download.sh $ENV_DIR $DATA_ROOT_DIR
bash run_tokenize.sh $ENV_DIR $DATA_ROOT_DIR
```

`$DATA_ROOT_DIR/raw` に生のデータセットが，`$DATA_ROOT_DIR/tokenized` にトークナイズされたデータセットが保存されます．
ファイル容量が 90GB を超えるため，保存先のディスク容量に注意してください．

## 継続事前学習

### huggingface -> mcore 形式への変換
事前学習では [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) を使用しますが， huggingface 形式のモデルを直接読み込むことができません．
そのため，まず huggingface 形式のモデルを mcore 形式に変換します．

```bash
cd pretrain
bash ckpt/run_hf_to_mcore.sh $ENV_DIR $HF_MODEL_PATH $MCORE_MODEL_PATH
```

`$HF_MODEL_PATH` には huggingface 形式のモデルのパスを，`$MCORE_MODEL_PATH` には変換後の mcore 形式のモデルの保存先を指定してください．
正しく変換されていれば $MCORE_MODEL_PATH に `iter_0000001` というディレクトリと `latest_checkpointed_iteration.txt` というファイルが生成されます．

### タスクディレクトリの準備
実験設定や結果の保存用にタスクディレクトリを作成します．`pretrain/tasks/openwebmath` にサンプルを格納しています．
タスクディレクトリ内には以下のディレクトリ・ファイルを準備する必要があります．

- `params.sh`: モデルハイパーパラメータ、optimizer設定、学習器の各種設定などを定義するスクリプト
  - Megatron-LM の `pretrain_gpt.py` に渡す引数をこのファイル内の変数に定義する
  - 継続事前学習の長さ（何トークン学習するか）はこのファイルを編集して設定する
- `train_data.sh`: 学習データのパス及び利用するトークン数などを定義するスクリプト
- `base_checkpoints`: 変換した mcore 形式のモデルを格納するディレクトリ
  - `base_checkpoints/iter_0000001`，`base_checkpoints/latest_checkpointed_iteration.txt` といった形で配置してあればよい
  - `base_checkpoints` がないと継続事前学習ではなく，いちからモデルを学習してしまうので注意
    - 学習実行時，モデル読み込みが成功すると学習ログに `successfully loaded checkpoint from ...` と表示されるので確認するとよい

`train_data.sh` 内の FIXME (`DATA_ROOT_DIR`) を上記で準備したトークナイズ済みデータセットのディレクトリに書き換えてください

### 学習の実行

```bash
cd pretrain
# train/qsub_train.sh 内の FIXME を適宜書き換える
bash train/run_train.sh $ENV_DIR $TASK_DIR $NUM_NODES
```

`$TASK_DIR` には上記で準備したタスクディレクトリのパスを，`$NUM_NODES` には使用するノード数を指定してください．

学習が正常に開始されると，`$TASK_DIR/logs` にログが，`$TASK_DIR/checkpoints` に学習済みモデルが保存されます．
学習済みモデルは `checkpoints/iter_XXXXXXX` というディレクトリに保存されます．
例えば以下のような形で保存されます．チェックポイントの保存頻度のアルゴリズムは[こちら](https://github.com/llm-jp/Megatron-LM/blob/3ff6479d0fc193ade5c313956a3f5ccc5682c1e7/megatron/training/training.py#L623)を参照してください．

```commandline
drwxr-x--- 2 xxx xxx 4096 Oct 16 yy:zz iter_0000100/
drwxr-x--- 2 xxx xxx 4096 Oct 16 yy:zz iter_0000200/
drwxr-x--- 2 xxx xxx 4096 Oct 16 yy:zz iter_0000300/
```

学習の進捗は `qsub_train.sh` 内で指定した wandb のプロジェクトページで確認できます．
また，何らかの理由で学習が中断した場合でも，同じコマンドを実行すれば最新のチェックポイントから学習を再開できます（上記の例だと `iter_0000300` を読み込んで学習を再開します）．

参考までに，8B パラメータのモデルを 1 ノード（8 GPU）で 1B トークン学習するのにかかる時間は約5時間です．

### mcore -> huggingface 形式への変換
学習済みモデルを huggingface 形式に変換します．これにより `transformers` ライブラリなど馴染み深い形式でモデルを利用できるようになります．

```bash
cd pretrain
bash ckpt/run_mcore_to_hf.sh $ENV_DIR $TASK_DIR $ITER_NUM 
```

`$TASK_DIR` には上記で準備したタスクディレクトリのパスを，`$ITER_NUM` には変換したいチェックポイントのイテレーション数（`500` など）を指定してください．

