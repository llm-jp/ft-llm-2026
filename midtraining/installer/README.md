# Installer for pretraining LLM-jp models on ABCI

## Usage

ABCI 3.0上で以下のコマンドを実行すると `<env_dir>` に環境を構築できる．
*実行前に `qsub_setup.sh` の `FIXME` 部分を適宜書き換えること*

```bash
cd installer
bash run_setup.sh <env_dir>
```

環境構築後，以下のコマンドで環境を有効化できる

```bash
source <env_dir>/scripts/environment.sh  # Load environment variables and modules
source <env_dir>/venv/bin/activate       # Activate Python virtual environment
```

各種ライブラリのバージョンに関しては `scripts/environment.sh` を参照
