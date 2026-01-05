# Singularity提出用サンプル

このディレクトリは、チューニングコンペティション2026でSingularityイメージを提出するための最小構成例です。
`main.py` が評価用入出力を処理し、Singularity イメージ内では `uv` で固定された Python 依存関係が利用されます。

## 同梱物

- `main.py` : Singularityから実行される推論スクリプト
- `download_model.py` : Hugging Face Hubからモデルを取得するユーティリティ
- `sample_problems.jsonl` : 推論手順を確認するためのサンプル入力
- `submission.def` : Singularity定義ファイル
- `pyproject.toml`, `uv.lock` : Python 依存関係定義

## 前提条件

- Singularity (ABCIでは `singularity-ce version 4.1.5-1.el9` が利用可能)
- [uv](https://docs.astral.sh/uv/)

## 1. モデルの準備

この例では `llm-jp/llm-jp-3.1-1.8b-instruct4` を使用します。
以下を実行すると `models/llm-jp/llm-jp-3.1-1.8b-instruct4` にダウンロードされます。

```bash
uv run python download_model.py --model_name llm-jp/llm-jp-3.1-1.8b-instruct4
```

## 2. Singularityイメージのビルド

1. `uv` のキャッシュディレクトリを環境変数に保持します。
   ```bash
   export UV_CACHE_DIR="$(uv cache dir)"
   ```
2. 含めたいモデルのリストを空白区切りで指定し、Singularityイメージをビルドします。
   この例では前節でダウンロードしたモデルのみを含めています。
   ```bash
   singularity build --fakeroot --force \
       --bind "${UV_CACHE_DIR}:/root/.cache/uv" \
       --build-arg MODEL_NAMES="llm-jp/llm-jp-3.1-1.8b-instruct4" \
       dist/submission.sif submission.def
   ```
   - 複数モデルを含めたい場合は `MODEL_NAMES` を空白区切りで列挙してください。
   - `--bind` によってビルド時の Python 依存関係取得が高速化されます。

## 3. ローカルでの動作確認

生成したイメージを使用してサンプル入力で推論を行います。GPUを利用するコマンド例は以下の通りです。

```bash
singularity run --nv --writable-tmpfs \
    --env CUDA_VISIBLE_DEVICES=0 --net --network none \
    dist/submission.sif \
    --input_path sample_problems.jsonl \
    --output_path "$(pwd)/output.jsonl"
```

推論結果は `output.jsonl` に書き出されます。

## 応用

- 推論ロジックを変更する場合は `main.py` を編集してください。
  引数を追加した場合は `singularity run` の呼び出しにも反映させてください。
- Python依存関係を変更する際は `pyproject.toml` を編集し、`uv lock` を実行して
  `uv.lock` を更新します。その後あらためてイメージを再ビルドしてください。
- OS 依存パッケージが必要になった場合は `submission.def` の `%post` セクションに
  `apt-get install` 等を追記します。イメージサイズに注意してください。

このサンプルをベースに、各チームの推論コードやモデルを組み込んだ提出イメージを作成してください。
