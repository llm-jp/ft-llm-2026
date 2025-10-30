# math-eval

`math-eval` は数学タスクの評価ツールです．
本ツールは　[Math-Verify](https://github.com/huggingface/Math-Verify) を使用して回答の抽出，正規化，一致判定を行います．

### インストール

`uv` を[公式のガイドライン](https://docs.astral.sh/uv/getting-started/installation/)に従ってインストールしてください．

### 使い方

以下のコマンドを実行してください：

```bash
uvx --from "git+https://github.com/llm-jp/ft-llm-2026#subdirectory=math-eval" math-eval <path/to/predictions.jsonl> <path/to/gold.jsonl> [-o <path/to/output.json>]
```

`<path/to/predictions.jsonl>` はモデルの予測を含む JSONL ファイルへのパス，`<path/to/gold.jsonl>` は正解を含む JSONL ファイルへのパスです．
`examples/` ディレクトリに例があります．
`-o` オプションでファイル名を指定すると，評価結果が JSON 形式で保存されます．
