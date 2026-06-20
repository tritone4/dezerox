# CLAUDE.md

このファイルはリポジトリ内での作業時に Claude Code (claude.ai/code) へ提供するガイダンスです。

## コマンド

スクリプトの実行：
```bash
python steps/step01-10.py
```

## アーキテクチャ

書籍「ゼロから作るDeep Learning ③ — DeZero」に沿って、自動微分フレームワークをステップごとに実装するプロジェクト。

**`steps/step01-10.py` の主要な抽象：**

- `Variable` — numpy配列のラッパー。`.data`（値）と `.grad`（勾配）を保持
- `Function` — 微分可能な演算の基底クラス。`__call__` が逆伝播のために入力を保存し `forward()` に委譲。サブクラスは `forward()` と `backward()` を実装する
- `Square`, `Exp` — `Function` の具体サブクラス。y=x² および y=eˣ を解析的な微分とともに実装
- `numeric_diff` — 中心差分による数値微分。逆伝播の正しさを検証するために使用

**逆伝播のパターン：** 現在は手動。各関数の `backward()` を逆順に明示的に呼ぶことで勾配を伝播させている。今後のステップで計算グラフを使った自動化を行う。

## 進捗

- 実装済み：step01–10（Variable、Function、Square、Exp、手動逆伝播、数値微分）
- 次のステップ：step11+（計算グラフによる自動逆伝播）

## 回答スタイル

- 実装の「なぜ」を省略しない
- 数式はLaTeXではなくプレーンテキストで（例：dy/dx、x**2）
- 日本語で説明して構わない
