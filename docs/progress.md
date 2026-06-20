# 学習進捗ログ

書籍「ゼロから作るDeep Learning ③」の実装記録。

## 完了済み

### step01–02: Variable と Function の基礎
- `Variable` クラス：numpy配列のラッパー。`.data` を保持
- `Function` クラス：`__call__` で入力を受け取り `forward()` に委譲する基底クラス

### step03–04: Square と合成関数
- `Square` クラス：y = x² の実装
- 関数の合成（`C(B(A(x)))` のように連結できる）

### step05: Exp の追加
- `Exp` クラス：y = exp(x) の実装
- `numeric_diff`：中心差分による数値微分（逆伝播の検証用）

### step06–10: 手動逆伝播（誤差逆伝搬）
- `Variable` に `.grad` を追加
- `Function` に `backward()` を追加（各関数が解析的な微分を返す）
- `Square.backward`：gx = 2x * gy
- `Exp.backward`：gx = exp(x) * gy
- 手動で逆順に `backward()` を呼ぶことで勾配を伝播

**現在の逆伝播の呼び方（手動）：**
```python
y.grad = np.array(1.0)
b.grad = C.backward(y.grad)
a.grad = B.backward(b.grad)
x.grad = A.backward(a.grad)
```

---

## 今後の予定

### step11+: 計算グラフによる自動逆伝播
- `Function.__call__` で入出力の接続を記録
- `Variable.backward()` を呼ぶだけで自動的に勾配が伝播するようにする

### その先
- 高階微分
- GPU対応
- RNN / LSTM
- （④巻）Transformer / Attention
