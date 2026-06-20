# 設計メモ・概念ノート

「なぜこう設計するのか」の記録。実装の背景にある意図をまとめる。

---

## なぜ Function を基底クラスにするのか

`forward()` と `backward()` を対称に保つため。
`__call__` に副作用（入力の保存）を集約することで、すべての演算が自動的に逆伝播の準備をする。

```python
def __call__(self, input):
    x = input.data
    y = self.forward(x)
    output = Variable(y)
    self.input = input   # ← ここが重要：backward で使うために覚えておく
    return output
```

---

## なぜ数値微分と解析微分を両方実装するのか

数値微分（`numeric_diff`）は実装が正しいかの検証用。
本番では使わない（計算コストが O(N) のforward pass 必要）。
解析微分（`backward()`）は連鎖律を使って O(1) の backward pass で全勾配を計算できる。

---

## バックプロパゲーションが速い理由

連鎖律により、中間勾配を一度計算したら再利用できる（動的計画法と同じ発想）。

数値微分：パラメータが N 個あれば N 回の forward pass が必要 → O(N)
backprop：backward pass 1回で全パラメータの勾配を計算 → O(1)

大規模モデル（GPT等）では N が数十億になるため、この差が致命的に重要。

---

## 手動逆伝播の限界

現在は逆順に `backward()` を手で呼ぶ必要がある：

```python
y.grad = np.array(1.0)
b.grad = C.backward(y.grad)   # 手動
a.grad = B.backward(b.grad)   # 手動
x.grad = A.backward(a.grad)   # 手動
```

問題：計算グラフが複雑になると手動管理は破綻する。
解決：`Function.__call__` で入出力の接続をグラフとして記録し、`Variable.backward()` を呼ぶだけで自動伝播させる（step11+）。

---

## Variable が numpy配列を直接持たずラップする理由

勾配（`.grad`）や計算グラフの接続情報（将来の `.creator`）を付随させるため。
numpy配列そのものにメタ情報を持たせることはできないが、ラッパーにすれば自由に属性を追加できる。
