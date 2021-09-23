# Step06 updated 2021.09.21
# Step05 updated 2021.09.21
import numpy as np

class Variable:
    def __init__(self,data):
        self.data =data
        self.grad =None     # step06 added

# Functionクラス定義
class Function:
    # __call__メソッドは、クラスインスタンスを関数呼び出しのようにできるように
    # するための特殊メソッド。特殊メソッドは、クラスインスタンスの振る舞いを
    # 調整、拡張するためのメソッド
    def __call__(self, input): 
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        self.input = input      # remember input Variable
        return output

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):     # step06
        raise NotImplementedError()

class Square(Function):     # Function classを継承
    def forward(self, x):   # forwardメソッドをオーバーライド
        y = x**2
        return y

    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx

class Exp(Function):        # Function classを継承
    def forward(self, x):   # forwardメソッドをオーバーライド
        y = np.exp(x)
        return y

    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx

# ===========================================
# 数値微分
def numeric_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2*eps)

def f(x):
    A = Square()
    B = Exp()
    C = Square()
    return C(B(A(x)))

# ===========================================
x = Variable(np.array(0.5))
dy = numeric_diff(f, x)
print("numeric_diff dy = " + str(dy))

# ============================================
# 誤差逆伝搬
A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)

y.grad = np.array(1.0)
b.grad = C.backward(y.grad)
a.grad = B.backward(b.grad)
x.grad = A.backward(a.grad)
print(x.grad)