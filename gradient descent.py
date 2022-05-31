#REFERENCE: tistory블로그 IT끄적이기 Python으로 경사하강법(Gradient Descent) 구현
#위의 블로그의 코드 임을 밝힙니다.

#업데이트 할 W: learning rate*{(Y예측-Y실체)*X}평균
#업데이트 할 b: learning rate*{(Y예측-Y실체)*1}평균

import numpy as np
import matplotlib.pyplot as plt

X = np.random.rand(100)
Y = 0.2 * X * 0.5

plt.figure(figsize=(8, 6)) #figure 명령어: 
plt.scatter(X, Y) #scatter 명령어: 
plt.show() 

def plot_prediction(pred, y):  
    plt.figure(figsize=(8, 6))
    plt.scatter(X, y)
    plt.scatter(X, pred)
    plt.show

## Gradient Descent 구현 구간
W = np.random.uniform(-1, 1)
b = np.random.uniform(-1, 1)

learning_rate = 0.7

for epoch in range(100):
    Y_Pred = W * X + b

    error = np.abs(Y_Pred - Y).mean()
    if error < 0.001:
        break
    #gradient descent 
    w_grad = learning_rate * ((Y_Pred - Y)*X).mean()
    b_grad = learning_rate * ((Y_Pred - Y)).mean()

    #W, b값 갱신 
    W = W - w_grad
    b = b - b_grad

    if epoch % 5 == 0:
        Y_Pred = W * X + b
        plot_prediction(Y_Pred, Y)
