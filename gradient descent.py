#REFERENCE: tistory블로그 IT끄적이기 Python으로 경사하강법(Gradient Descent) 구현
#위의 블로그의 코드 임을 밝힙니다.

#업데이트 할 W: learning rate*{(Y예측-Y실체)*X}평균
#업데이트 할 b: learning rate*{(Y예측-Y실체)*1}평균

import numpy as np
import matplotlib.pyplot as plt

X = np.random.rand(100) #rand: 0부터 100까지의 난수 설정
Y = 0.2 * X * 0.5

plt.figure(figsize=(8, 6)) #figure: matplotlib로 그래프를 그리려면 figure라는 객체가 필요함. Axes객체도 필요.
plt.scatter(X, Y) #scatter: (X, Y)에 기본 마커가 표시됨. 산점도를 그리는데 사용한다.
plt.show()  #show: 그래프를 화면에 나타나도록 하는 함수

def plot_prediction(pred, y):  
    plt.figure(figsize=(8, 6))
    plt.scatter(X, y)
    plt.scatter(X, pred)
    plt.show

## Gradient Descent 구현 구간
W = np.random.uniform(-1, 1) #np.random.uniform()은 균등분포 함수이다. 최소값, 최대값, 데이터 개수 순서로 파라미터를 입력해줌. 여기에는 1과 -1뿐이라 그 사이의 값을 뽑아준다.
b = np.random.uniform(-1, 1) #가중치와 바이어스를 -1과 1사이의 임의의 값으로 초기화.

learning_rate = 0.7

for epoch in range(200): #epoch는 학습진행을 의미
    Y_Pred = W * X + b

    error = np.abs(Y_Pred - Y).mean() #평균 오차 정의, abs: 숫자의 절대값을 구하는 데에 사용되는 함수
    if error < 0.001: 
        break #error의 값이 0.001미만이 될 때 까지 계속 돌림.
    #gradient descent 
    w_grad = learning_rate * ((Y_Pred - Y)*X).mean()
    b_grad = learning_rate * ((Y_Pred - Y)).mean()

    #W, b값 갱신 (error의 값이 0.001이 될 때 까지)
    W = W - w_grad
    b = b - b_grad

    if epoch % 10 == 0: #epoch가 5로 나누어 떨어지게 하는 이유는 그래프의 간격을 5로하기 위함.
        Y_Pred = W * X + b
        plot_prediction(Y_Pred, Y) #plot_prediction 함수 이용
