import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


df = pd.read_csv('Salary_Data.csv')

df.head() #데이터 프레임의 앞부분을 확인한다.

df.isna().sum() #Nan데이터가 있는지 확인한다.

X = df.iloc[:,0]

y = df['Salary']

X.head()

y.head()

from sklearn.model_selection import train_test_split # 테스트와 트레이닝셋을 나눈다.

X_train, X_test, y_train, y_test =train_test_split(X,y, test_size = 0.2, random_state = 5)



from sklearn.linear_model import LinearRegression  

regressor = LinearRegression()



X_train=X_train.values

X_train

X_train = X_train.reshape(-1,1)          # 학습을 위해 2차원으로 변경

X_train

#### 학습 시 사용하는 함수 fit

regressor.fit(X_train,y_train)



X_test = X_test.values.reshape(-1,1)



y_pred = regressor.predict(X_test)

y_test

y_test = y_test.values

y_test - y_pred 



plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_test, y_pred, color = 'blue')
plt.show()