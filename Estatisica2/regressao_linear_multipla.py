import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as sm

base = pd.read_csv('../Dados/mt_cars.csv')
base = base.drop(['Unnamed: 0'], axis = 1)
x = base.iloc[:, 2].values
y = base.iloc[:, 0].values
correlacao = np.corrcoef(x, y)
print("Correlacao entre x e y: ")
print(correlacao)

x = x.reshape(-1, 1)

modelo = LinearRegression()
modelo.fit(x, y)
print('Valor de Intercepção: ')
print(modelo.intercept_)

print("Coeficiente de Inclinação: ")
print(modelo.coef_)

print("Coeficiente de determinação: ")
print(modelo.score(x, y))

previsoes = modelo.predict(x)

print("Coeficiente de determinação ajustado: ")
modelo_ajustado = sm.ols(formula = 'mpg ~ disp', data = base)
modelo_treinado = modelo_ajustado.fit()
print(modelo_treinado.summary())

plt.scatter(x, y)
plt.plot(x, previsoes, color = "red")
plt.show()

x1 = base.iloc[:, 1:4].values
modelo.fit(x1, y)

print("Coeficiente de determinação: ")
print(modelo.score(x1, y))

print("Coeficiente de determinação ajustado: ")
modelo_ajustado2 = sm.ols(formula = 'mpg ~ cyl + disp + hp', data = base)
modelo_treinado2 = modelo_ajustado2.fit()
print(modelo_treinado2.summary())

novo = np.array([4, 100, 200])
novo = novo.reshape(1, -1)
print(modelo.predict(novo))
