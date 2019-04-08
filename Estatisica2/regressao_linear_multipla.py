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
print('Intercepção: ')
print(modelo.intercept_)

print("Inclinação: ")
print(modelo.coef_)

print("Coeficiente de determinação: ")
print(modelo.score(x, y))

#print("Coeficiente de determinação ajustado: ")

previsoes = modelo.predict(x)
print(previsoes)