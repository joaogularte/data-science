import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

base = pd.read_csv("../Dados/Eleicao.csv", sep=";")

print("Informações estatisticas sobre a base: ")
print(base.describe())
print("")

correlacao = np.corrcoef(base.DESPESAS, base.SITUACAO)
print("Correlação entre DESPESAS e SITUACAO: ")
print(correlacao)

x = base.iloc[:, 2].values
y = base.iloc[:, 1].values
x = x[:, np.newaxis]

modelo = LogisticRegression()
modelo.fit(x, y)

print("Valor de inclinação: ")
print(modelo.coef_)
print("Valor de intercepção: ")
print(modelo.intercept_)

X_teste = np.linspace(10, 3000, 100)

def model(x):
    return 1 / (1 + np.exp(-x))

r = model(X_teste * modelo.coef_ + modelo.intercept_).ravel()


plt.scatter(x, y)
plt.plot(X_teste, r, color = "red")
plt.show()