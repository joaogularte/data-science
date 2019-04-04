import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

base = pd.read_csv('../Dados/cars.csv')
#Deleta uma coluna da base
base = base.drop(['Unnamed: 0'], axis = 1)


x = base.iloc[:, 1].values
x = x.reshape(-1, 1)
y = base.iloc[:, 0].values
#Cria a correlação entre x e y
#correlacao = np.corrcoef(x, y)
#print(correlacao)

#Classe utilizada por fazer a regressão linear
modelo = LinearRegression()
modelo.fit(x, y)
print("Intercipação entre x e y: ")
print(modelo.intercept_)
print("Inclinação: ")
print(modelo.coef_)

plt.scatter(x, y)
plt.plot(x, modelo.predict(x), color='red')
plt.show()

#Distancia 22 pés
print(modelo.intercept_ + modelo.coef_ *22)
print(modelo.predict([[22]]))



