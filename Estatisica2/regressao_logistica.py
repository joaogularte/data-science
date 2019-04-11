import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

base = pd.read_csv("../Dados/Eleicao.csv", sep=";")
#plt.scatter(base.DESPESAS, base.SITUACAO)

print("Informações estatisticas sobre a base: ")
print(base.describe())

print("Correlação entre DESPESAS e SITUACAO: ")
np.corrcoef(base.DESPESAS, base.SITUACAO)

x = base.iloc[:, 2].values
y = base.iloc[:, 1].values
x = x[:, np.newaxis]
print(x)