#!/usr/bin/python3
import pandas as pd
import numpy as np

#Seta a semente geradora
np.random.seed(2345)

#Lê um arquivo tipo csv
base = pd.read_csv('../Dados/iris.csv')
#Gera uma amostra do tipo simples
amostra = np.random.choice([0, 1], 150, True, [0.5, 0.5])

count = 0
selecionados = []
for i in amostra:
    if(i == 1):
        selecionados.append(count)
    count = count + 1

print(amostra)
print(base.loc[selecionados])