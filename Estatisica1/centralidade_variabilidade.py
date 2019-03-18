#!/usr/bin/python3
import numpy as np
from scipy import stats

jogadores = [12000, 18000, 4000, 250000, 30000, 140000, 3000000, 4000000, 8000000]

print("Media: " , np.mean(jogadores))
print("Mediana: ", np.median(jogadores))

print("Desvio padrao", np.std(jogadores, ddof=1))

print(stats.describe(jogadores))