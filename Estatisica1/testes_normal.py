#!/usr/bin/python3
from scipy import stats
from scipy.stats import norm
import matplotlib.pyplot as plt

dados = norm.rvs(size=1000)
stats.probplot(dados, plot=plt)
plt.show()