#!/usr/bin/python3
from scipy.stats import binom

#binom.pmf(qnt sucessos, qnt todos as opções, probabilidade)

#Jogar uma moeda 5 vezes, qual a probabilidade de dar cara 3 vezes?

prob = binom.pmf(3, 5, 0.5)
print(prob)

#Passar por 4 sinais de 4 tempos, qual a probabilidade de pegar sinal verde, nenhuma, 1, 2, 3 ou 4 vezes seguidas?

prob = binom.pmf(2, 4, 0.25)
print(prob)

#probabilidade acumulativa
prob = binom.cdf(4, 4, 0.25)
print(prob)

#Concurso com 12 questões, qual a probabilidade de acertar 7 questões considerando que cada questão tem 4 alternativas?
prob = binom.pmf(7, 12, 0.25)
print(prob)