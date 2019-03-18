#!/usr/bin/python3
import pandas as pd
from sklearn.model_selection import train_test_split

iris = pd.read_csv('../Dados/iris.csv')
x, _, y, _ = train_test_split(iris.iloc[:, 0:3], iris.iloc[:, 4], test_size=0.5, stratify=iris.iloc[:, 4])
print(y)

infert = pd.read_csv('../Dados/infert.csv')
x1, _, y1, _ = train_test_split(infert.iloc[:, 2:9], infert.iloc[:, 1], test_size=0.6, stratify=infert.iloc[:, 1])
print(y1)

