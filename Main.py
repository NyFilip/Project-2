import numpy as np
import matplotlib.pyplot as plt

catsAndDogs = np.loadtxt('catdogdata.txt')
numbers=np.loadtxt('Numbers.txt')
print(catsAndDogs.shape,numbers.shape)
print(catsAndDogs,numbers)