import numpy as np
import matplotlib.pyplot as plt

catsAndDogs = np.genfromtxt('catdogdata.txt')
numbers=np.genfromtxt('Numbers.txt')
print(catsAndDogs.shape,numbers.shape)
print(catsAndDogs,numbers)