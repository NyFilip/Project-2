import numpy as np
import nilsFunction as nf
catsAndDogs = np.loadtxt('catdogdata.txt')
catsAndDogs = catsAndDogs[:, 1:]  
cats = catsAndDogs[:6, :]
print(cats)
numbers=np.loadtxt('Numbers.txt')
kurt = nf.display_images_from_rows(cats)
