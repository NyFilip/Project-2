import natanaelFunction as na
import dataSet as ds
import matplotlib.pyplot as plt
import numpy as np


mFull, mLabels, mMatrix, mList = ds.mnist()
cdFull, cdLAbels, cdMatrix, cdList = ds.catdog()

print(f'{mFull.shape} \n {cdFull.shape}')




if __name__ == '__main__':
    ds.visCatDog(cdMatrix)
    ds.visMnist(mMatrix)
