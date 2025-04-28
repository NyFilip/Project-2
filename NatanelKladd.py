import natanaelFunction as na
import dataSet as ds

mFull, mLabels, mMatrix, mList = ds.mnist()
cdFull, cdLAbels, cdMatrix, cdList = ds.catdog()

print(f'{mFull.shape} \n {cdFull.shape}')
