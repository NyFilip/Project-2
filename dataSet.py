import os
import numpy as np


def minist(filePath):
    imagesList = []
    imagesMatrix = []
    labels = []
    full = []
    with open(filePath, 'r') as file:
        for line in file:
            parts = line.strip().split()
            
            index = parts[0].strip()
            label = np.array(int(parts[1]))
            fullTemp  = np.array(list(map(float,parts[1:])))
            pixels = list(map(float,parts[2:]))
            
            pixelArray = np.array(pixels).reshape(16,16)
            normArray = (pixelArray + 1) / 2
              
            
            labels.append(label)
            imagesMatrix.append(normArray)
            full.append(fullTemp)
            imagesList.append(np.array(normArray).reshape(1,256))
    
    full = np.array(full)
    labels = np.array(labels)
    imagesMatrix = np.array(imagesMatrix)
    imagesList = np.array(imagesList)                    
    return full, labels, imagesMatrix, imagesList

            

if __name__ == '__main__':
    mFull, mLabel, mImageM, mImageL = minist('Numbers.txt')
    print(f'{mFull.shape} \n {mLabel.shape} \n {mImageM.shape} \n {mImageL.shape}')
