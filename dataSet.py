import os
import numpy as np


def mnist(filePath):
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
            
def catdog(filePath):
    labels = []
    labels[:98] = np.zeros(99)
    labels[99:] = np.ones(99)
    imagesMatrix = []
    imagesList = []
    full = []
    i = 0
    with open(filePath, 'r') as file:
        for line in file:
            parts = line.strip().split()
            
            index = parts[0].strip()
            pixels = list(map(int,parts[1:]))

            imagesMatrix.append(np.array(pixels).reshape(64,64))
            imagesList.append(np.array(pixels))

            fullTemp = list(pixels)
            if i < 99:
                fullTemp.insert(0, 0)
            else:
                fullTemp.insert(0,1)
            full.append(fullTemp)
            i += 1

    imagesMatrix = np.array(imagesMatrix)
    imagesList = np.array(imagesList)
    full = np.array(full)
    labels = np.array(labels)
    return full, labels, imagesMatrix, imagesList


if __name__ == '__main__':

