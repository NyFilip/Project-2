import numpy as np
import matplotlib.pyplot as plt

def FeedFoorward(data,classiferFunction,errorThreshold,maxNumberOfFeatures):
    
    maxLoops=min(maxNumberOfFeatures,data.shape[1]-1)
    dimensions=np.arange(1,data.shape[1])
    bestCVerror=10
    iterations=1
    
    errorDifferance=10
    while iterations <=maxLoops and errorDifferance>errorThreshold:

        selectedFeatures=[]
        bestfeature=0
        prevError=bestCVerror
        for feature in dimensions:

            CVerror=KFoldCrossValidation(data[[selectedFeatures,feature]],numberOfSplits=5)

            if CVerror < bestCVerror:

                bestfeature=feature
                bestCVerror=CVerror
        
        errorDifferance=prevError-bestCVerror
        
        selectedFeatures.append(bestfeature)
    return selectedFeatures

def KFoldCrossValidation(data,classifierFunction,numberOfSplits):
        
    
    splitData=np.array_split(data,numberOfSplits,axis=1)
    accuracyList=[]

    for iteration in range(numberOfSplits):
        trainingData = np.vstack(splitData[np.arange(len(splitData))!=iteration])

        testData=splitData[iteration]

        accuracy=classifierFunction(trainingData,testData)

        accuracyList.append(accuracy)
    meanError=np.mean(accuracyList)
    return meanError



import render_image
catsAndDogs = np.loadtxt('catdogdata.txt')
numbers=np.loadtxt('Numbers.txt')

print(catsAndDogs.shape,numbers.shape)
print(catsAndDogs,numbers)
print(catsAndDogs[0,1:])
render_image.display_images_from_rows(catsAndDogs[[98,99],1:])