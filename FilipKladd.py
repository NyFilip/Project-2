import numpy as np
import matplotlib.pyplot as plt
import dataSet
import nilsFunction
def FeedFoorward(data,classiferFunction,errorThreshold,maxNumberOfFeatures):
    
    maxLoops=min(maxNumberOfFeatures,data.shape[1]-1)
    
    dimensions=np.arange(1,data.shape[1])
    
    bestCVerror=100
    iterations=1
    
    errorDifferance=10
    selectedFeatures=[0]
    while iterations <=maxLoops and errorDifferance>errorThreshold:

        
        bestfeature=0
        prevError=bestCVerror
        for feature in dimensions:
            testfeatures=selectedFeatures+[feature]
            # print("testfeatures",data[:,testfeatures])
            CVerror=KFoldCrossValidation(data[:,testfeatures],classifierFunction=classiferFunction,numberOfSplits=5)
            print("error of feature ",feature," was ",CVerror)
            if CVerror < bestCVerror:

                bestfeature=feature
                bestCVerror=CVerror
        
        errorDifferance=prevError-bestCVerror
        print("bestfeature",bestfeature)
        dimensions=np.delete(dimensions,bestfeature-1)
        selectedFeatures.append(bestfeature)
        print("selectedFeatures",selectedFeatures)
    return selectedFeatures

def KFoldCrossValidation(data,classifierFunction,numberOfSplits):
        
    
    splitData=np.array_split(data,numberOfSplits,axis=0)
    
    accuracyList=[]

    for iteration in range(numberOfSplits):
        trainingData = np.vstack([splitData[i] for i in range(len(splitData)) if i != iteration])
        
        testData=splitData[iteration]
        
        testLabels=classifierFunction(trainingData,testData)
        accuracy=Accuracy(testLabels,testData[:,0])
        
        accuracyList.append(accuracy)
    meanAccuracy=np.mean(accuracyList)
    
    return 100-meanAccuracy

def KNearestNeighboors(trainingSet, testSet, k=4, norm=2):
    testSetLabels = []

    for index,image in enumerate(testSet):
        
        distances = np.linalg.norm(trainingSet[:,1:]-image[1:],axis=1,ord=norm)

        neighbors = np.argsort(distances)[:k]
        
        labels = trainingSet[neighbors, 0]  

        labels =list(labels)
        label = max(labels, key=labels.count)
        testSetLabels.append(label)
    



   
    # distances = np.linalg.norm(trainingSet[:,1:]-testSet[:,1:].reshape(testSet.shape[0],1,-1),axis=2,ord=norm)
    
    
    # neighbors = np.argsort(distances, axis=1)[:, :k]
    
    
    # neighborLabels = trainingSet[neighbors,0]
    
    
    # testSetLabels = [max(labels, key=list(labels).count) for labels in neighborLabels]
    # testSetLabels=[]
    # for labels in neighborLabels:
    #     unique,counts=np.unique(labels,return_counts=True)
    #     assignedLabel=unique[np.argmax(counts)]
    #     testSetLabels.append(assignedLabel)
    
    return testSetLabels
def Accuracy(predictedLabels,testSetLabels):
    # Calculate accuracy
    correct = 0
    for i in range(len(testSetLabels)):
        if predictedLabels[i] == testSetLabels[i]:
            correct += 1
    accuracy = correct / len(testSetLabels) * 100
    return accuracy


mnistDigits= dataSet.minist('numbers.txt')
numbers=mnistDigits[0]
print(numbers)
catsAndDogs = np.loadtxt('catdogdata.txt')
# numbers=np.loadtxt('Numbers.txt')

# print(catsAndDogs.shape,numbers.shape)
# print(catsAndDogs,numbers)
# print(catsAndDogs[0,1:])
# nilsFunction.display_images_from_rows(catsAndDogs[[98,99],1:])
selectedFeatures=FeedFoorward(numbers,KNearestNeighboors,0.001,10)
