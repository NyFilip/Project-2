import numpy as np

def FeedForward(data,classiferFunction,errorThreshold,maxNumberOfFeatures):
    
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

def KNearestNeighboors(trainingSet, testSet, k, norm=2):
    testSetLabels = []

    for index,image in enumerate(testSet):
        
        distances = np.linalg.norm(trainingSet[:,1:]-image[1:],axis=1,ord=norm)

        neighbors = np.argsort(distances)[:k]
        
        labels = trainingSet[neighbors, 0]  

        labels =list(labels)

        label = max(labels, key=labels.count)

        testSetLabels.append(label)
    
    return testSetLabels
def Accuracy(predictedLabels,testSetLabels):
    
    correct = 0
    for i in range(len(testSetLabels)):
        if predictedLabels[i] == testSetLabels[i]:
            correct += 1
    accuracy = correct / len(testSetLabels) * 100
    return accuracy


