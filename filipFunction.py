import numpy as np
from sklearn.feature_selection import f_classif

def custom_feed_forward(X, y, classifier, max_features=None, kfoldCV=None):
    """
    Custom feed-forward feature selection.
    
    Parameters:
        X : np.ndarray
            Feature matrix.
        y : np.ndarray
            Labels.
        classifier : function
            Classifier function to evaluate features.
        max_features : int
            Maximum number of features to select.
        kfoldCV : function
            Cross-validation function to evaluate accuracy.
    
    Returns:
        selected_features : list
            Indices of selected features.
        accuracies : list
            Accuracy for each feature added.
    """
    if max_features is None:
        max_features = X.shape[1]

    selected_features = []
    remaining_features = list(range(X.shape[1]))
    accuracies = []

    for _ in range(max_features):
        best_feature = None
        best_accuracy = 0

        for feature in remaining_features:
            # Test the current feature set
            current_features = selected_features + [feature]
            X_subset = X[:, current_features]
            accuracy = kfoldCV(data=np.column_stack((y, X_subset)), classifierFunction=classifier, numberOfSplits=2)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_feature = feature

        if best_feature is not None:
            selected_features.append(best_feature)
            remaining_features.remove(best_feature)
            accuracies.append(best_accuracy)

    return selected_features, accuracies

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

def KFoldCrossValidation(data,classifierFunction,numberOfSplits,return_folds=False):
        
    
    splitData=np.array_split(data,numberOfSplits,axis=0)
    
    accuracyList=[]

    for iteration in range(numberOfSplits):
        trainingData = np.vstack([splitData[i] for i in range(len(splitData)) if i != iteration])
        
        testData=splitData[iteration]
        
        testLabels=classifierFunction(trainingData,testData)
        accuracy=Accuracy(testLabels,testData[:,0])
        
        accuracyList.append(accuracy)

    if return_folds:
        return accuracyList
    else:
        return np.mean(accuracyList)
    
    

def KNearestNeighboors(trainingSet, testSet, k=4, norm=2):
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

def FTestFeatureSelection(data, n_features):
    """
    Perform F-test feature selection and return filtered dataset with labels.
    
    Parameters:
        data (numpy.ndarray): Dataset with labels in the first column.
        n_features (int): Number of top features to select based on F-statistic.
    
    Returns:
        filtered_data (numpy.ndarray): Dataset with labels and selected features.
    """
    images = data[:, 1:]  # Features
    labels = data[:, 0]   # Labels

    f_values, _ = f_classif(images, labels)
    # Get indices of the top n features based on F-statistic
    top_features = np.argsort(f_values)[-n_features:]  # Select top n features
    top_features = np.sort(top_features)  # Sort indices to maintain column order

    # Filter the features and add labels back as the first column
    filtered_features = images[:, top_features]
    filtered_data = np.column_stack((labels, filtered_features))
    return filtered_data
