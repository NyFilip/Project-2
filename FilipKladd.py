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
            # print("error of feature ",feature," was ",CVerror)
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
#random forest: generatenode->generatetree->generateforest->
#need gini stability too
def GiniImpurity(partitionLabels):

    #1-sum of porportion of labels belonging to each class
    uniqueLabels,counts= np.unique(partitionLabels,return_counts=True)
    gini=1-np.sum(counts**2/np.sum(counts))
    return gini
def GenerateNode(mFeatures,data):

    bestFeature=0
    bestThreshold=0
    lowestGini=100
    numberOfSamples=data.shape[1]
    isLeaf=False
    for feature in mFeatures:
        sortedByFeature=data[data[:, feature].argsort()]
        for threshold in range(1,numberOfSamples-1):
            left,right=np.split(sortedByFeature,threshold,axis=1)
            giniLeft=GiniImpurity(left)
            giniRight=GiniImpurity(right)

            if giniLeft==1 or giniRight==1:
                bestThreshold=sortedByFeature[threshold,feature]
                bestFeature=feature
                isLeaf=True
                return bestFeature, bestThreshold, isLeaf


            giniTotal=left.shape[1]/numberOfSamples*giniLeft+right.shape[1]/numberOfSamples*giniRight

            if giniTotal <lowestGini:
                bestFeature=feature
                lowestGini=giniTotal
                bestThreshold=sortedByFeature[threshold,feature]

    return bestFeature,bestThreshold,isLeaf

def SoftThreshold(rho,lam):
    if rho < -lam:
        return rho + lam
    elif rho > lam:
        return rho - lam
    else:
        return 0.0

def LassoCoordinateDescentFast(images,regularizationParam,maxIterations=10000,tolerance=1e-4):
    X=images[:,1:]
    
    #centering X
    
    y=images[:,0]
    X=X-np.mean(X,axis=0)
    y=y-np.mean(y)
    # print(X.shape,y.shape)
    features=X.shape[1]
    numbOfImages=X.shape[0]
    
    beta=np.zeros(features)
    residual=y.copy()
    XCollumnNorms=np.sum(X**2,axis=0)
    for iteration in range(maxIterations):
        print("current iteration: ",iteration)
        oldBeta=beta.copy()
        for feature in range(features):
            # print("current feature: ", feature)
            
            rho=np.dot(X[:,feature].T,residual)+beta[feature]*XCollumnNorms[feature]
            raw=SoftThreshold(rho,regularizationParam)
            newBeta=raw/XCollumnNorms[feature]

            delta=newBeta-beta[feature]

            residual=residual-delta*X[:,feature]

            beta[feature]=newBeta
        
        if np.linalg.norm(beta-oldBeta,ord=1)<tolerance:
            break
    return beta


# def lassoCoordinateDescentFast(images, lam, max_iter=1000, tol=1e-4):
#     X=images[:,1:]
    
#     y=images[:,0]
#     n, p = X.shape
#     beta = np.zeros(p)
#     residual = y.copy()  # 

#     X_squared_norms = np.sum(X ** 2, axis=0)  

#     for iteration in range(max_iter):
#         beta_old = beta.copy()

#         for j in range(p):
#             X_j = X[:, j]
#             rho = X_j @ residual + beta[j] * X_squared_norms[j]  # Add back j-th contribution

#             # Soft thresholding
#             new_beta_j = SoftThreshold(rho / X_squared_norms[j], lam)

#             # Update residual incrementally
#             delta = new_beta_j - beta[j]
#             residual = residual- delta * X_j

#             # Update coefficient
#             beta[j] = new_beta_j

#         if np.linalg.norm(beta - beta_old, ord=1) < tol:
#             break

#     return beta

    



mnistDigits= dataSet.mnist('numbers.txt')
numbers=mnistDigits[0]
# print(numbers)
catsAndDogs = dataSet.catdog('catdogdata.txt')
animals=catsAndDogs[0]

# from sklearn.linear_model import Lasso
# from sklearn.datasets import make_regression

# X, y = make_regression(n_samples=100, n_features=20, noise=0.1)
# model = Lasso(alpha=0.1)  # alpha is λ
# model.fit(X, y)

# print(model.coef_)
# print(animals)
# numbers=np.loadtxt('Numbers.txt')

# print(catsAndDogs.shape,numbers.shape)
# print(catsAndDogs,numbers)
# print(catsAndDogs[0,1:])
# nilsFunction.display_images_from_rows(catsAndDogs[[98,99],1:])
# selectedFeatures=FeedFoorward(numbers,KNearestNeighboors,0.001,10)
images=animals
X=images[:,1:]
y=images[:,0]

from sklearn.linear_model import Lasso

from sklearn.feature_selection import f_classif

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

# model = Lasso(alpha=0.1, fit_intercept=False, max_iter=10000)
# model.fit(X, y)
# beta = model.coef_

# predictedLabels= KNearestNeighboors(animals[:150,:],animals[150:,:],k=4)
# acc=Accuracy(predictedLabels,animals[150:,0])

# print(acc)
# # print(animals[:,1000:3000])
# beta=LassoCoordinateDescentFast(animals,20)
# nonZeroIndices=np.nonzero(beta)
# print(beta[nonZeroIndices].shape,nonZeroIndices)
# testlabels=nilsFunction.MulticlassLogisticClassifier(animals[:150],animals[150:],)
# print(testlabels)
# selectedfeatures=FeedFoorward(animals,nilsFunction.MulticlassLogisticClassifier(animals[:150,:],animals[150:,:]),1e-4,10)
# print(selectedfeatures)

# predictedLabels= KNearestNeighboors(animals[:150,[0,1487, 2605, 2473, 2152, 3089]],animals[150:,[0, 1487, 2605, 2473, 2152,3089]])


# Perform F-test feature selection
n_features = 1000  # Number of top features to select
filtered_data = FTestFeatureSelection(animals, n_features)
print("Filtered data shape:", filtered_data.shape)

# Split the filtered data into training and testing sets
training_data = filtered_data[:150]
testing_data = filtered_data[150:]

# Use the filtered features in your classifier
predictedLabels = KNearestNeighboors(training_data, testing_data)

# Calculate accuracy
acc = Accuracy(predictedLabels, testing_data[:, 0])
print("Accuracy:", acc)

