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

# from sklearn.linear_model import LogisticRegression

def LogisticRegressionBinary_sklearn(trainingSet, testSet):
    X_train = trainingSet[:, 1:]
    y_train = trainingSet[:, 0]

    X_test = testSet[:, 1:]

    # Use scikit-learn's Logistic Regression
    clf = LogisticRegression(solver='liblinear', max_iter=1000)
    clf.fit(X_train, y_train)

    testSetLabels = clf.predict(X_test)
    return testSetLabels

def LogisticRegressionMulticlass_sklearn(trainingSet, testSet):
    X_train = trainingSet[:, 1:]
    y_train = trainingSet[:, 0].astype(int)

    X_test = testSet[:, 1:]

    clf = LogisticRegression(solver='saga', multi_class='multinomial', max_iter=1000)
    clf.fit(X_train, y_train)

    testSetLabels = clf.predict(X_test)
    return testSetLabels