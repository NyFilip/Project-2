from sklearn.linear_model import LogisticRegression

def LogisticRegressionBinary_sklearn(trainingSet, testSet):
    X_train = trainingSet[:, 1:]
    y_train = trainingSet[:, 0]

    X_test = testSet[:, 1:]

    clf = LogisticRegression(solver='liblinear', max_iter=1000)
    clf.fit(X_train, y_train)

    testSetLabels = clf.predict(X_test)
    return testSetLabels

def LogisticRegressionMulticlass_sklearn(trainingSet, testSet):
    X_train = trainingSet[:, 1:]
    y_train = trainingSet[:, 0].astype(int)

    X_test = testSet[:, 1:]

    clf = LogisticRegression(solver='saga', max_iter=5000)
    clf.fit(X_train, y_train)

    testSetLabels = clf.predict(X_test)
    return testSetLabels