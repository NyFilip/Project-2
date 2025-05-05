import numpy as np

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

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # for numerical stability
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def LogisticRegressionBinary(trainingSet, testSet, lr=0.01, epochs=1000):
    X_train = trainingSet[:, 1:]
    y_train = trainingSet[:, 0]

    X_test = testSet[:, 1:]

    # Add bias
    X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
    X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))

    weights = np.zeros(X_train.shape[1])

    for _ in range(epochs):
        z = np.dot(X_train, weights)
        predictions = sigmoid(z)

        gradient = np.dot(X_train.T, predictions - y_train) / y_train.size
        weights -= lr * gradient

    # Predict on test
    test_probs = sigmoid(np.dot(X_test, weights))
    testSetLabels = (test_probs >= 0.5).astype(int)  # This is now testSetLabels, not predictions

    return testSetLabels

def LogisticRegressionMulticlass(trainingSet, testSet, lr=0.01, epochs=1000):
    X_train = trainingSet[:, 1:]
    y_train = trainingSet[:, 0].astype(int)

    X_test = testSet[:, 1:]

    num_classes = np.unique(y_train).size
    num_features = X_train.shape[1] + 1

    # Add bias
    X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
    X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))

    weights = np.zeros((num_classes, num_features))

    for cls in range(num_classes):
        y_binary = (y_train == cls).astype(int)
        w = np.zeros(num_features)

        for _ in range(epochs):
            z = np.dot(X_train, w)
            predictions = sigmoid(z)
            gradient = np.dot(X_train.T, predictions - y_binary) / y_binary.size
            w -= lr * gradient

        weights[cls] = w

    # Predict on test
    scores = np.dot(X_test, weights.T)
    probs = softmax(scores)
    testSetLabels = np.argmax(probs, axis=1)  # The correct labels for the test set

    return testSetLabels
