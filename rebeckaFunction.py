from sklearn.linear_model import LogisticRegression
import numpy as np

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

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def C_matrix(true_labels, predicted_labels, class_names=None, title='Confusion Matrix'):
    cm = confusion_matrix(true_labels, predicted_labels)

    # If class_names is not provided, generate them automatically
    if class_names is None:
        class_names = [str(int(lbl)) for lbl in np.unique(true_labels)]
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names if class_names else 'auto',
                yticklabels=class_names if class_names else 'auto')
    
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()
    
    return cm