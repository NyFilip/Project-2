import numpy as np
import Classifiers as clf
import dataSet as ds

def preprocess_mnist():
    # Load the MNIST data
    full, labels, imagesMatrix, imagesList = ds.mnist('Numbers.txt')
    
    # Split the data into training and test sets (80% train, 20% test)
    train_size = int(0.8 * len(full))
    mnist_train = full[:train_size]
    mnist_test = full[train_size:]
    
    # Split the labels accordingly
    mnist_train_labels = labels[:train_size]
    mnist_test_labels = labels[train_size:]
    
    return mnist_train, mnist_train_labels, mnist_test, mnist_test_labels

def preprocess_catdog():
    # Load the Cats and Dogs data
    full, labels, imagesMatrix, imagesList = ds.catdog('catdogdata.txt')
    
    # Split the data into training and test sets (80% train, 20% test)
    train_size = int(0.8 * len(full))
    cats_dogs_train = full[:train_size]
    cats_dogs_test = full[train_size:]
    
    # Split the labels accordingly
    cats_dogs_train_labels = labels[:train_size]
    cats_dogs_test_labels = labels[train_size:]
    
    return cats_dogs_train, cats_dogs_train_labels, cats_dogs_test, cats_dogs_test_labels

def run_logistic_regression():
    # For Cats vs Dogs (binary classification)
    testSetLabels_binary = clf.LogisticRegressionBinary(cats_dogs_train, cats_dogs_test)

    # For MNIST (multiclass classification)
    testSetLabels_multiclass = clf.LogisticRegressionMulticlass(mnist_train, mnist_test)

    print("Cats vs Dogs Test Labels (Logistic Regression):")
    print(testSetLabels_binary)

    print("\nMNIST Test Labels (Logistic Regression):")
    print(testSetLabels_multiclass)

def run_knn(k=3):
    # For Cats vs Dogs (binary classification)
    knn_labels_binary = clf.KNearestNeighboors(cats_dogs_train, cats_dogs_test, k)
    
    # For MNIST (multiclass classification)
    knn_labels_multiclass = clf.KNearestNeighboors(mnist_train, mnist_test, k)

    print(f"Cats vs Dogs Test Labels (KNN with k={k}):")
    print(np.unique(knn_labels_binary))

    print("\nMNIST Test Labels (KNN):")
    print(np.unique(knn_labels_multiclass))



if __name__ == '__main__':
    # Preprocess data
    cats_dogs_train, cats_dogs_train_labels, cats_dogs_test, cats_dogs_test_labels = preprocess_catdog()
    mnist_train, mnist_train_labels, mnist_test, mnist_test_labels = preprocess_mnist()

    # Run Logistic Regression Classifiers
    run_logistic_regression()

    # Run K-Nearest Neighbors Classifier with k=3 (you can change k as needed)
    run_knn(k=3)