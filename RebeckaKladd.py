import numpy as np
import rebeckaFunction as rF
import dataSet as ds
from filipFunction import Accuracy

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

def run_logistic_regression(cats_dogs_train, cats_dogs_test, cats_dogs_test_labels,
                             mnist_train, mnist_test, mnist_test_labels):
    # For Cats vs Dogs (binary classification)
    predicted_binary = rF.LogisticRegressionBinary_sklearn(cats_dogs_train, cats_dogs_test)
    acc_binary = Accuracy(predicted_binary, cats_dogs_test_labels)
    print("Cats vs Dogs Test Labels (Logistic Regression):")
    print(predicted_binary)
    print("Cats vs Dogs Accuracy (Logistic Regression): {:.2f}%".format(acc_binary))

    # For MNIST (multiclass classification)
    predicted_multiclass = rF.LogisticRegressionMulticlass_sklearn(mnist_train, mnist_test)
    acc_multiclass = Accuracy(predicted_multiclass, mnist_test_labels)
    print("\nMNIST Test Labels (Logistic Regression):")
    print(predicted_multiclass)
    print("MNIST Accuracy (Logistic Regression): {:.2f}%".format(acc_multiclass))

    return predicted_binary, predicted_multiclass

if __name__ == '__main__':
    # Preprocess data
    cats_dogs_train, cats_dogs_train_labels, cats_dogs_test, cats_dogs_test_labels = preprocess_catdog()
    mnist_train, mnist_train_labels, mnist_test, mnist_test_labels = preprocess_mnist()

    # Run Logistic Regression Classifiers
    predicted_binary, predicted_multiclass = run_logistic_regression(cats_dogs_train, cats_dogs_test, cats_dogs_test_labels,
        mnist_train, mnist_test, mnist_test_labels)
    
    cm_cats_dogs = rF.C_matrix(cats_dogs_test_labels, predicted_binary, class_names=['Cat', 'Dog'], title='Confusion Matrix: Cats & Dogs')

    cm_mnist = rF.C_matrix(mnist_test_labels, predicted_multiclass, title='Confusion Matrix: MNIST')
