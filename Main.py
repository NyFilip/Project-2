import numpy as np
import matplotlib.pyplot as plt
import dataSet
import nilsFunction
import natanaelFunction
import filipFunction
import rebeckaFunction

# Load the datasets
catsAndDogs = dataSet.catdog()[0]  # Full dataset with labels and features
mnist = dataSet.mnist()[0]         # Full dataset with labels and features

# Functions and classifiers
ftest = filipFunction.FTestFeatureSelection
kfoldCV = filipFunction.KFoldCrossValidation
classifiers = [
    filipFunction.KNearestNeighboors,
    nilsFunction.QDAClassifier,
    rebeckaFunction.LogisticRegressionMulticlass_sklearn
]

# Datasets to evaluate
datasets = {
    "Cats and Dogs": catsAndDogs,
    "MNIST": mnist
}

# Perform F-test and cross-validation for each classifier and dataset
results = {}
feature_ranges_per_dataset = {}  # Store feature ranges for each dataset

for dataset_name, dataset in datasets.items():
    num_features = dataset.shape[1] - 1  # Total features (excluding labels)
    feature_ranges = [int(num_features * p) for p in [0.01, 0.05, 0.1, 0.2, 0.3,0.5,0.7,0.8,1.0]]  # Dynamic feature ranges
    feature_ranges_per_dataset[dataset_name] = feature_ranges  # Save feature ranges for plotting

    results[dataset_name] = {}
    for classifier in classifiers:
        classifier_name = classifier.__name__
        results[dataset_name][classifier_name] = []

        for n_features in feature_ranges:
            # Perform F-test feature selection
            filtered_data = ftest(dataset, n_features)

            # Split data into features and labels
            X = filtered_data[:, 1:]  # Features
            y = filtered_data[:, 0]  # Labels

            # Perform 3-fold cross-validation
            accuracy = kfoldCV(data=filtered_data, classifierFunction=classifier, numberOfSplits=4)
            results[dataset_name][classifier_name].append(accuracy)

# Plot the results
for dataset_name, dataset_results in results.items():
    feature_ranges = feature_ranges_per_dataset[dataset_name]  # Get the correct feature ranges for this dataset
    plt.figure(figsize=(10, 6))
    for classifier_name, accuracies in dataset_results.items():
        plt.plot(feature_ranges, accuracies, label=classifier_name)

    plt.title(f"Classifier Performance on {dataset_name}")
    plt.xlabel("Number of Selected Features")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()