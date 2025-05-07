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
feed_forward = filipFunction.custom_feed_forward # Assuming this function exists
kfoldCV = filipFunction.KFoldCrossValidation
classifiers = [
    filipFunction.KNearestNeighboors,
    nilsFunction.QDAClassifier,
    nilsFunction.MulticlassLogisticClassifier
]

# Datasets to evaluate
datasets = {
    "Cats and Dogs": catsAndDogs,
    "MNIST": mnist
}

def evaluate_with_ftest_only():
    """
    Perform F-test feature selection and evaluate classifiers.
    """
    results = {}
    feature_ranges_per_dataset = {}  # Store feature ranges for each dataset

    for dataset_name, dataset in datasets.items():
        num_features = dataset.shape[1] - 1  # Total features (excluding labels)
        feature_ranges = [int(num_features * p) for p in [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 1.0]]  # Dynamic feature ranges
        feature_ranges_per_dataset[dataset_name] = feature_ranges  # Save feature ranges for plotting

        results[dataset_name] = {}
        for classifier in classifiers:
            classifier_name = classifier.__name__
            results[dataset_name][classifier_name] = []

            for n_features in feature_ranges:
                # Perform F-test feature selection
                filtered_data = ftest(dataset, n_features)

                # Perform 3-fold cross-validation
                accuracy = kfoldCV(data=filtered_data, classifierFunction=classifier, numberOfSplits=4)
                results[dataset_name][classifier_name].append(accuracy)

    # Plot the results
    for dataset_name, dataset_results in results.items():
        feature_ranges = feature_ranges_per_dataset[dataset_name]  # Get the correct feature ranges for this dataset
        plt.figure(figsize=(10, 6))
        for classifier_name, accuracies in dataset_results.items():
            plt.plot(feature_ranges, accuracies, label=classifier_name)

        plt.title(f"Classifier Performance with F-test on {dataset_name}")
        plt.xlabel("Number of Selected Features")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)
        plt.show()
def evaluate_with_ftest_and_feed_forward(save_to_file=True):
    """
    Perform F-test preprocessing followed by feed-forward feature selection and evaluate classifiers.
    
    Parameters:
    save_to_file (bool): If True, save the accuracy of selected features for each classifier to a text file.
    """
    results = {}
    feature_accuracies_per_dataset = {}  # Store accuracies for each feature added

    for dataset_name, dataset in datasets.items():
        num_features = dataset.shape[1] - 1  # Total features (excluding labels)
        number_of_features = num_features // 2  # Use half the features for F-test preprocessing
        number_of_features = 100
        results[dataset_name] = {}
        feature_accuracies_per_dataset[dataset_name] = {}
        print(dataset_name)
        for classifier in classifiers:
            # print(classifier_name)
            
            classifier_name = classifier.__name__
            print(classifier_name)
            results[dataset_name][classifier_name] = []
            feature_accuracies_per_dataset[dataset_name][classifier_name] = []

            # Perform F-test feature selection
            filtered_data = ftest(dataset, number_of_features)

            # Split data into features and labels
            X = filtered_data[:, 1:]  # Features
            y = filtered_data[:, 0]  # Labels

            # Perform feed-forward feature selection
            selected_features, accuracies = feed_forward(
                X, y, classifier, max_features=number_of_features, kfoldCV=kfoldCV
            )
            
            # Save the accuracies for each feature added
            feature_accuracies_per_dataset[dataset_name][classifier_name] = accuracies

            # Use the final selected features for evaluation
            X_selected = X[:, selected_features]
            filtered_data_selected = np.column_stack((y, X_selected))

            # Perform 3-fold cross-validation
            final_accuracy = kfoldCV(data=filtered_data_selected, classifierFunction=classifier, numberOfSplits=4)
            results[dataset_name][classifier_name].append(final_accuracy)

            # Save accuracies to a text file if save_to_file is True
            if save_to_file:
                file_name = f"{dataset_name}_{classifier_name}_accuracies.txt".replace(" ", "_")
                with open(file_name, "w") as file:
                    file.write(f"Feed-Forward Feature Selection Accuracies for {classifier_name} on {dataset_name}\n")
                    file.write("Feature Count, Accuracy\n")
                    for i, acc in enumerate(accuracies, start=1):
                        file.write(f"{i}, {acc}\n")

    # Plot the results for feed-forward feature selection
    for dataset_name, dataset_results in feature_accuracies_per_dataset.items():
        plt.figure(figsize=(10, 6))
        for classifier_name, accuracies in dataset_results.items():
            plt.plot(range(1, len(accuracies) + 1), accuracies, label=classifier_name)

        plt.title(f"Feed-Forward Feature Selection Accuracy on {dataset_name}")
        plt.xlabel("Number of Features Added")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)
        plt.show()
def evaluate_with_kfoldcv_only():
    """
    Evaluate classifiers using k-fold cross-validation without feature selection and plot the results.
    """
    results = {}

    for dataset_name, dataset in datasets.items():
        results[dataset_name] = {}
        print(f"Evaluating on {dataset_name} dataset...")
        
        for classifier in classifiers:
            classifier_name = classifier.__name__
            print(f"Running {classifier_name}...")
            results[dataset_name][classifier_name] = []

            # Perform k-fold cross-validation
            accuracy = kfoldCV(data=dataset, classifierFunction=classifier, numberOfSplits=4)
            results[dataset_name][classifier_name].append(accuracy)

    # Plot the results
    for dataset_name, dataset_results in results.items():
        plt.figure(figsize=(10, 6))
        classifier_names = list(dataset_results.keys())
        accuracies = [dataset_results[classifier_name][0] for classifier_name in classifier_names]

        plt.bar(classifier_names, accuracies, color='skyblue')
        plt.title(f"Classifier Performance with k-fold CV on {dataset_name}")
        plt.xlabel("Classifiers")
        plt.ylabel("Accuracy")
        plt.xticks(rotation=45)
        plt.grid(axis='y')
        plt.show()



evaluate_with_kfoldcv_only()
# evaluate_with_ftest_and_feed_forward()