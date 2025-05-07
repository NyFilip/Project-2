import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import dataSet
import nilsFunction
import natanaelFunction
import filipFunction
import rebeckaFunction
import os

# Create folder if it doesn't exist
os.makedirs('confusion_matrices', exist_ok=True)


# Load the datasets
catsAndDogs = dataSet.catdog()[0]
mnist = dataSet.mnist()[0]

# Functions and classifiers
ftest = filipFunction.FTestFeatureSelection
feed_forward = filipFunction.custom_feed_forward
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

def C_matrix(true_labels, predicted_labels, class_names=None, title='Confusion Matrix', filename=None):
    cm = confusion_matrix(true_labels, predicted_labels)

    if class_names is None:
        # Default class names from labels
        class_names = [str(int(lbl)) for lbl in np.unique(true_labels)]

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()

    if filename:
        plt.savefig(filename)
        print(f"Saved confusion matrix to {filename}")
    plt.close()

    return cm


def split_data(data, test_size=0.25):
    """Simple train/test split keeping labels in first column"""
    n_samples = data.shape[0]
    n_train = int((1 - test_size) * n_samples)
    indices = np.random.permutation(n_samples)
    train_idx, test_idx = indices[:n_train], indices[n_train:]
    return data[train_idx], data[test_idx]

def run_all():
    for dataset_name, dataset in datasets.items():
        print(f"\n--- Dataset: {dataset_name} ---")

        # Define nice label names for confusion matrix
        if dataset_name == "Cats and Dogs":
            label_names = ["Cat", "Dog"]
        else:
            label_names = None  # Let it use numeric labels (for MNIST)

        for classifier in classifiers:
            classifier_name = classifier.__name__
            print(f"\nClassifier: {classifier_name}")

            # 1. Without feature selection
            train_data, test_data = split_data(dataset)

            preds = classifier(train_data, test_data)
            true_labels = test_data[:, 0]

            file_without = f'confusion_matrices/{dataset_name}_{classifier_name}_without_FS.png'
            C_matrix(true_labels, preds, title=f'{classifier_name} without FS on {dataset_name}',
                     filename=file_without, class_names=label_names)

            # 2. With feature selection
            num_features_total = dataset.shape[1] - 1
            number_of_features = min(100, num_features_total // 2)

            filtered_data = ftest(dataset, number_of_features)
            X_filtered = filtered_data[:, 1:]
            y_filtered = filtered_data[:, 0]

            selected_features, _ = feed_forward(
                X_filtered, y_filtered, classifier,
                max_features=number_of_features, kfoldCV=kfoldCV
            )

            X_selected = X_filtered[:, selected_features]
            filtered_data_selected = np.column_stack((y_filtered, X_selected))

            train_data_sel, test_data_sel = split_data(filtered_data_selected)

            preds_sel = classifier(train_data_sel, test_data_sel)
            true_labels_sel = test_data_sel[:, 0]

            file_with = f'confusion_matrices/{dataset_name}_{classifier_name}_with_FS.png'
            C_matrix(true_labels_sel, preds_sel, title=f'{classifier_name} with FS on {dataset_name}',
                     filename=file_with, class_names=label_names)

    for dataset_name, dataset in datasets.items():
        print(f"\n--- Dataset: {dataset_name} ---")

        for classifier in classifiers:
            classifier_name = classifier.__name__
            print(f"\nClassifier: {classifier_name}")

            # 1. Without feature selection
            train_data, test_data = split_data(dataset)

            preds = classifier(train_data, test_data)
            true_labels = test_data[:, 0]
            print("Confusion Matrix WITHOUT Feature Selection")
            file_without = f'confusion_matrices/{dataset_name}_{classifier_name}_without_FS.png'
            C_matrix(true_labels, preds, title=f'{classifier_name} without FS on {dataset_name}', filename=file_without)


            # 2. With F-test and Feed-Forward feature selection
            num_features_total = dataset.shape[1] - 1
            number_of_features = min(100, num_features_total // 2)

            # F-test
            filtered_data = ftest(dataset, number_of_features)

            # Prepare for feed-forward selection
            X_filtered = filtered_data[:, 1:]
            y_filtered = filtered_data[:, 0]

            selected_features, _ = feed_forward(
                X_filtered, y_filtered, classifier,
                max_features=number_of_features, kfoldCV=kfoldCV
            )

            # Filter down to selected features
            X_selected = X_filtered[:, selected_features]
            filtered_data_selected = np.column_stack((y_filtered, X_selected))

            # Split again after feature selection
            train_data_sel, test_data_sel = split_data(filtered_data_selected)

            preds_sel = classifier(train_data_sel, test_data_sel)
            true_labels_sel = test_data_sel[:, 0]
            print("Confusion Matrix WITH Feature Selection (F-test + Feed-Forward)")
            file_with = f'confusion_matrices/{dataset_name}_{classifier_name}_with_FS.png'
            C_matrix(true_labels_sel, preds_sel, title=f'{classifier_name} with FS on {dataset_name}', filename=file_with)

if __name__ == "__main__":
    run_all()
