import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt

catsAndDogs = np.loadtxt('catdogdata.txt')
# Remove the first column
if catsAndDogs.shape[1] == 4097:
    catsAndDogs = catsAndDogs[:, 1:]
labels = np.zeros(198)
labels[99:] = 1
# Split into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(catsAndDogs, labels, test_size=0.2, random_state=42)

# --- Feature Selection Method 1: Filter Method using ANOVA F-test ---
# Select the top 100 features using F-test
filter_selector = SelectKBest(score_func=f_classif, k=100)
X_train_filter = filter_selector.fit_transform(X_train, y_train)
X_test_filter = filter_selector.transform(X_test)

# --- Feature Selection Method 2: Wrapper Method using Recursive Feature Elimination (RFE) ---
# Use logistic regression inside RFE to select features
log_reg = LogisticRegression(max_iter=1000)
wrapper_selector = RFE(estimator=log_reg, n_features_to_select=100, step=50)
X_train_wrapper = wrapper_selector.fit_transform(X_train, y_train)
X_test_wrapper = wrapper_selector.transform(X_test)

# --- Classifiers ---
# We'll use 3 simple classifiers

# 1. Logistic Regression
clf1 = LogisticRegression(max_iter=1000)

# 2. Random Forest
clf2 = RandomForestClassifier()

# 3. Support Vector Machine
clf3 = SVC()

# Define a helper function to test classifiers
def evaluate_model(clf, X_train, y_train):
    # Use 5-fold cross-validation to evaluate accuracy
    scores = cross_val_score(clf, X_train, y_train, cv=5)
    print(f"{clf.__class__.__name__} Average Accuracy: {scores.mean():.2f}")

print("=== Using Filter Method (Top 100 features by ANOVA) ===")
evaluate_model(clf1, X_train_filter, y_train)
evaluate_model(clf2, X_train_filter, y_train)
evaluate_model(clf3, X_train_filter, y_train)

print("\n=== Using Wrapper Method (RFE with Logistic Regression) ===")
evaluate_model(clf1, X_train_wrapper, y_train)
evaluate_model(clf2, X_train_wrapper, y_train)
evaluate_model(clf3, X_train_wrapper, y_train)

# Optional: See which features were selected
print("\nSelected features (filter method):", np.where(filter_selector.get_support())[0])
print("Selected features (wrapper method):", np.where(wrapper_selector.get_support())[0])

# Function to visualize selected features
def visualize_selected_features(mask, title):
    # Reshape the 1D boolean mask (length 4096) to 64x64
    mask_2d = mask.reshape(64, 64)
    
    plt.figure(figsize=(4, 4))
    plt.imshow(mask_2d, cmap='gray', interpolation='nearest')
    plt.title(title)
    plt.axis('off')
    plt.show()

# Visualize selected features (filter method)
visualize_selected_features(filter_selector.get_support(), "Filter Method (F-test) Selected Pixels")

# Visualize selected features (wrapper method)
visualize_selected_features(wrapper_selector.get_support(), "Wrapper Method (RFE) Selected Pixels")








#numbers=np.genfromtxt('Numbers.txt')
#print(catsAndDogs.shape,numbers.shape)
#print(catsAndDogs,numbers)