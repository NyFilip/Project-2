import rebeckaFunction as rF

if __name__ == '__main__':
    rF.hello()



import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from collections import Counter


# Remove the first column
if catsAndDogs.shape[1] == 4097:
    catsAndDogs = catsAndDogs[:, 1:]
labels = np.zeros(198)
labels[99:] = 1
# Split into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(catsAndDogs, labels, test_size=0.2, random_state=42)

# Feature Selection Method: Filter Method using ANOVA F-test
# Select the top 100 features using F-test
filter_selector = SelectKBest(score_func=f_classif, k=100)
X_train_filter = filter_selector.fit_transform(X_train, y_train)
X_test_filter = filter_selector.transform(X_test)

# Feature Selection Method: Wrapper Method using Recursive Feature Elimination (RFE)
# Use logistic regression inside RFE to select features
log_reg = LogisticRegression(max_iter=1000)
wrapper_selector = RFE(estimator=log_reg, n_features_to_select=100, step=50)
X_train_wrapper = wrapper_selector.fit_transform(X_train, y_train)
X_test_wrapper = wrapper_selector.transform(X_test)

# Classifiers
LogReg = LogisticRegression(max_iter=1000)
RandomForest = RandomForestClassifier()
SVC = SVC()

# Function to test classifiers
def evaluate_model(clf, X_train, y_train):
    # Use 5-fold cross-validation to evaluate accuracy
    scores = cross_val_score(clf, X_train, y_train, cv=5)
    print(f"{clf.__class__.__name__} Average Accuracy: {scores.mean():.2f}")

print("=== Using Filter Method (Top 100 features by ANOVA) ===")
evaluate_model(LogReg, X_train_filter, y_train)
evaluate_model(RandomForest, X_train_filter, y_train)
evaluate_model(SVC, X_train_filter, y_train)

print("\n=== Using Wrapper Method (RFE with Logistic Regression) ===")
evaluate_model(LogReg, X_train_wrapper, y_train)
evaluate_model(RandomForest, X_train_wrapper, y_train)
evaluate_model(SVC, X_train_wrapper, y_train)

# Function to run selection N times and collect selected features
def repeat_feature_selection(method='filter', n_repeats=5):
    selected_indices = []

    for seed in range(n_repeats):
        X_train, _, y_train, _ = train_test_split(catsAndDogs, labels, test_size=0.2, random_state=seed)
        
        if method == 'filter':
            selector = SelectKBest(score_func=f_classif, k=100)
        elif method == 'wrapper':
            selector = RFE(LogisticRegression(max_iter=1000), n_features_to_select=100, step=50)
        else:
            raise ValueError("method must be 'filter' or 'wrapper'")

        selector.fit(X_train, y_train)
        selected = np.where(selector.get_support())[0]
        selected_indices.extend(selected)

    return selected_indices

# Run both methods
filter_selected = repeat_feature_selection(method='filter', n_repeats=10)
wrapper_selected = repeat_feature_selection(method='wrapper', n_repeats=10)

# Count how often each pixel is selected
filter_counts = Counter(filter_selected)
wrapper_counts = Counter(wrapper_selected)

# Visualize how stable the selections are
def visualize_stability(counts, title):
    heatmap = np.zeros(4096)
    for idx, freq in counts.items():
        heatmap[idx] = freq
    plt.figure(figsize=(5, 5))
    plt.imshow(heatmap.reshape(64, 64), cmap='hot')
    plt.title(title + " (10 runs)")
    plt.axis('off')
    plt.colorbar(label='Times Selected')
    plt.show()

visualize_stability(filter_counts, "Filter Method Stability")
visualize_stability(wrapper_counts, "Wrapper Method Stability")


# Function to overlay selected pixel heatmap on an actual image
def overlay_on_image(image_row, selection_counts, title):
    image_2d = image_row.reshape(64, 64)
    
    # Create heatmap: normalize counts to [0, 1]
    heatmap = np.zeros(4096)
    for idx, count in selection_counts.items():
        heatmap[idx] = count
    heatmap = heatmap.reshape(64, 64)
    heatmap = heatmap / np.max(heatmap)
    
    plt.figure(figsize=(5, 5))
    plt.imshow(image_2d, cmap='gray')
    plt.imshow(heatmap, cmap='hot', alpha=0.5)
    plt.title(title)
    plt.axis('off')
    plt.show()

# Pick a sample image
sample_cat = catsAndDogs[0]
sample_dog = catsAndDogs[150]

# Overlay heatmaps
overlay_on_image(sample_cat, filter_counts, "Filter Method Overlay on Cat")
overlay_on_image(sample_cat, wrapper_counts, "Wrapper Method Overlay on Cat")
overlay_on_image(sample_dog, filter_counts, "Filter Method Overlay on Dog")
overlay_on_image(sample_dog, wrapper_counts, "Wrapper Method Overlay on Dog")