import numpy as np
import matplotlib.pyplot as plt

def mnist(filePath = 'Numbers.txt'):
    imagesList = []
    imagesMatrix = []
    labels = []
    full = []
    with open(filePath, 'r') as file:
        for line in file:
            parts = line.strip().split()
            
            index = parts[0].strip()
            label = np.array(int(parts[1]))
            fullTemp  = np.array(list(map(float,parts[1:])))
            pixels = list(map(float,parts[2:]))
            
            pixelArray = np.array(pixels).reshape(16,16)
            normArray = (pixelArray + 1) / 2
              
            
            labels.append(label)
            imagesMatrix.append(normArray)
            full.append(fullTemp)
            imagesList.append(np.array(normArray).reshape(1,256))
    
    full = np.array(full)
    labels = np.array(labels)
    imagesMatrix = np.array(imagesMatrix)
    imagesList = np.array(imagesList)                    
    return full, labels, imagesMatrix, imagesList
def catdog(filePath = 'catdogdata.txt'):
    labels = []
    labels[:98] = np.zeros(99)
    labels[99:] = np.ones(99)
    sLabels = []
    sImagesMatrix = []
    sImagesList = []
    full = []
    i = 0
    with open(filePath, 'r') as file:
        for line in file:
            parts = line.strip().split()
            
            index = parts[0].strip()
            pixels = list(map(int,parts[1:]))

            fullTemp = list(pixels)
            if i < 99:
                fullTemp.insert(0, 0)
            else:
                fullTemp.insert(0,1)
            full.append(fullTemp)
            i += 1
    full = np.array(full)

    np.random.shuffle(full)

    for line in full:
        slabel = line[0]
        spixels = line[1:]
        sLabels.append(slabel)
        sImagesList.append(spixels)
        sImagesMatrix.append(np.array(spixels).reshape(64,64))
    sLabels = np.array(sLabels)
    sImagesList = np.array(sImagesList)
    sImagesMatrix = np.array(sImagesMatrix)
    
    
    return full, sLabels, sImagesMatrix, sImagesList
#full, labels, imagesMatrix, imagesList = catdog('catdogdata.txt')
full, labels, imagesMatrix, imagesList = mnist('Numbers.txt')
print(full.shape)

#train_set = full[:100]
#test_set = full[100:120]
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

def QDAClassifier(trainingSet, testSet, scale=True):
    """
    QDA classifier using scikit-learn.
    
    Parameters:
        trainingSet: np.ndarray
            First column is label, rest are features.
        testSet: np.ndarray
            First column is label, rest are features.
        scale: bool
            Whether to standardize input features.

    Returns:
        y_pred: list of predicted labels
    """
    X_train = trainingSet[:, 1:]
    y_train = trainingSet[:, 0].astype(int)

    X_test = testSet[:, 1:]
    y_test = testSet[:, 0].astype(int)

    # Optional feature scaling
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    clf = QuadraticDiscriminantAnalysis()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    return y_pred.tolist()


    X_train = trainingSet[:, 1:]
    y_train = trainingSet[:, 0].astype(int)

    X_test = testSet[:, 1:]
    y_test = testSet[:, 0].astype(int)

    # Normalize if desired
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    # Train model
    clf = LogisticRegression(
        multi_class='multinomial',
        solver=solver,
        penalty=penalty,
        C=C,
        max_iter=max_iter
    )
    clf.fit(X_train, y_train)

    # Predict
    y_pred = clf.predict(X_test)
    return y_pred.tolist()

#logistic_preds = MulticlassLogisticClassifier(train_set, test_set)
def show_predictions(testSet, predicted_labels, num_rows=10, num_cols=10):
    """
    Display test images in a grid with predicted labels.
    Automatically handles square image reshaping based on feature count.
    """
    num_images = min(num_rows * num_cols, len(testSet))
    
    # Determine image size from the number of features (excluding label column)
    num_pixels = testSet.shape[1] - 1
    image_dim = int(np.sqrt(num_pixels))
    if image_dim ** 2 != num_pixels:
        raise ValueError("Image does not appear to be square.")

    plt.figure(figsize=(num_cols, num_rows))

    for i in range(num_images):
        img = testSet[i, 1:].reshape(image_dim, image_dim)
        label = predicted_labels[i]
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(f'{label}', fontsize=8)
        plt.axis('off')

    plt.tight_layout()
    plt.show()



# Example usage:
test_set = full[1500:1600]
training_set = full[:1500]
predicted = QDAClassifier(training_set, test_set)

def compute_error(y_true, y_pred):
    return np.mean(np.array(y_true) != np.array(y_pred))

# Example usage:
y_true = test_set[:, 0].astype(int)          # true labels from first column
#y_pred = MulticlassLogisticClassifier(full[:100], test_set)  # predicted labels

test_error = compute_error(y_true, predicted)
print(f"accuracy: {1-test_error:.4f}")


show_predictions(test_set, predicted)
