import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

def display_images_from_rows(array, cmap='gray'):
    """
    Display each row of a 2D numpy array as a square image.

    Parameters:
    - array: np.ndarray, shape (n_images, image_size), each row is a flattened square image.
    - cmap: colormap to use for displaying the images (default: 'gray')
    """
    if array.ndim == 1:
        array = array[np.newaxis, :]  # shape becomes (1, image_size)
    num_images = array.shape[0]
    print(f"Number of images: {num_images}")
    for i in range(num_images):
        img_flat = array[i]
        side = int(math.sqrt(len(img_flat)))
        if side * side != len(img_flat):
            raise ValueError(f"Row {i} cannot be reshaped into a square image.")
        
        img = img_flat.reshape((side, side)).T
        plt.figure()
        plt.imshow(img, cmap=cmap, vmin=0, vmax=255)
        plt.title(f"Image {i}")
        plt.axis('off')
    plt.show()

def MulticlassLogisticClassifier(trainingSet, testSet, scale=True, C=1.0, penalty=None, solver='lbfgs', max_iter=5000):
    """
    Multiclass logistic classifier using scikit-learn.
    
    Parameters:
        trainingSet : np.ndarray, shape (n_samples_train, n_features+1)
            First column is label, rest are features.
        testSet : np.ndarray, shape (n_samples_test, n_features+1)
            First column is label, rest are features.
        scale : bool
            Whether to standardize the input features.
        C : float
            Inverse regularization strength.
        penalty : str
            Regularization type: 'l2', 'l1', or 'none'.
        solver : str
            Optimization algorithm. 'lbfgs' or 'saga' for L1.
        max_iter : int
            Max iterations for solver.
    
    Returns:
        y_pred : list of predicted labels
    """
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
        
        solver=solver,
        penalty=penalty,
        C=C,
        max_iter=max_iter
    )
    clf.fit(X_train, y_train)

    # Predict
    y_pred = clf.predict(X_test)
    return y_pred.tolist()

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

def catdog_with_contamination(filePath='catdogdata.txt', contamination_rate=0.0):
    import numpy as np
    sLabels = []
    sImagesMatrix = []
    sImagesList = []
    full = []
    i = 0

    with open(filePath, 'r') as file:
        for line in file:
            parts = line.strip().split()
            index = parts[0].strip()
            pixels = list(map(int, parts[1:]))
            fullTemp = list(pixels)
            fullTemp.insert(0, 0 if i < 99 else 1)
            full.append(fullTemp)
            i += 1

    full = np.array(full)
    np.random.seed(10)
    np.random.shuffle(full)

    if contamination_rate > 0.0:
        n_pixels = 64 * 64
        n_contaminate = int(np.round(contamination_rate * n_pixels))
        contam_indices = np.random.choice(n_pixels, size=n_contaminate, replace=False)

        # Generate random values between 2 and 255
        random_values = np.random.randint(2, 256, size=n_contaminate)

        full[:, contam_indices + 1] = random_values
        # reshape images after contamination
    for line in full:
        slabel = line[0]
        spixels = line[1:]
        sLabels.append(slabel)
        sImagesList.append(spixels)
        sImagesMatrix.append(np.array(spixels).reshape(64, 64))

    sLabels = np.array(sLabels)
    sImagesList = np.array(sImagesList)
    sImagesMatrix = np.array(sImagesMatrix)

    return full, sLabels, sImagesMatrix, sImagesList

def mnist_with_contamination(filePath='Numbers.txt', contamination_rate=0.0):
    import numpy as np
    imagesList = []
    imagesMatrix = []
    labels = []
    full = []

    with open(filePath, 'r') as file:
        for line in file:
            parts = line.strip().split()
            index = parts[0].strip()
            label = np.array(int(parts[1]))
            fullTemp = np.array(list(map(float, parts[1:])))
            pixels = list(map(float, parts[2:]))
            pixelArray = np.array(pixels).reshape(16, 16)
            normArray = (pixelArray + 1) / 2  # now in [0, 1]
            labels.append(label)
            imagesMatrix.append(normArray)
            full.append(fullTemp)
            imagesList.append(np.array(normArray).reshape(1, 256))

    full = np.array(full)
    labels = np.array(labels)
    imagesMatrix = np.array(imagesMatrix)
    imagesList = np.array(imagesList)

    if contamination_rate > 0.0:
        n_pixels = 256
        n_contaminate = int(np.round(contamination_rate * n_pixels))
        contam_indices = np.random.choice(n_pixels, size=n_contaminate, replace=False)

        # Generate one random value per contaminated pixel (same across all images)
        random_values = np.random.uniform(0.0, 1.0, size=n_contaminate)

        full[:, contam_indices + 1] = random_values
        for i in range(len(imagesList)):
            imagesList[i][0, contam_indices] = random_values
            imagesMatrix[i].flat[contam_indices] = random_values

    return full, labels, np.array(imagesMatrix), np.array(imagesList)
