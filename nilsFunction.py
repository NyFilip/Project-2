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

    clf = QuadraticDiscriminantAnalysis(reg_param=1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    return y_pred.tolist()


