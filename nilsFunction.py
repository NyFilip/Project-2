import numpy as np
import matplotlib.pyplot as plt
import math

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

def MulticlassLogisticClassifier(trainingSet, testSet, max_iter=2000, tol=1e-7):
    import numpy as np

    def softmax(Z):
        eZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return eZ / np.sum(eZ, axis=1, keepdims=True)

    def one_hot(y, K):
        return np.eye(K)[y]

    # --- Step 1: Extract features and labels ---
    X_train = trainingSet[:, 1:]
    y_train = trainingSet[:, 0].astype(int)

    # --- Step 2: Map original labels to 0...K-1 indices ---
    unique_labels = np.unique(y_train)
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    index_to_label = {idx: label for label, idx in label_to_index.items()}
    y_indexed = np.array([label_to_index[y] for y in y_train])
    num_classes = len(unique_labels)

    # --- Step 3: Prepare design matrix ---
    N, p = X_train.shape
    p_aug = p + 1
    X_aug = np.hstack([np.ones((N, 1)), X_train])
    Y = one_hot(y_indexed, num_classes)

    # --- Step 4: Initialize parameters ---
    theta = np.random.randn(num_classes, p_aug) * 0.01
    theta[:, 0] = 0.0  # intercepts

    # --- Step 5: IRLS training loop ---
    for iteration in range(max_iter):
        Z = X_aug @ theta.T
        P = softmax(Z)
        grad = (Y - P).T @ X_aug

        H = np.zeros((num_classes * p_aug, num_classes * p_aug))
        for i in range(N):
            x_i = X_aug[i:i+1, :]
            p_i = P[i, :]
            for k in range(num_classes):
                for l in range(num_classes):
                    coeff = -p_i[k] * ((k == l) - p_i[l])
                    H_k_l = coeff * (x_i.T @ x_i)
                    row_start = k * p_aug
                    col_start = l * p_aug
                    H[row_start:row_start + p_aug, col_start:col_start + p_aug] += H_k_l

        grad_flat = grad.flatten()
        try:
            delta = np.linalg.solve(H, grad_flat)
        except np.linalg.LinAlgError:
            print("Hessian not invertible. Stopping early.")
            break

        theta_new = theta.flatten() + delta
        theta_new = theta_new.reshape(num_classes, p_aug)

        if np.linalg.norm(theta_new - theta) < tol:
            break

        theta = theta_new

    # --- Step 6: Predict on test set ---
    X_test = testSet[:, 1:]
    X_test_aug = np.hstack([np.ones((X_test.shape[0], 1)), X_test])
    Z_test = X_test_aug @ theta.T
    P_test = softmax(Z_test)
    y_pred_indices = np.argmax(P_test, axis=1)
    y_pred_labels = [index_to_label[idx] for idx in y_pred_indices]

    return y_pred_labels
    