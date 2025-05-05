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
full, labels, imagesMatrix, imagesList = catdog('catdogdata.txt')
print(full)
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


#train_set = full[:100]
#test_set = full[100:120]

#logistic_preds = MulticlassLogisticClassifier(train_set, test_set)


import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import numpy as np

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
test_set = full[150:197]
training_set = full[:150]
predicted = MulticlassLogisticClassifier(training_set, test_set)

def compute_error(y_true, y_pred):
    return np.mean(np.array(y_true) != np.array(y_pred))

# Example usage:
y_true = test_set[:, 0].astype(int)          # true labels from first column
#y_pred = MulticlassLogisticClassifier(full[:100], test_set)  # predicted labels

test_error = compute_error(y_true, predicted)
print(f"Test error rate: {test_error:.4f}")


show_predictions(test_set, predicted)
