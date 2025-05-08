import numpy as np
import matplotlib.pyplot as plt
import filipFunction as fF
import nilsFunction as nF
from typing import Callable, Tuple, List
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import matplotlib.patches as patches
def catdog_subsection(filePath='catdogdata.txt', subsection_origin=None, subsection_size=None):
    sLabels = []
    sImagesMatrix = []
    sImagesList = []
    full = []

    use_crop = subsection_origin is not None and subsection_size is not None
    x0, y0 = subsection_origin if use_crop else (0, 0)
    s = subsection_size if use_crop else 64

    with open(filePath, 'r') as file:
        for i, line in enumerate(file):
            parts = line.strip().split()
            pixels = list(map(int, parts[1:]))
            label = 0 if i < 99 else 1
            fullTemp = [label] + pixels
            full.append(fullTemp)

    full = np.array(full)
    np.random.seed(10)
    np.random.shuffle(full)

    for line in full:
        slabel = line[0]
        spixels = line[1:]
        full_img = np.array(spixels).reshape(64, 64).T

        if use_crop:
            crop = full_img[y0:y0+s, x0:x0+s]
            sImagesMatrix.append(crop)
            sImagesList.append(crop.reshape(1, s * s))
        else:
            sImagesMatrix.append(full_img)
            sImagesList.append(spixels)

        sLabels.append(slabel)

    return (
        full,
        np.array(sLabels),
        np.array(sImagesMatrix),
        np.array(sImagesList)
    )

def mnist_subsection(filePath='Numbers.txt', subsection_origin=None, subsection_size=None):
    imagesList = []
    imagesMatrix = []
    labels = []
    full = []

    use_crop = subsection_origin is not None and subsection_size is not None
    x0, y0 = subsection_origin if use_crop else (0, 0)
    s = subsection_size if use_crop else 16

    with open(filePath, 'r') as file:
        for line in file:
            parts = line.strip().split()
            label = int(parts[1])
            pixels = list(map(float, parts[2:]))
            pixelArray = np.array(pixels).reshape(16, 16)
            normArray = (pixelArray + 1) / 2

            if use_crop:
                crop = normArray[y0:y0+s, x0:x0+s]
                imagesMatrix.append(crop)
                imagesList.append(crop.reshape(1, s * s))
            else:
                imagesMatrix.append(normArray)
                imagesList.append(normArray.reshape(1, 256))

            fullTemp = np.array(list(map(float, parts[1:])))
            full.append(fullTemp)
            labels.append(label)

    return (
        np.array(full),
        np.array(labels),
        np.array(imagesMatrix),
        np.array(imagesList)
    )

def show_first_images(imagesMatrix, labels=None, num_images=10):
    """
    Displays the first `num_images` from imagesMatrix.
    Optionally shows labels as titles.
    """
    num_images = min(num_images, len(imagesMatrix))
    img_shape = int(np.sqrt(imagesMatrix.shape[1])) if imagesMatrix.ndim == 2 else imagesMatrix.shape[1:3]

    plt.figure(figsize=(15, 2))
    for i in range(num_images):
        img = imagesMatrix[i]
        if img.ndim == 1:  # flat image
            img = img.reshape(img_shape)
        plt.subplot(1, num_images, i + 1)
        plt.imshow(img, cmap='gray')
        title = f"#{i}" if labels is None else f"{labels[i]}"
        plt.title(title, fontsize=10)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def plot_images(images, labels=None, num=9, title_prefix='Label'):
    """
    Plots a square grid of images (or the first `num` images).
    
    Parameters:
    - images: numpy array of shape (N, H, W)
    - labels: optional, array of labels of length N
    - num: how many images to show (default 9)
    - title_prefix: prefix to add before each label in title
    """
    num = min(num, len(images))
    grid_size = int(np.ceil(np.sqrt(num)))
    fig, axs = plt.subplots(grid_size, grid_size, figsize=(10, 10))
    axs = axs.flatten()
    
    for i in range(grid_size**2):
        axs[i].axis('off')
        if i < num:
                axs[i].imshow(images[i], cmap='gray')  # remove vmin/vmax so matplotlib auto-scales            if labels is not None:
                axs[i].set_title(f"{title_prefix}: {labels[i]}")
    plt.tight_layout()
    plt.show()

def get_subsections(big_side, sub_side):
    if sub_side > big_side:
        return []

    positions = []

    # How many full subsections fit in each direction
    n = big_side // sub_side

    if n == 0:
        # Only one subsection fits: place it in the center
        margin = (big_side - sub_side) // 2
        return [[margin, margin]]

    # Grid spacing
    step = sub_side

    # Offset to center the grid in the big square
    start = (big_side - n * sub_side) // 2

    # Add the grid positions
    grid_centers = []
    for i in range(n):
        for j in range(n):
            x = start + i * step
            y = start + j * step
            positions.append([x, y])
            grid_centers.append((x + sub_side // 2, y + sub_side // 2))

    # Add extra subsections centered between grid subsections
    # These are at centers of "double-squares"
    for i in range(n - 1):
        for j in range(n - 1):
            cx = (grid_centers[i * n + j][0] + grid_centers[(i + 1) * n + (j + 1)][0]) // 2
            cy = (grid_centers[i * n + j][1] + grid_centers[(i + 1) * n + (j + 1)][1]) // 2
            ul_x = cx - sub_side // 2
            ul_y = cy - sub_side // 2
            if 0 <= ul_x <= big_side - sub_side and 0 <= ul_y <= big_side - sub_side:
                positions.append([ul_x, ul_y])

    return positions
#print(get_subsections(64, 4))

    fF.KFoldCrossValidation(data,classifierFunction,numberOfSplits) 

def evaluate_subsections(
    dataset_reader: Callable[[str, Tuple[int, int], int], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    classifier: Callable[[np.ndarray, np.ndarray], List[int]],
    file_path: str,
    image_size: int,
    subsection_size: int,
    n_splits: int = 5
):
    positions = get_subsections(image_size, subsection_size)
    results = []

    for pos in positions:
        # Load data using the reader function with given subsection
        full, labels, imagesMatrix, imagesList = dataset_reader(
            file_path, subsection_origin=pos, subsection_size=subsection_size
        )

        # Combine label and features
        data = np.hstack([labels.reshape(-1, 1), imagesList.reshape(len(labels), -1)])

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        acc_scores = []

        for train_idx, test_idx in skf.split(data[:, 1:], data[:, 0]):
            train_set = data[train_idx]
            test_set = data[test_idx]

            y_pred = classifier(train_set, test_set)
            acc = accuracy_score(test_set[:, 0], y_pred)
            acc_scores.append(acc)

        avg_acc = np.mean(acc_scores)
        results.append((tuple(pos), avg_acc))

    return sorted(results, key=lambda x: -x[1])
def refine_top_subsections(
    dataset_reader: Callable[[str, Tuple[int, int], int], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    classifier: Callable[[np.ndarray, np.ndarray], List[int]],
    file_path: str,
    image_size: int,
    subsection_size: int,
    initial_results: List[Tuple[Tuple[int, int], float]],
    max_iter: int = 10,
    accuracy_tol: float = 1e-4,
    n_splits: int = 5,
    initial_step: int = 4,
    top_k: int = 3
):
    """
    Expands on refinement logic: instead of refining each subsection independently, expand
    the best step-wide neighbors collectively. At each iteration, the best `top_k` results
    from the pool of all visited and new neighbors are kept.
    """
    visited = set(tuple(pos) for pos, _ in initial_results)
    current_pool = initial_results[:top_k]

    def evaluate_position(x, y):
        pos = (x, y)
        if pos in visited:
            return None
        visited.add(pos)

        full, labels, _, imagesList = dataset_reader(file_path, subsection_origin=pos, subsection_size=subsection_size)
        data = np.hstack([labels.reshape(-1, 1), imagesList.reshape(len(labels), -1)])

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        acc_scores = []

        for train_idx, test_idx in skf.split(data[:, 1:], data[:, 0]):
            train_set = data[train_idx]
            test_set = data[test_idx]
            y_pred = classifier(train_set, test_set)
            acc_scores.append(accuracy_score(test_set[:, 0], y_pred))

        return pos, np.mean(acc_scores)

    step = initial_step
    improving = True
    best_avg_acc = np.mean([acc for _, acc in current_pool])

    while step >= 1 and improving:
        candidates = []

        for (x0, y0), _ in current_pool:
            for dx in [-step, 0, step]:
                for dy in [-step, 0, step]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x0 + dx, y0 + dy
                    if 0 <= nx <= image_size - subsection_size and 0 <= ny <= image_size - subsection_size:
                        result = evaluate_position(nx, ny)
                        if result:
                            candidates.append(result)

        # Combine current with new candidates and keep best top_k
        combined = current_pool + candidates
        combined.sort(key=lambda x: -x[1])
        new_pool = combined[:top_k]

        new_avg_acc = np.mean([acc for _, acc in new_pool])
        if new_avg_acc > best_avg_acc + accuracy_tol:
            current_pool = new_pool
            best_avg_acc = new_avg_acc
        else:
            if step == 1:
                improving = False
            else:
                step = step // 2

    return sorted(current_pool, key=lambda x: -x[1])[:3]

results = evaluate_subsections(
    dataset_reader=mnist_subsection,
    classifier=nF.MulticlassLogisticClassifier,
    file_path='Numbers.txt',
    image_size=16,
    subsection_size=16,
    n_splits=5
)

top_20 = results[:max(3, int(len(results) * 0.2))]

refined = refine_top_subsections(
    dataset_reader=mnist_subsection,
    classifier=nF.MulticlassLogisticClassifier,
    file_path='Numbers.txt',
    image_size=16,
    subsection_size=16,
    initial_results=top_20
)



side_sizes = [16, 12, 10, 8, 5, 4, 3, 2]
top_spots_by_size = {}

for size in side_sizes:
    # Step 1: Evaluate grid-based subsections
    results = evaluate_subsections(
        dataset_reader=mnist_subsection,
        classifier=nF.MulticlassLogisticClassifier,
        file_path='Numbers.txt',
        image_size=16,
        subsection_size=size
    )

    # Step 2: Take top 20% for refinement
    top_20 = results[:max(3, int(len(results) * 0.2))]


    # Step 3: Refine and store the top 3
    refined_top3 = refine_top_subsections(
        dataset_reader=mnist_subsection,
        classifier=nF.MulticlassLogisticClassifier,
        file_path='Numbers.txt',
        image_size=16,
        subsection_size=size,
        initial_results=top_20
    )

    top_spots_by_size[size] = refined_top3

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import random

def plot_top_subsections_per_size(
    top_spots_by_size, image_matrix_set, side_sizes, num_examples=4
):
    """
    For each block size, plot a separate figure showing multiple images
    with top 3 subsections overlaid and their accuracies listed beside each image.
    """
    rank_colors = ['red', 'blue', 'green']  # top 1, 2, 3

    for size in side_sizes:
        top_results = sorted(top_spots_by_size.get(size, []), key=lambda x: -x[1])[:3]
        random_indices = random.sample(range(len(image_matrix_set)), num_examples)

        fig, axs = plt.subplots(1, num_examples, figsize=(4 * num_examples + 2, 4))
        if num_examples == 1:
            axs = [axs]

        for col_idx, img_idx in enumerate(random_indices):
            ax = axs[col_idx]
            img = image_matrix_set[img_idx]
            ax.imshow(img, cmap='gray', vmin=0, vmax=1 if np.max(img) <= 1 else 255)
            ax.axis('off')

            for i, ((x, y), acc) in enumerate(top_results):
                color = rank_colors[i % len(rank_colors)]
                rect = patches.Rectangle((x, y), size, size, linewidth=2, edgecolor=color, facecolor='none')
                ax.add_patch(rect)

                # Accuracy label to the right of the image
                ax.text(
                    img.shape[1] + 2,  # x-position just outside the image
                    (i + 0.5) * (img.shape[0] / 3),  # y evenly spaced
                    f"#{i + 1}: {acc:.2f}",
                    color=color,
                    fontsize=10,
                    verticalalignment='center'
                )

        fig.suptitle(f"Top 3 Subsections (Size: {size}×{size})", fontsize=16)
        plt.tight_layout()
        plt.show()




_, _, image_matrix, _ = mnist_subsection('Numbers.txt')
reference_image = image_matrix[0:2]
_, _, image_matrix, _ = mnist_subsection('Numbers.txt')
plot_top_subsections_per_size(top_spots_by_size, image_matrix, side_sizes)
def plot_accuracy_vs_block_size(top_spots_by_size):
    block_sizes = sorted(top_spots_by_size.keys())
    best_accuracies = [max([acc for (_, acc) in top_spots_by_size[size]]) for size in block_sizes]

    plt.figure(figsize=(8, 4))
    plt.plot(block_sizes, best_accuracies, marker='o')
    plt.xlabel("Block size")
    plt.ylabel("Best accuracy")
    plt.title("Best accuracy vs. block size")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
plot_accuracy_vs_block_size(top_spots_by_size)
