import numpy as np
import matplotlib.pyplot as plt

def FeedFoorward(data,classiferFunction,errorThreshold,maxNumberOfFeatures):
    
    maxLoops=min(maxNumberOfFeatures,data.shape[1]-1)
    dimensions=np.arange(1,data.shape[1])
    bestCVerror=10
    iterations=1

    while iterations <=maxLoops and bestCVerror>errorThreshold:

        selectedFeatures=[]
        bestfeature=0

        for feature in dimensions:

            CVerror=KFoldCrossValidation(data[[selectedFeatures,feature]],numberOfSplits=5)

            if CVerror < bestCVerror:

                bestfeature=feature
                bestCVerror=CVerror

        selectedFeatures.append(bestfeature)
    return selectedFeatures

def KFoldCrossValidation(data,classifierFunction,numberOfSplits):
        
    
    splitData=np.array_split(data,numberOfSplits,axis=1)
    accuracyList=[]

    for iteration in range(numberOfSplits):
        trainingData = np.vstack(splitData[np.arange(len(splitData))!=iteration])

        testData=splitData[iteration]

        accuracy=classifierFunction(trainingData,testData)

        accuracyList.append(accuracy)
    meanError=np.mean(accuracyList)
    return meanError

def read_mnist_txt(file_path):
    images = []
    labels = []
    
    with open(file_path, 'r') as file:
        for line in file:
            # Skip empty lines
            if not line.strip():
                continue
                
            # Split the line into components
            parts = line.strip().split()
            
            # The first part is the image number in quotes (e.g., "1"), 
            # the second is the label, and the rest are pixel values
            image_num = parts[0].strip('"')
            label = int(parts[1])
            pixels = list(map(float, parts[2:]))
            
            # Convert pixel values to a 16x16 numpy array
            # The values are between -1 and 1, so we normalize to 0-1 for display
            pixel_array = np.array(pixels).reshape(16, 16)
            normalized_array = (pixel_array + 1) / 2  # Scale from [-1, 1] to [0, 1]
            
            images.append(normalized_array)
            labels.append(label)
    
    return images, labels    

# import render_image

def display_images_from_rows(array, cmap='gray'):
    """
    Display each row of a 2D numpy array as a square image.

    Parameters:
    - array: np.ndarray, shape (n_images, image_size), each row is a flattened square image.
    - cmap: colormap to use for displaying the images (default: 'gray')
    """
    num_images = array.shape[0]
    
    
    for i in range(num_images):
        img_flat = array[i]
        side = int(np.sqrt(len(img_flat)))
        if side * side != len(img_flat):
            raise ValueError(f"Row {i} cannot be reshaped into a square image.")
        
        img = img_flat.reshape((side, side))
        plt.figure()
        plt.imshow(img, cmap=cmap, vmin=0, vmax=255)
        plt.title(f"Image {i}")
        plt.axis('off')
    plt.show()
import render_image
catsAndDogs = np.loadtxt('catdogdata.txt')
numbers=np.loadtxt('Numbers.txt')
print(catsAndDogs.shape,numbers.shape)
print(catsAndDogs,numbers)
print(catsAndDogs[0,1:])
render_image.display_images_from_rows(catsAndDogs[[98,99],1:])