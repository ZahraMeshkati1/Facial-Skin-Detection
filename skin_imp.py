import numpy as np
import cv2
import matplotlib.pyplot as plt
from glob import glob
import time

# Load images
def load_images(image_paths):
    images = []
    for path in image_paths:
        image = cv2.imread(path)
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images.append(image)
        else:
            print(f"Warning: Could not read image at path {path}")
    return images

# Extract features from each image
def extract_features(images):
    features = []
    shape = None
    for image in images:
        shape = image.shape
        height, width, channels = shape
        for y in range(height):
            for x in range(width):
                b, g, r = image[y, x]
                rgb = (r, g, b)
                features.append(rgb)
    return np.array(features), shape

#euclidean_distance
def distance(X_train, x_test):
    return np.linalg.norm(X_train - x_test, axis=1)

# KNN for each test point
def knn_each(X_train, y_train, x_test, k=3):
    distances = distance(X_train, x_test)
    neighbors = np.argsort(distances)[:k]
    top_labels = y_train[neighbors]
    prediction = np.bincount(top_labels).argmax()
    return prediction

# KNN for the entire test set
def knn(X_train, y_train, X_test, k=3):
    y_pred = np.array([knn_each(X_train, y_train, x_test, k) for x_test in X_test])
    return y_pred


# Visualize segmentation results
def visualize_segmentation(image, labels, shape, test_indices):
    height, width, channels = shape
    segmented_image = np.zeros((height, width, channels), dtype=np.uint8)
    
    # Ensure labels are of the correct length
    if len(labels) != len(test_indices):
        raise ValueError("The number of labels must match the number of test indices.")
    
    # Initialize the entire segmented image with non-skin color
    for y in range(height):
        for x in range(width):
            segmented_image[y, x] = [0, 0, 0]  # Black for non-skin

    # Update the segmented image for test set pixels using the predictions
    idx = 0
    for i in test_indices:
        y = i // width
        x = i % width
        if labels[idx] == 1:  # Assuming label 1 is the skin label
            segmented_image[y, x] = [255, 0, 0]  # Red for skin
        idx += 1
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(segmented_image)
    plt.title('Segmentation Result')
    plt.axis('off')
    
    plt.show()


# Main execution
start_time = time.time()

image_paths = glob(r'D:\EL studies\term9\image3\*.jpg')
images = load_images(image_paths)

features, shape = extract_features(images)

# Read skin labels from file
with open('skin_labels.txt', 'r') as file:
    skin_labels = [int(line.strip()) for line in file]

true_labels = np.array(skin_labels)

# Split the data into training and testing sets
index = int(0.5 * len(features))
train_features, test_features = features[:index], features[index:]
train_labels, test_labels = true_labels[:index], true_labels[index:]

k = 1
y_pred = knn(train_features, train_labels, test_features, k)

end_time = time.time()

accuracy = np.mean(y_pred == test_labels)

# Get test set indices 
test_indices = range(index, len(features)) 
# Visualize segmentation result 
visualize_segmentation(images[0], y_pred, shape, test_indices)
print(f"Accuracy of custom KNN classifier: {accuracy}")
print(f"Run time: {end_time - start_time:.4f} seconds")
