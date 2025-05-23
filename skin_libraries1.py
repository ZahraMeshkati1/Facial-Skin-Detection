import numpy as np
import cv2
import matplotlib.pyplot as plt
from glob import glob
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score ,f1_score

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
            segmented_image[y, x] = [255, 255, 255]  # white for skin
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

# Split the data into training and testing sets
X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(features, skin_labels, range(len(skin_labels)), test_size=0.5, random_state=42)

# Perform KNN classification using scikit-learn
k = 1
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

end_time = time.time()

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
f1= f1_score(y_test,y_pred)

print(f"Accuracy of custom KNN classifier: {accuracy:.4f}")
print(f"Run time: {end_time - start_time:.4f} seconds")
print(f"F1 Score: {f1:.4f}")

# Visualize 
visualize_segmentation(images[0], y_pred, shape, test_indices)
