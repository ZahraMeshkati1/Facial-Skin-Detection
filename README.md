# Facial Skin Pixel Detection

### Dataset Preparation  
To detect facial skin pixels, we first need a dataset where each pixel is labeled as either `skin` (1) or `skin_none` (0).

**File Name:** `skin_labels.txt`

### Feature Extraction  
To begin the implementation, we load the target image using the `cv_open` library.  
For learning, we need a feature array. In the `features_extract` function, each pixel is stored as a tuple in the `features` array.  
Each image has three dimensions: **height**, **width**, and **channels**. Since the image format is **RGB**, the number of channels is `3`.  
Each pixel has `r`, `g`, and `b` values, which we extract and store in the `features` list as tuples.  
The final feature array has a shape of `(y * x, 3)`, representing the tuples of extracted pixel features.

### Classification Using KNN  
Now, we are ready to implement the classification algorithm, **K-Nearest Neighbors (KNN)**.  
Before training, we split the dataset into two subsets: **train** and **test**.  
We then call the `knn` function, which outputs a list of predicted labels for `features_test`.

For each point in `test_X`, the function `each_knn` is called.  
In this function:
- The distance between the target point and all points in `train_X` is computed using the `distance` function.
- The distances are returned as an array.
- We sort the distances using `numpy.argsort`.
- The labels of the `k` closest neighbors are stored in `labels_top`.
- The most frequent label among the `k` neighbors is assigned as the predicted label for the test point.

This process is repeated for all points in `test_X`, and the predicted labels are stored in `pred_y`.

### Model Evaluation  
To evaluate the model, we use **accuracy** as the metric.  
Accuracy is computed as the number of correctly predicted labels divided by the total number of labels.  
Since both classes in the dataset are balanced, accuracy is an appropriate evaluation metric.

Additionally, the `segmentation_visualize` function is used to render the image with predicted labels.  
- Initially, all pixels are assigned the default color of `skin_none` (black).  
- Then, only **test** pixels are colored based on their predicted labels.  
- **Train** pixels remain black.

