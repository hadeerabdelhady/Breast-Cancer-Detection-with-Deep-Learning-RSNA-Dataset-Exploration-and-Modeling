# Breast-Cancer-Detection-with-Deep-Learning-RSNA-Dataset-Exploration-and-Modeling
Breast Cancer Detection with Deep Learning
RSNA Breast Cancer Detection Dataset | TensorFlow & Keras

Overview
This notebook presents an end-to-end deep learning pipeline for binary classification of breast cancer using the RSNA Breast Cancer Detection dataset. The primary objective is to identify whether a given breast image indicates cancer (1) or not (0). The solution uses TensorFlow and Keras for model building and evaluation.

Libraries Used
numpy and pandas: For numerical operations and structured data manipulation.

matplotlib.pyplot: For visualizing images and model performance.

os: To handle directory paths and image file access.

tensorflow.keras: Used for building, training, and evaluating the deep learning model.

PIL.Image: For image manipulation and loading.

sklearn: For data splitting and evaluation metrics like the confusion matrix.

Dataset
Path: /kaggle/input/rsna-bcd-1024x512-preprocessed/train_images

CSV File: train.csv

Content:

image_id: Unique image identifiers.

cancer: Target label (1 for cancer, 0 for no cancer).

The notebook reads this CSV file using pandas.read_csv() and constructs image paths dynamically for training.

Data Preprocessing
Image paths and their corresponding labels are extracted and stored in lists.

Images are assumed to be stored in .png format under the specified directory.

A match["cancer"].values[0] expression is used to map each image ID to its cancer label from the DataFrame.

Train-Test Split
Data is split into training and testing sets using train_test_split() from scikit-learn, maintaining stratified sampling for balanced class distribution.

Data Augmentation
ImageDataGenerator is used to:

Normalize image pixel values.

Augment training data with techniques like zoom, shift, and flips to improve generalization.

Model Architecture
A Sequential Neural Network built using Keras:

Fully connected Dense layers.

Final Dense layer with sigmoid activation for binary classification.

Uses Adam optimizer and binary_crossentropy loss.

Training and Evaluation
The model is compiled and trained using .fit().

Validation performance is tracked across epochs.

Accuracy and loss curves are plotted using matplotlib.

A confusion matrix and classification report (precision, recall, f1-score) are printed using sklearn.metrics.

Performance Metrics
Accuracy

Precision

Recall

F1 Score

Confusion Matrix

These metrics give a comprehensive overview of model performance, especially important for medical datasets where false negatives can be critical.

Future Improvements
Add convolutional layers (CNN) for better image feature extraction.

Use pre-trained models (e.g., EfficientNet, ResNet).

Apply k-fold cross-validation.

Improve image preprocessing (e.g., resizing, normalization consistency).

Conclusion
This project demonstrates a foundational deep learning approach to medical image classification. It offers a baseline that can be expanded with more advanced CNN architectures, better preprocessing, and robust evaluation strategies.

