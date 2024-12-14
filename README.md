# MNIST Classification with k-NN and SVM Using PCA

This project demonstrates the application of machine learning techniques on the MNIST dataset to classify handwritten digits. It includes preprocessing, dimensionality reduction using Principal Component Analysis (PCA), and classification using k-Nearest Neighbors (k-NN) and Support Vector Machines (SVM). The project is implemented in R and showcases the impact of PCA on classification performance.

---

## Introduction
The goal of this project is to classify handwritten digits from the MNIST dataset using k-NN and SVM, enhanced with PCA. The focus is on:
- Exploring PCA for dimensionality reduction.
- Comparing the performance of k-NN and SVM with and without PCA.
- Evaluating model accuracy and error rates.

---

## Dataset
The MNIST dataset is used, consisting of:
- **Training set:** 60,000 grayscale images (28x28 pixels each).
- **Test set:** 10,000 grayscale images.

The dataset is normalized, and PCA is applied to reduce dimensionality.

---

## Methods
1. **Data Preprocessing:**
   - Normalized pixel values to range [0, 1].
   - Checked for missing values and ensured data integrity.

2. **Dimensionality Reduction:**
   - Applied PCA to identify the number of components capturing 90% variance.
   - Visualized the data in reduced dimensions.

3. **Classification Algorithms:**
   - **k-NN:** Experimented with different values of `k` using the elbow method.
   - **SVM:** Used a radial kernel for classification.
   - Evaluated both models with and without PCA.

4. **Performance Metrics:**
   - Accuracy.
   - Confusion Matrix.
   - Kappa Statistic.
   - Error Rate.

---

## Results
- **k-NN (without PCA):** Accuracy = 90.15%, Optimal k = 3.
- **SVM (without PCA):** Accuracy = 88.42%.
- **PCA-enhanced k-NN:** Accuracy = 97.4%, Error Rate = 2.6%.
- **PCA-enhanced SVM:** Accuracy = 98.17%, Error Rate = 1.83%.

PCA significantly improved the performance of both models.

---

## Setup Instructions
   
Install the required R packages:
install.packages(c("class", "caret", "factoextra", "tidytable", "ggplot2", "e1071"))

Set the MNIST dataset files in the appropriate directory.

Run the R script to execute the analysis:
source("Machine learning R code.R")

## Key Findings
PCA reduced dimensionality while retaining essential features, improving model performance.
PCA-enhanced SVM achieved the highest accuracy (98.17%).
PCA-enhanced k-NN offers a simpler alternative with slightly lower accuracy but faster training.

## Future Work
Explore advanced models like Convolutional Neural Networks (CNNs) for further accuracy improvements.
Apply additional feature engineering techniques like LDA or hyperparameter tuning.
Test the models on other datasets to generalize the findings.

## Contributing
Contributions are welcome! Feel free to fork this repository and submit a pull request. For major changes, please open an issue first to discuss the proposed changes.

