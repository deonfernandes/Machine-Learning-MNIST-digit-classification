Handwritten Digit Classification Using Machine Learning
This project demonstrates solid machine learning concepts by classifying handwritten digits from the MNIST dataset using Principal Component Analysis (PCA) for dimensionality reduction and two classification models: k-Nearest Neighbors (k-NN) and Support Vector Machine (SVM).

Key Highlights:
Dataset: MNIST handwritten digits (28x28 grayscale images).
Preprocessing: Normalization and PCA for dimensionality reduction.
Algorithms:
k-NN: Explored with different values of k.
SVM: Implemented with and without PCA.
Evaluation: Comparison using accuracy, confusion matrices, and kappa statistics.
Project Features:
Preprocessing and Data Exploration:

Normalized pixel values to a 0-1 scale.
Checked and confirmed data integrity.
Dimensionality Reduction:

Applied PCA to visualize data in lower dimensions.
Retained 87 principal components to capture 90% variance.
Classification Models:

Implemented k-NN with optimized k values using the elbow method.
Applied SVM with radial kernel for robust classification.
Performance Evaluation:

Compared models before and after PCA.
Measured performance using confusion matrices, accuracy, and error rates.
Results:
Without PCA:

k-NN: Accuracy ~90.15% (k=3).
SVM: Accuracy ~88.42%.
With PCA:

PCA-based k-NN: Accuracy ~97.4%.
PCA-based SVM: Accuracy ~98.17%.
Repository Structure:
Machine Learning project Report.pdf: Detailed report of the project.
Machine learning R code.R: R script implementing the preprocessing, PCA, and classification algorithms.
README.md: Overview of the project (this file).
How to Run:
Install R and required libraries (caret, factoextra, tidytable, ggplot2, etc.).
Place the MNIST dataset files in the working directory (train-images-idx3-ubyte, train-labels-idx1-ubyte, etc.).
Run the R script:
R
Copy code
source("Machine learning R code.R")
Key Insights:
PCA enhances classification accuracy by reducing noise and optimizing feature space.
k-NN is simple but sensitive to the choice of k.
SVM is robust for high-dimensional data and slightly outperforms k-NN after PCA.
Future Improvements:
Experiment with deep learning models like CNNs for better feature extraction.
Optimize hyperparameters using cross-validation.
Incorporate additional dimensionality reduction techniques like LDA.
This project is a great example of applying classical machine learning techniques to image data and showcases the benefits of dimensionality reduction.
