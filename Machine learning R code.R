library(class)
library(caret)
library(factoextra)
library(tidytable)
library(ggplot2)

# set the working directory to where the MNIST dataset files are located
setwd("C:/Desktop/Machine learning/MNIST-data")

# Define a function to load the MNIST dataset
load_mnist <- function() {
  # function to load image files
  load_image_file <- function(filename) {
    ret = list()  
    f = file(filename, 'rb') 
    readBin(f, 'integer', n=1, size=4, endian='big')  
    ret$n = readBin(f, 'integer', n=1, size=4, endian='big')  
    nrow = readBin(f, 'integer', n=1, size=4, endian='big')  
    ncol = readBin(f, 'integer', n=1, size=4, endian='big')  
    x = readBin(f, 'integer', n=ret$n*nrow*ncol, size=1, signed=F)  
    ret$x = matrix(x, ncol=nrow*ncol, byrow=T)  
    close(f)  
    ret  
  }
  # function to load label files
  load_label_file <- function(filename) {
    f = file(filename, 'rb') 
    readBin(f, 'integer', n=1, size=4, endian='big') 
    n = readBin(f, 'integer', n=1, size=4, endian='big')  
    y = readBin(f, 'integer', n=n, size=1, signed=F)  
    close(f) 
    y  # 
  }
  train_data <<- load_image_file('train-images-idx3-ubyte')  # Load training images
  test_data <<- load_image_file('t10k-images-idx3-ubyte')  # Load testing images
  
  train_data$y <<- load_label_file('train-labels-idx1-ubyte')  # Load training labels
  test_data$y <<- load_label_file('t10k-labels-idx1-ubyte')  # Load testing labels
  
  return(
    list(
      train = train_data,
      test = test_data
    )
  )  # Returns a list containing both training and testing datasets
}

# function to display a digit
show_digit <- function(arr784, col=gray(12:1/12), ...) {
  image(matrix(arr784, nrow=28)[,28:1], col=col, ...)  # to reate an image plot of the digit
}

# Load data
mnist <- load_mnist()
str(mnist)  # Display the structure of the MNIST dataset

#######################################

# Display the first 25 MNIST digits along with their labels
labels <- paste(mnist$train$y[1:25], collapse = ", ")  
par(mfrow=c(5,5), mar=c(0.1,0.1,0.1,0.1))  
for(i in 1:25) show_digit(mnist$train$x[i,], axes=F)  

# Display summary statistics for the pixel values and labels in the training data
summary(train_data$x)
summary(train_data$y)

# Display the dimensions of the image and label datasets
dim(train_data$x)
dim(test_data$x)

# Visualize the first image from the training dataset
train_data$x[1,]
show_digit(train_data$x[1,])

# Plotting pairs of selected pixel values to explore relationships
pairs(test_data$x[,200:205],  col=c("red","green","blue","aquamarine","burlywood","darkmagenta","chartreuse","yellow","chocolate","darkolivegreen")[test_data$y+1])

# Compute and display the correlation matrix for selected pixels
correlation_matrix <- cor(test_data$x[, 200:205])

# Compute and display the covariance matrix for the training image data
C <- cov(train_data$x)
image(C)  # Visualize the covariance matrix as an image

# Count the number of observations in the training and test datasets
NROW(train_data$x)
NROW(test_data$x)

# Check for any missing values in the dataset
any(is.na(train_data$x))
any(is.na(train_data$y))
any(is.na(test_data$x))
any(is.na(test_data$y))

# Normalize the pixel values of the datasets to range from 0 to 1
x_train <- train_data$x / 255
x_test <- test_data$x / 255
y_train <- train_data$y
y_test <- test_data$y

# Create a subset of the original training dataset to 3% of it
Find_k <- createDataPartition(y_train, p=0.03, list=FALSE, times=1)
train_kx <- x_train[Find_k,]
train_ky <- y_train[Find_k]

# check the number of rows after reduction
NROW(train_kx)
NROW(train_ky)

#########################


# Apply PCA on without scaling since data is already normalized (normalized training data)
pca_train <- prcomp(x_train, center=TRUE, scale.=FALSE)  


##########################

# Visualizing the explained variance by each principal component
var_explained <- summary(pca_train)$importance[2, ]  
plot(var_explained, type = 'b', main = "Explained Variance by Principal Components", xlab = "Principal Component", ylab = "Proportion of Variance Explained")

# Heuristics rule to Determine the number of components needed to capture at least 80% variance
cum_var_explained <- cumsum(var_explained) 
num_components <- which(cum_var_explained >= 0.90)[1]  # for the number of components that explain at least 90% variance
cat("Number of principal components: ", num_components, "\n")  
#Output:Number of principal components chosen:  87 

########################


# Data transformation and visualization using selected principal components
train_final = as.matrix(x_train) %*% pca_train$rotation[,1:87]  # Project the training data onto the first 44 principal components
head(train_final)  # Display the first few rows of the transformed data

train_final_df <- as.data.frame(train_final)  # Convert matrix to data frame
train_final_df$Label <- as.factor(y_train)   # Add labels as a factor

# Let's check the structure to ensure it's correct
str(train_final_df)

# Advanced PCA-based plots using ggplot2
# Basic ggplot call with minimal aesthetic mappings
ggplot(train_final_df, aes(x = PC1, y = PC2, color = Label)) +
  geom_point() +
  labs(x= "PC1",y= "PC2") + ggtitle("PCA for 1st and 2nd components")

# Visualize the 3rd and 4th principal components
ggplot(train_final_df, aes(PC3,PC4,color=Label)) + geom_point() +
  labs(x= "PC3",y= "PC4") + ggtitle("PCA for 3rd and 4th components")

##############################

# Reconstruction of MNIST digits with varying numbers of principal components
reconstruction_1PC = t(t(pca_train$x[,1:1] %*%  # Reconstruct using 1 principal component
                           t(pca_train$rotation[,1:1])) +
                         pca_train$center)
reconstruction_50PC = t(t(pca_train$x[,1:50] %*%  # Reconstruct using 44 principal components
                            t(pca_train$rotation[,1:50])) +
                          pca_train$center)
reconstruction_87PC = t(t(pca_train$x[,1:87] %*%  # Reconstruct using 150 principal components
                            t(pca_train$rotation[,1:87])) +
                          pca_train$center)
reconstruction_150PC = t(t(pca_train$x[,1:150] %*%  # Reconstruct using all 784 principal components
                             t(pca_train$rotation[,1:150])) +
                           pca_train$center)
# Visualize reconstructed digits using different numbers of components
par(mfrow=c(2,2))  # Set plotting area to 2x2 grid
show_digit(reconstruction_1PC[340,], main="1 Component")  # Show digit reconstructed with 1 PC
show_digit(reconstruction_50PC[340,], main="50 Components")  # Show digit reconstructed with 44 PCs
show_digit(reconstruction_87PC[340,], main="87 Components")  # Show digit reconstructed with 150 PCs
show_digit(reconstruction_150PC[340,], main="150 Components")  # Show digit reconstructed with all PCs



####################################################
#kNN
# Define a range of k values from 0 to 42, based on the square root of the number of observations(1764), choosing odd values to avoid ties
error_train_full <- replicate(0,42)
for(k in seq(1, 42, by=2)) {
  predictions <- knn(train=train_kx, test=train_kx, cl=train_ky, k)
  error_train_full[k] <- 1-mean(predictions==train_ky)  # Calculate training error
}
error_train_full <- unlist(error_train_full, use.names=FALSE)

error_test_full <- replicate(0,42)
for(k in seq(1, 42, by=2)) {
  predictions <- knn(train=train_kx, test=x_test, cl=train_ky, k)
  error_test_full[k] <- 1-mean(predictions==y_test)  # Calculate test error
}
error_test_full <- unlist(error_test_full, use.names=FALSE)

# Plot the  errors for training and test sets to find the optimal k value using the elbow method
png("k_value.png", height=800, width=1000)
plot(error_train_full, type="o", ylim=c(0,0.11), col="blue", xlab="K values", ylab="Misclassification errors", main="Test vs train error for varying k values without PCA")
lines(error_test_full, type="o", col="red")
legend("topright", legend=c("Training error", "Test error"), col=c("blue","red"), lty=1:1)
dev.off()

#####################################


# kNN classification for different values of k

# For K=3
predict_noPCA_3 <- knn(train= train_kx, test = x_test, cl= train_ky, k=3, prob=TRUE)  # Perform k-NN where k=3
cm_noPCA_3 <- confusionMatrix(as.factor(predict_noPCA_3), as.factor(y_test))  # Calculate the confusion matrix
print(cm_noPCA_3)  # Print the confusion matrix for k=3

# For K=5
predict_noPCA_5 <- knn(train= train_kx, test = x_test, cl= train_ky, k=5, prob=TRUE)
cm_noPCA_5 <- confusionMatrix(as.factor(predict_noPCA_5), as.factor(y_test))
print(cm_noPCA_5) 

##########################

# PCA Section: Principal Component Analysis (PCA) implementation and analysis
min(x_train)  # Find the minimum value in the training set
max(x_train)  # Find the maximum value in the training set
#0 and 1 indicates that the normalization was done correctly

# Compute covariance matrix of the training data
cov_train <- cov(x_train)  # Calculate the covariance matrix

#########################

# Select a specific number of principal components for further analysis
num_components <- 87  # Set number of components to use

# Transform both training and testing data using the selected principal components
train_final <- as.matrix(x_train) %*% pca_train$rotation[, 1:num_components] 
test_final <- x_test %*% pca_train$rotation[, 1:num_components]  

#### Train k-NN classifier using PCA-transformed data ###

predict_pca <- knn(train = train_final, test = test_final, cl = y_train, k = 3) 

# Evaluate classifier performance
conf_matrix <- confusionMatrix(factor(predict_pca, levels = levels(factor(y_train))), factor(y_test)) 

# Print confusion matrix and visualize eigenvalues
print(conf_matrix)  # Display confusion matrix

# Plot scree plot 
fviz_eig(pca_train, addlabels = TRUE)  

######################

#To run SVM model

svm_model <- svm(train_kx, as.factor(train_ky), type = 'C-classification', kernel = 'radial')

svm_predictions <- predict(svm_model, x_test)

#Confusion matrix
svm_conf_matrix <- confusionMatrix(as.factor(svm_predictions), as.factor(y_test))
print(svm_conf_matrix)

#PCA on training data
pca_train <- prcomp(x_train, center = TRUE, scale. = FALSE)

# Retrain SVM on PCA-transformed Data
svm_pca_model <- svm(train_final, as.factor(y_train), type = 'C-classification', kernel = 'radial')

# Evaluate SVM on PCA Data
svm_pca_predictions <- predict(svm_pca_model, test_final)
svm_pca_conf_matrix <- confusionMatrix(as.factor(svm_pca_predictions), as.factor(y_test))
print(svm_pca_conf_matrix)


#calculating error rates
# predictions for the PCA-transformed SVM model
svm_pca_predictions <- predict(svm_pca_model, test_final)

#  error rate for the PCA SVM 
error_rate_svm_pca <- mean(svm_pca_predictions != y_test)
print(paste("Error rate for PCA-transformed SVM: ", error_rate_svm_pca))

#  error rate for the PCA k-NN 
error_rate_knn_pca <- mean(predict_pca != y_test)
print(paste("Error rate for PCA-transformed k-NN: ", error_rate_knn_pca))
