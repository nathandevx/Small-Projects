"""
Data school's tutorial on how we can evaluate classification models

We'll be using train_test_split because it's often preferred due to it's speed and simplicity.

Which model evaluation metrics you use depend on if you are evaluating a classification or regression problem.

Regression: mean absolute error, mean squared error, root mean squared error
Classification: classification accuracy, and more


"""


import pandas as pd
from sklearn.preprocessing import OneHotEncoder, binarize, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, roc_curve
from matplotlib import pyplot as plt

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'flower']
dataset = pd.read_csv('bezdekIris.data', header=None, names=column_names)

x = dataset.iloc[:, 0:4]
y = dataset.iloc[:, 4]


"""
YAS. This is how you OneHotEncode a column

"""

ohe = OneHotEncoder()
ohe = ohe.fit(dataset.flower.values.reshape(-1, 1))
# This is the y portion of the data set in 0, 1, 2
ohe_labels = ohe.transform(dataset.flower.values.reshape(-1, 1)).toarray()
print(y)
print(ohe_labels)

"""
Moving on...
"""

# Splitting the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Training a logistic regression model on the training sets
logistic_regression = LogisticRegression()
logistic_regression.fit(x_train, y_train)

# Storing the predicted columns in y_pred
y_pred = logistic_regression.predict(x_test)


"""
Classification accuracy is not always helpful. For example, if the accuracy score is 80% and we have a y column of 0s 
and 1s and 80% of the y column's values are 1s (calculated by 'y_test.mean()') then our model is 'dumb' because it 
appears to be predicting '1' every time.

Null accuracy: accuracy that could be achieved by always predicting the most frequent class, which in our example is
'1'. 

Classification accuracy may be misleading because it does not tell us about the distribution of y_test. Like y_test
consisting of 80% '1' and 20% '0'.


# Calculate percentage of 1's
print(y_test.mean()

# Calculate percentage of 0's
print(1 - y_test.mean())

# Calculate null accuracy for binary classification (0's and 1's)
print(max(y_test.mean(), 1 - y_test.mean()))

# Calculate null accuracy for multi-class classification problems (multiple y values)
print(y_test.value_counts().head(1) / len(y_test))

# Find out if the model is getting 0's right but incorrectly classifying 1's and 0's
# This is addressed by the confusion matrix
print(f"True: {y_test.values[0:25]}")
print(f"Pred: {y_pred.values[0:25]}")


"""
# Finding the accuracy
accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
print(f"Accuracy: {accuracy}")

# Shows us how many times each label in the y column occurs in our dataset (class distribution)
# You could replace y_test with any dataframe object
print(f".value_counts(): \n{y_test.value_counts()}")


"""
The confusion matrix
0, 1, 2 = predictions

0, 0 = how many 0s were correctly classified as 0s
0, 1 = how many 0s the classifier incorrectly predicted as 1s
1, 1 = how many 1s were correctly classified as 1s
1, 2 = how many 1s were incorrectly classified as 2s


   0  1  2 = columns
0 11  0  0
1  0 12  1
2  0  0  6


"""

# If you pass y_test and y_pred in the reverse order, the confusion matrix will be reversed
conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
print(f"Confusion matrix: \n{conf_matrix}")

# Sorta like [rows, columns]
correct_0 = conf_matrix[0, 0]
incorrect_0_1 = conf_matrix[0, 1]
incorrect_0_2 = conf_matrix[0, 2]

correct_1 = conf_matrix[1, 1]
incorrect_1_0 = conf_matrix[1, 0]
incorrect_1_2 = conf_matrix[1, 2]


# Calculating the confusion matrix sensitivity
# The recall score basically divides the total number of incorrectly classified predictions by the total number of
# predictions, in this case it's like this: 29/30
# 'micro' tells recall_score that this is a multi-class problem
recall_score_ = recall_score(y_true=y_test, y_pred=y_pred, average='micro')
print(f"Recall score: {recall_score_}\n")

# Calculating the percentage of how often 1s were incorrectly classified as 0s or 2s
# Literally 1/13 (1 being total number of incorrect classifications) and (13 being total number of "1" predictions)
not_1s = (conf_matrix[1, 0] + conf_matrix[1, 2]) / (conf_matrix[1, 0] + conf_matrix[1, 1] + conf_matrix[1, 2])
print(f"Percentage of how often 1s were incorrectly classified as 0s or 2s: {not_1s:.4f}\n")

# predict_proba: each row... is a row.. and each column represents a response class (0, 1, 2)
# The classes
# 0      1      2
# The probability of the row being a certain class, these will add up to 1 (use Spyder to analyze predict_proba
# .0008  .1742  .8248
# You could do this predict_proba(x_test)[0:10, 1] to get the first 10 rows of column 1
predict_proba_ = logistic_regression.predict_proba(x_test)
print(f"Predict proba: \n{predict_proba_}\n")

# Plotting the predicted probabilities of class 1
# [rows, columns]
# This is helpful because let's say that a lot of observations in column 1 had a predict proba of 40%
# They don't get classified as 1s because the prediction probability isn't high enough. We could change this by using
# binarize
plt.hist(predict_proba_[:, 1], bins=10)
plt.title("Predicted probabilities of class 1")
plt.xlabel("predict_proba")
plt.ylabel("Frequency")
# plt.show()

# Predict the observation as a 1 if the predicted probability is greater than 30%
# It increases model sensitivity
binarizing = binarize(predict_proba_, threshold=0.3)
# This shows that the classifier predicted that the series of measurements can be classified as a 1 or a 2
# 0  1  2 = class predictions
# 0  1  1
print(f"Binarizing: \n{binarizing}\n")

# There is a way to search for an optimal threshold for binarize, by using the ROC curve

"""
ROC curve and AUC (Area under the Curve)

ROC curve is used to visualize the performance of a binary classifier (classifier that has 2 outputs)
The ROC curve shows you what 'threshold' you should use

"""

# **** And it was at this moment that I realized that the ROC curve is for binary classification problems only****
# specificity, sensitivity, threshold = roc_curve(y_test, y_pred)
#
# plt.plot(specificity, sensitivity)
# plt.xlim(0, 1)
# plt.ylim(0, 1)
# plt.title("ROC Curve")
# plt.xlabel("1 - specificity")
# plt.ylabel("Sensitivity")
# # plt.grid(True)
# plt.show()

































