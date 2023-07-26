"""
This class makes predictions based on the SERPS data. (GetData class)

"""

# Preprocessing tools
from sklearn.preprocessing import StandardScaler
# Model selection tools
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_val_predict
# Prediction metrics
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
# Machine learning algorithm
from sklearn.neighbors import KNeighborsClassifier
# PCA
from sklearn.decomposition import PCA

# Graphing tools
from matplotlib import pyplot as plt
import seaborn as sns

# Data processing tools
import pandas as pd
import numpy as np

"""
Loading and cleaning the data

"""

# Loading the dataset
dataset = pd.read_csv(filepath_or_buffer='serps_data_original.csv', header=0)

# Column names
# There are now 11 columns
dataset = dataset.iloc[:, [2, 3, 5, 6, 8, 9, 11, 12, 13, 14, 15]]
# Replacing the spaces in the columns names with '_'
dataset.columns = dataset.columns.str.replace(' ', '_')

# Finding all unique values in each feature column
# for column in dataset.columns.values:
#     print(f"{column}:\n{dataset[column].value_counts()}\n\n")

# Replacing 'unscrapable' and other values that basically mean 'NaN' with 'NaN'
dataset = dataset.replace('unscrapable', np.NaN)
dataset = dataset.replace('not 200', np.NaN)
dataset = dataset.replace('nt 200', np.NaN)
dataset = dataset.replace('no h1 tag', np.NaN)
dataset.dropna(inplace=True)

# Replacing values in 'Domain_length' <= 10 to 1, <= 20 to 2
dataset.loc[dataset.Domain_length <= 10, 'Domain_length'] = 0
dataset.loc[dataset.Domain_length > 10, 'Domain_length'] = 1

# Replacing values in 'Keywords_in_domain'
dataset.loc[dataset.Keywords_in_domain == 0, 'Keywords_in_domain'] = 0
dataset.loc[dataset.Keywords_in_domain > 0, 'Keywords_in_domain'] = 1


# Replacing values in 'Permalink_length'
dataset.loc[dataset.Permalink_length <= 30, 'Permalink_length'] = 0
dataset.loc[(dataset.Permalink_length > 30) & (dataset.Permalink_length <= 50), 'Permalink_length'] = 1
dataset.loc[dataset.Permalink_length > 50, 'Permalink_length'] = 2


# Replacing values in 'Keywords_in_permalink'
dataset.loc[dataset.Keywords_in_permalink <= 1, 'Keywords_in_permalink'] = 0
dataset.loc[(dataset.Keywords_in_permalink == 2) | (dataset.Keywords_in_permalink == 3), 'Keywords_in_permalink'] = 1
dataset.loc[dataset.Keywords_in_permalink == 4, 'Keywords_in_permalink'] = 2


# Replacing values in 'H1_tag_length'
# This converts string datatypes to floats
dataset.H1_tag_length = pd.to_numeric(dataset.H1_tag_length)
dataset.loc[dataset.H1_tag_length < 20, 'H1_tag_length'] = 0
dataset.loc[(dataset.H1_tag_length >= 20) & (dataset.H1_tag_length <= 30), 'H1_tag_length'] = 1
dataset.loc[dataset.H1_tag_length > 30, 'H1_tag_length'] = 2


# Replacing values in 'Keywords_in_H1'
dataset.Keywords_in_H1 = pd.to_numeric(dataset.Keywords_in_H1)
dataset.loc[dataset.Keywords_in_H1 <= 1, 'Keywords_in_H1'] = 0
dataset.loc[(dataset.Keywords_in_H1 == 2) | (dataset.Keywords_in_H1 == 3), 'Keywords_in_H1'] = 1
dataset.loc[dataset.Keywords_in_H1 == 4, 'Keywords_in_H1'] = 2


# Replacing values in 'Title_tag_length'
dataset.Title_tag_length = pd.to_numeric(dataset.Title_tag_length)
dataset.loc[dataset.Title_tag_length <= 50, 'Title_tag_length'] = 0
dataset.loc[(dataset.Title_tag_length > 50) & (dataset.Title_tag_length <= 70), 'Title_tag_length'] = 1
dataset.loc[dataset.Title_tag_length > 70, 'Title_tag_length'] = 2


# Replacing values in 'Keywords_in_title'
dataset.Keywords_in_title = pd.to_numeric(dataset.Keywords_in_title)
dataset.loc[dataset.Keywords_in_title == 1, 'Keywords_in_title'] = 0
dataset.loc[(dataset.Keywords_in_title == 2) | (dataset.Keywords_in_title == 3), 'Keywords_in_title'] = 1
dataset.loc[dataset.Keywords_in_title == 4, 'Keywords_in_title'] = 2


# Replacing values in 'Keywords_on_page'
dataset.Keywords_on_page = pd.to_numeric(dataset.Keywords_on_page)
dataset.loc[dataset.Keywords_on_page <= 80, 'Keywords_on_page'] = 0
dataset.loc[(dataset.Keywords_on_page > 80) & (dataset.Keywords_on_page <= 200), 'Keywords_on_page'] = 1
dataset.loc[dataset.Keywords_on_page > 200, 'Keywords_on_page'] = 2


# Replacing values in 'Character_count'
dataset.Character_count = pd.to_numeric(dataset.Character_count)
dataset.loc[dataset.Character_count <= 50000, 'Character_count'] = 0
dataset.loc[(dataset.Character_count > 50000) & (dataset.Character_count <= 100000), 'Character_count'] = 1
dataset.loc[dataset.Character_count > 100000, 'Character_count'] = 2


# Replacing values in 'Rank_position'
dataset.loc[dataset.Rank_position <= 10, 'Rank_position'] = 0
dataset.loc[(dataset.Rank_position > 10) & (dataset.Rank_position <= 20), 'Rank_position'] = 1
dataset.loc[(dataset.Rank_position > 20) & (dataset.Rank_position <= 30), 'Rank_position'] = 2

# Finding the unique values in each column
# for column in dataset.columns.values:
#     print(f"{column}:\n{dataset[column].value_counts()}\n\n")


"""
Exploring the data

"""

# print(f"Total rows: {dataset.shape[0]}, total columns: {dataset.shape[1]}\n\n")
# print(f"{dataset.info()}\n\n")
# Visualizing missing values in the dataset
# sns.heatmap(dataset.isnull(), cbar=False)
# plt.show()


"""
Defining the input and output variables.

The input variables the features (columns) of the dataset and the output variables are what we're trying to predict. 

"""

x = dataset.iloc[:, :10]
y = dataset.iloc[:, 10]
# Printing the input variable columns (x)
# print(x.columns.values)
# Printing the output variable columns (y)
# print(y.name)


"""
-Finding the best parameters of k-nearest neighbors (KNN)- 

k-nearest neighbors explained:
- Finds the euclidean distance from the given point (the prediction point) to every other point.
- Then assuming k = 3, gets the 3 points with the smallest distance from the prediction point. 
- Then decides which class the prediction point should belong to

GridSearchCV process: 
- Given a parameter grid
- For each parameter 
-   Test the model on that parameter
-   Get the score
- It will repeat this 10 times with kfold and then get the average accuracy (The cross validated accuracy)

"""

# KNN classifier
knn = KNeighborsClassifier(p=2, metric='euclidean')
parameter_grid = {
    'n_neighbors': list(range(1, 21)),
    'weights': ['uniform', 'distance'],
}
# iid=False to avoid a 'DeprecationWarning'
grid = GridSearchCV(estimator=knn, param_grid=parameter_grid, cv=10, scoring='accuracy', iid=False)
grid.fit(x, y)
print(f"KNN best parameters: {grid.best_params_}\n\n")


"""
-The confusion matrix-

The confusion matrix shows us how many correct and incorrect predictions our model got. 

"""

# Confusion matrix
y_pred = cross_val_predict(estimator=knn, X=x, y=y, cv=10)
conf_matrix = confusion_matrix(y_true=y, y_pred=y_pred)
print(f"Confusion matrix: \n{conf_matrix}\n\n")


"""
-Calculating the cross validation score-

Cross validation is a much better way of calculating the accuracy as opposed to 'accuracy score'.

cross_val_score process:
- It splits the data into 'cv' or 'k' sets (divides the total number of rows by 'cv')
- all_accuracy_scores_list = []
- For each set in the sets
-   One set will be used for testing the model and the others will be used for training the model
-   Find the accuracy of the model
-   Append that accuracy to 'all_accuracy_scores_list'
- After each set has had a chance to be a testing set, the loop breaks
- The mean of all the accuracy scores is then taken. 
- Andddd that is the cross validation score

Why do this? Why not train_test_split():
- 'train_test_split' only tests the model on 20% of the data (if we set the test split to 0.2)
This can massively skew the accuracy and every other model metric (recall score, precision, etc) because
what if the test split only consists of one prediction label? This would mean that the model is only basing it's 
prediction accuracy on that one prediction label.
- it sounds cool lol
- It also allows you to use all your data (totally didn't google this xD)

"""

# Cross validation
knn = KNeighborsClassifier(n_neighbors=6, weights='distance')
scores = cross_val_score(estimator=knn, X=x, y=y, cv=10)
print(f"Cross validation score KNN: {scores.mean()*100:.2f}%\n\n")


"""
-Calculating null accuracy-

Null accuracy is the accuracy our model could achieve if we predicted the same prediction label over and over.
Null accuracy is calculated by taking the total number of observations (rows) and dividing that by the total
number of prediction labels.

For example: 
- let's say '0' has a null accuracy of 90%
- and '1' has a null accuracy of 10%

If our model predicted '0' 90% of the time, well um it would be right 90% of the time because 90% of the data is '0'.
So if our model has a 90% accuracy score, that might not be what you want.

"""

# Null accuracy
null_accuracy = y.value_counts() / len(y)
print("Null accuracy: ")
for index, score in enumerate(null_accuracy[::-1]):
    print(f"{index}: {score*100:.2f}%")
print("\n")


"""
-Calculating the accuracy score-

Usually cross validation accuracy should be used instead of accuracy score. 
Accuracy score is calculated by taking the total number of correct predictions divided by the total number of 
observations (rows). 

"""

# Accuracy score
acc_score = accuracy_score(y_true=y, y_pred=y_pred)
print(f"Accuracy score: {acc_score*100:.2f}%\n\n")


"""
-Manually calculating the recall score-

Recall score explained:
- What percentage of 0s in my dataset were correctly classified as 0s?
- I calculated this by taking the total correctly classified 0s and dividing it by the total number of 0s 
in the dataset. (You can replace '0' with any prediction label)  

"""

print("----- Manually calculating the recall score -----")
# How accurate the model is at predicting 0s as 0s
score = conf_matrix[0][0] / (conf_matrix[0][0] + conf_matrix[0][1] + conf_matrix[0][2])
print(f"Score for predicting 0s as 0s 44/(44 + 20 + 13): {score*100:.2f}%\n")


# How accurate the model is at predicting 1s as 1s
score = conf_matrix[1][1] / (conf_matrix[1][0] + conf_matrix[1][1] + conf_matrix[1][2])
print(f"Score for predicting 1s as 1s 47/(21 + 47 + 19): {score*100:.2f}%\n")


# How accurate the model is at predicting 2s as 2s
score = conf_matrix[2][2] / (conf_matrix[2][0] + conf_matrix[2][1] + conf_matrix[2][2])
print(f"Score for predicting 2s as 2s 27/(22 + 40 + 27): {score*100:.2f}%\n\n")


"""
Manually calculating the precision score.

Precision score explained:
- When the model predicts 0, what percentage of times is it accurate?
- I calculated this by dividing the total number of correctly classified 0s
by the total number of times the model predicts '0'

"""

print("----- Manually calculating the precision score -----")
score = conf_matrix[0][0] / (conf_matrix[0][0] + conf_matrix[1][0] + conf_matrix[2][0])
print(f"Score for predicting 0s as 0s 44/(44 + 21 + 22): {score*100:.2f}%\n")

score = conf_matrix[1][1] / (conf_matrix[1][1] + conf_matrix[0][1] + conf_matrix[2][1])
print(f"Score for predicting 1s as 1s 47/(20 + 47 + 40): {score*100:.2f}%\n")

score = conf_matrix[2][2] / (conf_matrix[2][2] + conf_matrix[0][2] + conf_matrix[1][2])
print(f"Score for predicting 1s as 1s 27/(13 + 19 + 27): {score*100:.2f}%\n\n")


"""
-Classification report-

- Shows the precision, recall, f1-score, and support.
- Support is the number of prediction labels.
- f1-score is the average of precision and recall ((precision + recall)/2)
The f1-score can be misleading because if precision is .1 and recall is .9, then the f1-score would be .5,
.5 does not accurately represent the mean of precision and recall. 

"""

report = classification_report(y, y_pred)
print(f"Classification report:\n{report}\n\n")


"""
Predicting data

"""

# Fitting the KNN model with the data
knn.fit(X=x, y=y)

# Predicting data
# Features = ['Domain_length' 'Keywords_in_domain' 'Permalink_length'
#  'Keywords_in_permalink' 'H1_tag_length' 'Keywords_in_H1'
#  'Title_tag_length' 'Keywords_in_title' 'Keywords_on_page'
#  'Character_count']
data = [5, 2, 50, 5, 20, 3, 30, 4, 80, 27000]

print("*Predicting data:* \n")
print(f"Features: {data}\n")
print(f"Prediction: {knn.predict([data])}\n")
print("1 = first page (of Google), 2 = second page, 3 = third page")


"""
-PCA-

I did not use PCA for this project but if I wanted to use it,
this is how I would prepare the 'x' (features) data.

I don't entirely understand how PCA works, but in a nutshell, it reduces
the dimensions (features aka columns) of the data.

"""

# Scaling the data is necessary for PCA
scalar = StandardScaler()
scalar.fit(X=x)
x_scaled = scalar.transform(X=x)

# Using PCA
pca = PCA(n_components=2)
pca.fit(X=x)
x_pca = pca.transform(x)

feature1 = x_pca[:, 0]
feature2 = x_pca[:, 1]

# Plotting the data
# sns.scatterplot(data=x_pca)
# plt.show()
