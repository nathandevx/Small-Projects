"""
Getting started with sklearn with the Iris dataset

"""

from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plot

from warnings import filterwarnings
filterwarnings('ignore')

iris = load_iris()

x = iris.data
y = iris.target

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(x, y)

knn.predict([[3, 5, 4, 2]])

predict_this = knn.predict([[3, 5, 4, 3], [5, 4, 3, 2]])
print(f"(KNN) Prediction for first unknown iris: {predict_this[0]}, second: {predict_this[1]}")


# Using a different classifier
logistic_regression = LogisticRegression(solver='lbfgs', multi_class='auto')
logistic_regression.fit(x, y)
predict_this = logistic_regression.predict([[3, 5, 4, 3], [5, 4, 3, 2]])
print(f"(Logistic) Prediction for first unknown iris: {predict_this[0]}, second: {predict_this[1]}")


# Well.. which model is correct? We don't know
# We need to figure out which model most likely has the correct prediction by evaluating the accuracy score
y_pred_knn = knn.predict(x)
y_pred_logistic = logistic_regression.predict(x)


print(f"\nKNN accuracy score: {accuracy_score(y, y_pred_knn)}")
print(f"LogisticRegression accuracy score: {accuracy_score(y, y_pred_logistic)}")


# Splitting the data into training and testing sets
# The downside to using train test split is that the performance can change a lot depending on the data that's in the...
# training set versus the ones in the test set. And vice versa.
# The alternative model evaluation is called Kfold cross validation that overcomes this by repeating the train test...
# split process multiple times and averages the results
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=4)


# Predicting with logistic regression and its accuracy score
print(f"Shapes: x_train: {x_train.shape}, y_train: {y_train.shape}, x_test: {x_test.shape}, y_test: {y_test.shape}")
logistic_regression.fit(x_train, y_train)
y_pred_logistic = logistic_regression.predict(x_test)
print(f"Accuracy score for logistic regression: {accuracy_score(y_test, y_pred_logistic)}")


# How to find the best 'k' for KNN
k_values = {}
ranger = range(1, 91)
scores = []
for k in range(1, 91):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    pred = knn.predict(x_test)
    score = accuracy_score(y_test, pred)
    k_values[k] = score
    scores.append(score)

# for key, value in k_values.items():
    # print(key, value)


# Plotting the 'k' and the accuracy score
plot.plot(ranger, scores)
plot.xlabel("Value of k")
plot.ylabel("Accuracy")
plot.show()
