"""
We use cross validation because train test split's testing accuracy isn't always right. For example, let's say we use
train test split and set the random state parameter to 0, we might get 95% accuracy. But if we set random state to 10
we may get 100% accuracy.

Cross validation solves this by averaging out the testing accuracy. We set 'k' in kfolds to how many partitions we
want. If we had a data set of 150 and we set k = 5 then the data set would be "folded" into 5 sets of 30. (2) We then
set the 1st fold as the testing set (30 observations (observations = rows)) and we combine the other folds into
training sets (120 observations). (3) Then we train the model on the training set and make predictions with the testing
set, then we calculate the testing accuracy.

Then we repeat steps (2) and (3) process 'k' times. Using a different fold as the testing set each time. Since k=5 in
our example, we would be repeating this process a total of 5 times. For example, during the second iteration, fold 2
would be the testing set and folds 1, 3, 4, 5 would be the training set.

Finally, the average testing accuracy (aka cross validation accuracy) is used as the estimate of out of sample accuracy.

Advantages of using train_test_split
- It runs 'k' times faster than KFold, since KFold basically repeats train_test_split 'k' times
- All you get back from KFold is the cross validation accuracy. Which makes it difficult to inspect the results using...
a ROC curve or confusion matrix on KFold

Cross validation tips
- k=10 is the go to because it produces the most reliable estimates of out of sample accuracy.
- When you use cross validation for classification problems, it's recommended that you use stratified sampling to...
create the folds. This means that each response class (predictions) should be represented with equal proportions in each
of the K folds. For example, if your data set has 2 response classes (cat and dog) and 20% of the observations were
cats, then each of your cross validation folds consist of approximately 20% cats. Sklearn's cross_val_score does this by
default.

"""


from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt


from warnings import filterwarnings
filterwarnings('ignore')


iris = load_iris()
x = iris.data
y = iris.target

"""
Example of how cross validation can help us with parameter tuning. The goal is to select the best tuning parameters, aka
"hyperparameters" for the KNN classifier 
"""

knn = KNeighborsClassifier(n_neighbors=5)
# Model, x data, y data, cv means 10 folds, we want to use classification accuracy as the evaluation metric
# cross_val_score has a similar function to KFolds. It splits the data into k folds (10), sets one fold as the testing
# set and the others as the training set. Calculates the testing accuracy, then it tests on fold 2, etc. When it's
# finished, it will return the 10 accuracy scores as a numpy array.
scores = cross_val_score(knn, x, y, cv=10, scoring='accuracy')
print(f"Scores: {scores}")
mean = scores.mean()
print(f"Mean: {mean}")

# Finding an optimal k value
k_range = range(1, 31)
k_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, x, y, cv=10, scoring='accuracy')
    k_scores.append(scores.mean())

print(f"k scores: {k_scores}")


# Plotting how the accuracy changes as 'k' changes
plt.plot(k_range, k_scores)
plt.xlabel('k_range')
plt.ylabel('k_scores')
# plt.show()

"""
It's generally recommended to choose the value, such as 'k' that produces the simplest model. For KNN, higher values of
'k' produce lower complexity models, so k=20 would be the best in our case. 

Now let's use cross validation to determine whether we should use KNN or logistic regression
"""

knn = KNeighborsClassifier(n_neighbors=20)
print(f"KNN accuracy: {cross_val_score(knn, x, y, cv=10, scoring='accuracy').mean()}")

logreg = LogisticRegression()
print(f"LogisticRegression accuracy: {cross_val_score(logreg, x, y, cv=10, scoring='accuracy').mean()}")

"""
As we can see, KNN is the best model for this task
"""
