from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from matplotlib import pyplot as plt

iris = load_iris()
x = iris.data
y = iris.target

knn = KNeighborsClassifier(n_neighbors=5)
scores = cross_val_score(estimator=knn, X=x, y=y, cv=10, scoring='accuracy')

"""
Finding the best 'k' parameter using a for loop
"""
print('\n' * 3)


k_range = range(1, 31)
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(estimator=knn, X=x, y=y, cv=10, scoring='accuracy').mean()
    k_scores.append(score)

# print(k_scores)

# Plotting the k scores and values
plt.plot(k_range, k_scores)
plt.xlabel('k_range')
plt.ylabel('k_scores')
# plt.show()

"""
There's a function that finds the best parameters for us, called GridSearchCV

Grid search allows you to define some parameters that you want to try with the given model and it will automatically run 
cross validation using each of those parameters while keeping track of the resulting scores.  

"""
print('\n' * 3)


# Specifying the k range. The '*' unpacks range and the '[]' converts it into a list. It's more efficient than
# list(range(1, 31))
k_range = [*range(1, 31)]

# Creating a parameter grid: a Python dictionary {key = parameter_name: value = list[k_range]}
parameter_grid = dict(n_neighbors=k_range)
print(parameter_grid)

# Creating the grid search
# The grid object does 10 fold cross validation on the KNN model using classification accuracy as the evaluation metric,
# in addition we give it the parameter grid so that it knows to repeat the 10 fold cross validation process 30 times.
# And also, each time the 'n_neighbors' parameter should be given a different value from the list.
grid = GridSearchCV(estimator=knn, param_grid=parameter_grid, cv=10, scoring='accuracy')

# Then we fit the grid with data by passing in x, y
# This may take a while because we're testing 30 parameters and performing 10 fold cross validation on each. Meaning
# that the KNN model is being fit and predictions are being made for a total of 300 times
grid.fit(X=x, y=y)

# Printing the results
# Prints the best parameters
print(f"Best parameters: {grid.best_params_}")
# You have to use a for loop to print out all the grid scores
params = grid.cv_results_['params']
means = grid.cv_results_['mean_test_score']
stds = grid.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, params):
    # Keep in mind that even though the accuracy is good, if the standard deviation is high, the cross validated
    # accuracy might not be reliable
    print(f"Param: {params}, mean: {mean:.2f}, std: {std:.2f}")

# Gathering the average accuracy scores
grid_mean_scores = [mean for mean in grid.cv_results_['mean_test_score']]
print(grid_mean_scores)

# Plotting the results
plt.plot(k_range, grid_mean_scores)
plt.xlabel('k')
plt.ylabel('accuracy')
# plt.show()

# best_index returns the index of the best parameter/score, best_score returns the best score, best_params returns the
# best parameters, and best_estimator returns the model's parameters
print(f"best_index_: {grid.best_index_}, best_score_: {grid.best_score_}, best_params_: {grid.best_params_},"
      f" \nbest_estimator: {grid.best_estimator_}")


"""
But what if we want to optimize multiple parameters for a model?

"""
print('\n' * 3)


# Define the parameter and it's values that should be searched
k_range = [*range(1, 31)]
weight_values = ['uniform', 'distance']

# Create the grid
parameter_grid = dict(n_neighbors=k_range, weights=weight_values)
print(parameter_grid)

# Instantiate and fit the grid
grid = GridSearchCV(estimator=knn, param_grid=parameter_grid, cv=10, scoring='accuracy')
grid.fit(X=x, y=y)

# Printing the grid scores
params = grid.cv_results_['params']
means = grid.cv_results_['mean_test_score']
stds = grid.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, params):
    # Keep in mind that even though the accuracy is good, if the standard deviation is high, the cross validated
    # accuracy might not be reliable
    print(f"Param: {params}, mean: {mean:.2f}, std: {std:.2f}")

# The best model
# 13 is still the best parameter and weight's default value 'uniform' is the same
print(f"Best score: {grid.best_score_}, best parameters: {grid.best_params_}")


"""
What to do with the optimal tuning parameters.

Before you make predictions on out-of-sample data (data you don't have), you have to train the model with the best
known parameters using all of your data. 

1) and 2) do the same thing
"""

# 1)
# Training the KNN model on all the data and best known parameters
# The reason why we don't train out model on let's say 80% of the data, is because you may be wasting valuable data that
# the model can learn from.
knn = KNeighborsClassifier(n_neighbors=13, weights='uniform')
knn.fit(x, y)
predicted = knn.predict([[3, 5, 4, 2]])
print(f"Predicted: {predicted}")

# 2)
# Or we can use the GridSearchCV to predict which sets the KNN to all
# By default it refits the model using the entire data set and the best parameters it found.
predicted = grid.predict([[3, 5, 4, 2]])
print(f"Predicted: {predicted}")


"""
Searching for the optimal parameter using GridSearchCV is very computationally expensive. RandomizedSearchCV searches 
only a random subset of the provided parameters and allowing you to control the number of different parameter 
combinations that are attempted. Which means you can decide on how long you want it to run for (for computational
costs).
"""

# Specify a continuous distribution (rather than a list of values) for continuous parameters.
# Such as a regularization parameter for a regression problem
param_dict = dict(n_neighbors=k_range, weights=weight_values)

# n_iter = number of searches
rand_cv = RandomizedSearchCV(estimator=knn, param_distributions=param_dict, cv=10, scoring='accuracy', n_iter=10,
                             random_state=0)

rand_cv.fit(x, y)

params = rand_cv.cv_results_['params']
means = rand_cv.cv_results_['mean_test_score']
stds = rand_cv.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, params):
    # Keep in mind that even though the accuracy is good, if the standard deviation is high, the cross validated
    # accuracy might not be reliable
    print(f"Param: {params}, mean: {mean:.2f}, std: {std:.2f}")

print(f"RandomSearchCV best score: {rand_cv.best_score_}")


# A cool way of finding out how often the RandomSearch finds a 98% accuracy with our data set
best_scores = []
for num in range(1, 20):
    rand_cv = RandomizedSearchCV(estimator=knn, param_distributions=param_dict, cv=10, scoring='accuracy', n_iter=10)
    rand_cv.fit(x, y)
    best_scores.append(round(number=rand_cv.best_score_, ndigits=3))
print(f"RandSearch: {best_scores}")
