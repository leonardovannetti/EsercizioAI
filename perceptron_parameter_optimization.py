#  Module for parameter optimization of the perceptron classifier

import data
from sklearn.linear_model import Perceptron
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

tuned_parameters = [{'max_iter': [100, 1000, 10000], 'class_weight': ['balanced', None]}]

print("# Tuning hyper-parameters for accuracy")
print()

# grid search utilizzando stratifiedkfold per mantenere la proporzione delle classi anche negli split

clf = GridSearchCV(Perceptron(n_jobs=-1), tuned_parameters, scoring='accuracy',
                   cv=StratifiedKFold(5).split(data.X_train, data.y_train))
clf.fit(data.X_train, data.y_train)
print("Best parameters set found on development set:")
print()
print(clf.best_params_)
print()
print("Grid scores on development set:")
print()
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))
