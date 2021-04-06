#  Module for parameter optimization of the random forest classifier

import data
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

tuned_parameters = [
    {'min_samples_leaf': [1, 2, 3, 4, 5],
     'min_impurity_decrease': [0, 0.01, 0.1],  # to check effectiveness of pre-pruning
     'min_samples_split': [2, 3, 4, 5, 6],
     'ccp_alpha': [0, 0.01, 0.015]}]  # to check effectiveness of post-pruning


print("# Tuning hyper-parameters for accuracy")
print()

# grid search utilizzando stratifiedkfold per mantenere la proporzione delle classi anche negli split

clf = GridSearchCV(RandomForestClassifier(criterion='entropy', n_jobs=-1, random_state=0), tuned_parameters,
                   scoring='accuracy', cv=StratifiedKFold(5).split(data.X_train, data.y_train))
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

