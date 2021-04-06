# Modulo di classificazione che utilizza Perceptron

from sklearn.linear_model import Perceptron
from sklearn import metrics
import data


clf = Perceptron(max_iter=100)
clf = clf.fit(data.X_train, data.y_train)

predicted = clf.predict(data.X_test)

print(f"Classification report for classifier {clf}:\n"
      f"{metrics.classification_report(data.y_test, predicted, digits=3, zero_division=0)}\n")

print(metrics.confusion_matrix(data.y_test, predicted))