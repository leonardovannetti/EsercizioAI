# Modulo di classificazione che utilizza Random Forest

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import data


clf = RandomForestClassifier(criterion='entropy', ccp_alpha=0, min_impurity_decrease=0, min_samples_leaf=1,
                             min_samples_split=4, random_state=0)

clf = clf.fit(data.X_train, data.y_train)

predicted = clf.predict(data.X_test)

print(f"Classification report for classifier {clf}:\n"
      f"{metrics.classification_report(data.y_test, predicted, digits=3)}\n")

print(metrics.confusion_matrix(data.y_test, predicted))