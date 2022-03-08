import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from preprocessing import remove_punct, plot_conf_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier

data = pd.read_csv("mediaeval-2015-dataset.txt", "\t")

# Text needs to have same attributes across all of the dataset
# Removal of all capitalised letters
data['tweetText'] = data['tweetText'].apply(lambda x: x.lower())

data['tweetText'] = data['tweetText'].apply(remove_punct)

# Check for all values of 'label'
df = pd.DataFrame(data)
uniqueLabels = df['label'].unique()
print(uniqueLabels)

label = data['label']
for i in range(len(label)):
    if label[i] == 'humor':
        label[i] = 'fake'

# Check for all values of 'label'
uniqueLabels = label.unique()
print(uniqueLabels)

#Labels are now 'real' and 'fake' as desired
#Now remove the old column and add the new one
data = data.drop('label', axis = 1)
data['label'] = label

X_train,X_test,y_train,y_test = train_test_split(data['tweetText'],
                                                 data["label"],
                                                 test_size=0.2)

# Train first model
pipe = Pipeline([('vector', CountVectorizer()),
                 ('tfidf', TfidfTransformer()),
                 ('model', LinearSVC())])
model = pipe.fit(X_train, y_train)

predict = model.predict(X_test)

print("LinearSVC Model Accuracy: {}%".format(round(accuracy_score(y_test, predict)*100,2)))

cm = confusion_matrix(y_test, predict)
plot_conf_matrix(cm, classes = ['Fake','Real'], title = 'Confusion Matrix for LinearSVC Model')
plt.show()

# Decision Tree Classifier
pipe = Pipeline([('vect', CountVectorizer()),
                 ('tfidf', TfidfTransformer()),
                 ('model', DecisionTreeClassifier(criterion="entropy",
                                                  class_weight="balanced"))])
dtc = pipe.fit(X_train, y_train)
predict = dtc.predict(X_test)
print("Decision Tree Classifier Model Accuracy: {}%".format(round(accuracy_score(y_test, predict) * 100, 2)))

cm = confusion_matrix(y_test, predict)
plot_conf_matrix(cm, classes = ['Fake','Real'], title = 'Confusion Matrix for DTC Model')
plt.show()

# Now let's test Random Forest Classifier
pipe = Pipeline([('vect', CountVectorizer()),
                 ('tfidf', TfidfTransformer()),
                 ('model', RandomForestClassifier(criterion = "entropy",
                                                 class_weight = "balanced",
                                                  ))])
rfc = pipe.fit(X_train, y_train)

predict = rfc.predict(X_test)
print("Random Forest Classifier Model Accuracy: {}%".format(round(accuracy_score(y_test, predict)*100,2)))

cm = confusion_matrix(y_test, predict)
plot_conf_matrix(cm, classes = ['Fake','Real'], title = 'Confusion Matrix for RFC Model')
plt.show()

# MultinomialNB
pipe = Pipeline([('vect', CountVectorizer()),
                 ('tfidf', TfidfTransformer()),
                 ('model', MultinomialNB(alpha = 0.01))])
mnb = pipe.fit(X_train, y_train)

predict = mnb.predict(X_test)
print("MultinomialNB 2 Model Accuracy: {}%".format(round(accuracy_score(y_test, predict)*100,2)))

cm = confusion_matrix(y_test, predict)
plot_conf_matrix(cm, classes = ['Fake','Real'], title = 'Confusion Matrix for MultinomialNB Model')
plt.show()

# KNeighborsClassifier
pipe = Pipeline([('vect', CountVectorizer()),
                 ('tfidf', TfidfTransformer()),
                 ('model', KNeighborsClassifier(algorithm = 'brute',
                                                weights = 'distance'
                                               ))])
knn = pipe.fit(X_train, y_train)

predict = knn.predict(X_test)
print("KNN Model Accuracy: {}%".format(round(accuracy_score(y_test, predict)*100,2)))

cm = confusion_matrix(y_test, predict)
plot_conf_matrix(cm, classes = ['Fake','Real'], title = 'Confusion Matrix for KNN Model')
plt.show()

# SGDClassifier
pipe = Pipeline([('vect', CountVectorizer()),
                 ('tfidf', TfidfTransformer()),
                 ('model', SGDClassifier(alpha = 0.00001,
                                        penalty = 'elasticnet',
                                        l1_ratio = 0.05))])
sgdc = pipe.fit(X_train, y_train)

predict = sgdc.predict(X_test)
print("SGDClassifier 2 Model Accuracy: {}%".format(round(accuracy_score(y_test, predict)*100,2)))

cm = confusion_matrix(y_test, predict)
plot_conf_matrix(cm, classes = ['Fake','Real'], title = 'Confusion Matrix for SGDClassifier Model')
plt.show()