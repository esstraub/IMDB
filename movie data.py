import matplotlib.pyplot as plt
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv(r"C:\Projects - Coding\IMDB\IMDB-Dataset.csv")

# Review & understand the dataset
print(data.describe())
print(data.head())

# Check the distribution of sentiment
data.sentiment.value_counts().plot(kind='pie', autopct='%1.0f%%', colors=['purple','blue'])
plt.show() # Displays equal distribution of positive and negative reviews

# Change the classification to a numerical value
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["sentiment"] = le.fit_transform(data["sentiment"])
print(data.head())

# Split data so some can be used to train and some to test.
from sklearn.model_selection import train_test_split
y=data['sentiment']
x=data['review']
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.2)
print("Train: ", x_train.shape)
print("Test: ", x_test.shape)

# Bag of words with count vectoriser
from sklearn.feature_extraction.text import CountVectorizer

vectCount=CountVectorizer(ngram_range=(1,2))
x_train_trans_Count=vectCount.fit_transform(x_train)

#Compare 2 different model types for accuragey - logistic regression and decision tree
logreg = Pipeline([('lrmodel', LogisticRegression())])
logreg.fit(x_train_trans_Count,y_train)
pred_y=logreg.predict(vectCount.transform(x_test))
score_logreg=accuracy_score(y_test,pred_y)
report_logreg = classification_report(y_test,pred_y)

print("Logistic regression")
print("Accuracy: ", score_logreg)
print(report_logreg)

dtree = Pipeline([('dtreemodel', DecisionTreeClassifier())])
dtree.fit(x_train_trans_Count,y_train)
pred_y=dtree.predict(vectCount.transform(x_test))
score_dtree=accuracy_score(y_test,pred_y)
report_dtree = classification_report(y_test,pred_y)

print("Decision tree")
print("Accuracy: ", score_dtree)
print(report_dtree)

#Result - decision Tree not as accurate


