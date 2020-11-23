# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from flask  import Flask, render_template, url_for, request
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
#from sklearn.externals import joblib



df = pd.read_csv("spam.csv", encoding="latin-1")
df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)

df


from sklearn.preprocessing import LabelEncoder
LE= LabelEncoder()
df["class"]=LE.fit_transform(df["class"])



#renaming columns
df['label']=df["class"]

df.drop(['class'], axis=1, inplace=True)


#Extracting Features and Labels
X= df["message"]
y=df["label"]


#extracting feature with Countvectorizer
cv= CountVectorizer()
X=cv.fit_transform(X)


pickle.dump(cv,open('transform.pkl','wb'))


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2, random_state=42)

#Use Naive Bayes Classifier to train model
from sklearn.naive_bayes import MultinomialNB

#Instantiate an object ou of the class
clf=MultinomialNB()
clf.fit(X_train,y_train)




#Testing
y_pred= clf.predict(X_test)


#Printing the Confusion Matrix of y_true and y_pred
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

cm=confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True)




print(classification_report(y_test,y_pred))





#record score on data
v=clf.score(X_test,y_test)
v1=clf.score(X_train,y_train)

print(v1)


#Remember a new file to be predicted will be transformed using the CountVectorizer before Predicted


#pickle file
filename='nlp_model.pkl'
pickle.dump(clf,open(filename,'wb'))












