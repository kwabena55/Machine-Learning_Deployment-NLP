# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 18:36:23 2020

@author: User
"""

from sklearn.naive_bayes import MultinomialNB
from flask  import Flask, render_template, url_for, request
import pandas as pd
import pickle


#Load the model from the disk
filename='nlp_model.pkl'
filename2='transform.pkl'
clf=pickle.load(open(filename,'rb'))
cv= pickle.load(open(filename2,'rb'))


#Use of Decorators
app=Flask(__name__)



@app.route('/')
def home():
    return render_template('home.html')




@app.route('/predict',methods=['POST'])
def predict():

	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		vect = cv.transform(data).toarray()
		my_prediction = clf.predict(vect)
	return render_template('result.html',prediction = my_prediction)



if __name__ == '__main__':
	app.run(debug=True)