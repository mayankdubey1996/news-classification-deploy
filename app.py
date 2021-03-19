# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 13:15:09 2021

@author: mayan
"""
from flask import Flask,render_template,url_for,request
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
#from sklearn.externals import joblib
#import pickle

#load file
cv = pickle.load(open("tranformcv.pkl", 'rb'))
nb_cv = pickle.load(open("nbcv.pkl", 'rb'))
app = Flask(__name__)
@app.route('/',methods = ['GET'])
def home():
	return render_template('index.html')

def preprocess(article):
    corpus = []
    lemmatizer = WordNetLemmatizer()
    review = re.sub('[^a-zA-Z]', ' ', article) ## keeping only character
    review = review.lower() #lowering the words
    review = review.split() # converting sentences to words
    review = [lemmatizer.lemmatize(word)  for word in review if not word in stopwords.words('english')] # Lemmatizing or finding root word
    review = ' '.join(review)
    corpus.append(review)
    return corpus

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        article = request.form['article']
        corpus = preprocess(article)
        cv_transform = cv.transform(corpus)
        pred = nb_cv.predict(cv_transform)[0]
    return render_template('result.html',prediction = pred)

if __name__=="__main__":
    app.run(debug=True)
