#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re

from tokenizer import TwitterTokenizer
from preprocess import TwitterTextPreprocessor
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import KFold, cross_val_score

vect = CountVectorizer(tokenizer=TwitterTokenizer(), preprocessor=TwitterTextPreprocessor())
# analyze = vect.build_analyzer()

le = LabelEncoder()

from pymongo import MongoClient

client = MongoClient()
db = client.twitter_database
db_labeled_tweets = db.labeled_tweets

tweets = list(db_labeled_tweets.find({u'text': {'$exists': True}, u'class.overall': {'$exists': True}}))

raw_labels = [u'neutral' if tweet['class']['overall'] in ('neutral', 'objective', 'objective-OR-neutral') else tweet['class']['overall'] for tweet in tweets if tweet['class']['overall']]

X = vect.fit_transform(tweet.get(u'text') for tweet in tweets)
y = le.fit_transform(raw_labels)

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
classifiers = (
    DummyClassifier(strategy='most_frequent'),
    MultinomialNB(),
    #KNeighborsClassifier,
    #DecisionTreeClassifier,
    LogisticRegression(),
    LinearSVC(),
)

#clf = OneVsRestClassifier(SVC(probability=True))

for clf in classifiers:    
    print clf
    try:
        accuracy_10_fold = cross_val_score(clf, X, y, cv=10)
        f1_10_fold = cross_val_score(clf, X, y, cv=10, scoring='f1')
    except TypeError:
         accuracy_10_fold = cross_val_score(clf, X.toarray(), y, cv=10)
         f1_10_fold = cross_val_score(clf, X.toarray(), y, cv=10, scoring='f1')
    print 'Accuracy: {:2f} (+/- {:2f})'.format(accuracy_10_fold.mean(), accuracy_10_fold.std()*2)
    print 'F1: {:2f} (+/- {:2f})'.format(f1_10_fold.mean(), f1_10_fold.std()*2)
    print

exit(0)
clf.fit(X, y)

print le.inverse_transform(clf.predict(vect.transform([
        'this really sucks!', 
        'this saddens me greatly', 
        'this product is fantastic!', 
        'I love you man!', 
        'i hate you', 
        'jersey shore', 
        'the weather is sunny with a chance of meatballs'
    ])))
    
