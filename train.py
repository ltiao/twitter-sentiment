#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import KFold, cross_val_score

from datasets import load_semeval

semeval_tweets = {}
for subset in ('train', 'test'):
    semeval_tweets[subset] = load_semeval(subtask='b', subset=subset)

X_train, y_train = semeval_tweets['train'].data, semeval_tweets['train'].target
X_test, y_test = semeval_tweets['test'].data, semeval_tweets['test'].target

temp = load_semeval(subtask='b', subset='all')

print X_train.shape
print X_test.shape
print temp.data.shape

exit(0)

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier

classifiers = (
    ('Majority class', DummyClassifier(strategy='most_frequent')),
    ('Multinomial Naive Bayes', MultinomialNB()),
    # ('Maximum Entropy', LogisticRegression()),
    # ('Linear SVM', LinearSVC()),
)

#clf = OneVsRestClassifier(SVC(probability=True))

for name, clf in classifiers: 
    print clf.__class__.__name__
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
    
