#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import KFold, cross_val_score

import numpy as np
from time import time

x = np.arange(45).reshape(9, 5)

print x

def array_split_gen(*args, **kwargs):
    cumulative = bool(kwargs.pop('cumulative', False))
    ary_splt = np.array_split(*args, **kwargs)
    if cumulative:
        for subary in ary_splt:
            try:
                cum_ary = np.vstack((cum_ary, subary))
            except UnboundLocalError:
                cum_ary = subary
            yield cum_ary
    else:
        for subary in ary_splt:
            yield subary
    
print list(array_split_gen(x, 5, cumulative=True))

exit(0)

from datasets import load_semeval

semeval_tweets = {}
for subset in ('train', 'test'):
    semeval_tweets[subset] = load_semeval(subtask='b', subset=subset)

X_train, y_train = semeval_tweets['train'].data, semeval_tweets['train'].target
X_test, y_test = semeval_tweets['test'].data, semeval_tweets['test'].target
feature_names = semeval_tweets['train'].vectorizer.get_feature_names()

semeval_all = load_semeval(subtask='b', subset='all')
X_all, y_all = semeval_all.data, semeval_all.target

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix

classifiers = (
    ('Majority class', DummyClassifier(strategy='most_frequent')),
    ('Multinomial Naive Bayes', MultinomialNB()),
    # ('Maximum Entropy', LogisticRegression()),
    # ('Linear SVM', LinearSVC()),
)

#clf = OneVsRestClassifier(SVC(probability=True))

for name, clf in classifiers: 
    print 'Training classfier [{}]: {}'.format(name, clf)
    
    t0 = time()
    clf.fit(X_train, y_train)
    print 'training time: {:.3f}s'.format(time()-t0)
    
    t0 = time()
    pred = clf.predict(X_test)
    print 'test time: {:.3f}s'.format(time()-t0)
    
    print classification_report(y_test, pred, target_names=semeval_all.target_names)
    print confusion_matrix(y_test, pred)
    
    if hasattr(clf, 'coef_'):
        print 'Dimensionality (#features): {}'.format(clf.coef_.shape[1])
        n_keywords = 100
        print 'top {} keywords per class'.format(n_keywords)
        for i, klass in enumerate(semeval_all.target_names):
            top_n = np.argsort(clf.coef_[i])[-n_keywords:]
            print '{}: {}'.format(klass, np.asarray(feature_names)[top_n])
    
    # try:
    #     accuracy_10_fold = cross_val_score(clf, X, y, cv=10)
    #     f1_10_fold = cross_val_score(clf, X, y, cv=10, scoring='f1')
    # except TypeError:
    #      accuracy_10_fold = cross_val_score(clf, X.toarray(), y, cv=10)
    #      f1_10_fold = cross_val_score(clf, X.toarray(), y, cv=10, scoring='f1')
    # print 'Accuracy: {:2f} (+/- {:2f})'.format(accuracy_10_fold.mean(), accuracy_10_fold.std()*2)
    # print 'F1: {:2f} (+/- {:2f})'.format(f1_10_fold.mean(), f1_10_fold.std()*2)
    # print

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
    
