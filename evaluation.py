#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Setup Logging
from settings import setup_logging
from logging import getLogger

setup_logging()
logger = getLogger('eval')

# Import standard modules
from time import time

# Import 3rd-party modules
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.learning_curve import learning_curve

import matplotlib.pyplot as plt
import numpy as np


def array_split_gen(ary, indices_or_sections, cumulative=False, *args, **kwargs):
    ary_splt = np.array_split(ary, indices_or_sections, *args, **kwargs)
    if cumulative:
        for subary in ary_splt:
            if subary.size:
                try:
                    cum_ary = np.concatenate((cum_ary, subary))
                except UnboundLocalError:
                    cum_ary = subary
                yield cum_ary
    else:
        for subary in ary_splt:
            if subary.size:
                yield subary

# print list(array_split_gen(y, 2, cumulative=False))
# exit(0)
def array_split_pct_gen(ary, pct=0.1, axis=0, *args, **kwargs):
    # Signature for this function is correct, just need
    # to use arange or linspace to calculate indices (as
    # opposed to sections) and call array_split_gen()
    raise NotImplementedError

# def learning_curve(clf, X_train, X_test, y_train, y_test):
#     sections = 10
#     result = []
#     for i, (X_sub, y_sub) in enumerate(zip(array_split_gen(X_train, sections, cumulative=True), array_split_gen(y_train, sections, cumulative=True))):
#         clf.fit(X_sub, y_sub)
#         #y_test_pred = clf.predict(X_test)
#         result.append(clf.score(X_test, y_test))
#     return np.array(result)

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
    
    train_sizes, train_score, test_score = learning_curve(clf, X_train, y_train)
    
    print train_score, test_score
    
    fig, ax = plt.subplots()
    
    ax.plot(train_sizes, train_score, label='Train')
    ax.plot(train_sizes, test_score, label='Test')
    
    plt.xlabel('Training set size')
    plt.ylabel('Score')
    
    plt.xlim(train_sizes[0], train_sizes[-1])
    
    plt.legend(loc='best')
    
    plt.savefig('{}.png'.format(clf.__class__.__name__))
    
    continue
    
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
    
