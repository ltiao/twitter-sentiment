#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Setup Logging
from settings import setup_logging
from logging import getLogger

setup_logging()
logger = getLogger('eval')

logger.info('importing packages...')

# Import standard packages
from time import time, sleep
from pprint import pprint, pformat

# Import 3rd-party packages
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import KFold, StratifiedKFold, cross_val_score
from sklearn.learning_curve import learning_curve

from tqdm import tqdm as enumerate_progress

# Import project packages
from data import load_semeval

logger.info('loading SemEval-2013 data...')

semeval_tweets = {}
for subset in ('train', 'test'):
    semeval_tweets[subset] = load_semeval(subtask='b', subset=subset)

X_train, y_train = semeval_tweets['train'].data, semeval_tweets['train'].target
X_test, y_test = semeval_tweets['test'].data, semeval_tweets['test'].target
feature_names = semeval_tweets['train'].vectorizer.get_feature_names()

semeval_all = load_semeval(subtask='b', subset='all')
X_all, y_all = semeval_all.data, semeval_all.target

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, make_scorer

classifiers = (
    ('Majority class', DummyClassifier(strategy='most_frequent')),
    ('Multinomial Naive Bayes', MultinomialNB()),
    # ('Stochastic Gradient Descent', SGDClassifier()),
    ('Maximum Entropy', LogisticRegression()),
    ('Linear SVM', LinearSVC()),
)

logger.info('training...')

scoring_metrics = (
    'accuracy',
    'f1',
    # 'precision',
    # 'recall',
    # 'roc_auc',
)

for name, clf in classifiers: 
    logger.info('evaluating [{}] classifier'.format(name))
    logger.debug(clf)
    
    # skf_cv = StratifiedKFold(labels=y_all, n_folds=10)
    t0 = time()
    clf.fit(X_train, y_train)
    logger.info('training time: {:.3f}s'.format(time()-t0))
    
    t0 = time()
    pred = clf.predict(X_test)
    pred_proba = clf.predict_proba(X_test)
    logger.info('test time: {:.3f}s'.format(time()-t0))
    
    logger.info('\n' + format(classification_report(y_test, pred, target_names=semeval_tweets['test'].target_names)))
    
    errors = (pred == y_test)
        
    for probs, predicted, true, text, vect in zip(pred_proba[errors], pred[errors], y_test[errors], semeval_tweets['test'].ids[errors], X_test[errors].toarray()):
        # logger.info(np.asarray(semeval_tweets['test'].vectorizer.get_feature_names())[np.nonzero(vect)])
        logger.info('Text: {}'.format(text))
        logger.info('Probabilities: [{}] | Pred: {} | True: {}'.format(
                ' '.join('P({})={}'.format(semeval_tweets['test'].target_names[i], prob) for i, prob in enumerate(probs)),
                semeval_tweets['test'].target_names[predicted], 
                semeval_tweets['test'].target_names[true], 
            )
        )
        logger.info('')
    
    if hasattr(clf, 'coef_'):
        logger.info('Dimensionality (#features): {}'.format(clf.coef_.shape[1]))
        n_keywords = 100
        logger.info('top {} keywords per class'.format(n_keywords))
        for i, klass in enumerate(semeval_all.target_names):
            top_n = np.argsort(clf.coef_[i])[-n_keywords:]
            top_n_dict = dict(zip(np.asarray(feature_names)[top_n], clf.coef_[i][top_n]))
            logger.info('{}:'.format(klass))
            for feature in sorted(top_n_dict, key=lambda k: top_n_dict[k], reverse=True):
                logger.info(' * {:.<20}{:.>16}'.format(feature, top_n_dict[feature]))
    
    continue
    
    logger.info('computing cross-validated metrics')
    logger.debug('metrics: {}'.format(scoring_metrics))
    
    cross_val_scores = dict((metric, cross_val_score(clf, X_all, y_all, cv=10, scoring=metric)) for metric in scoring_metrics)
    
    logger.info('generating learning curve data for metrics')
    learning_curves = dict((metric, learning_curve(clf, X_all, y_all, cv=10, scoring=metric)) for metric in scoring_metrics)
    
    for metric in scoring_metrics:
        logger.info('{name}: {mean:.3f} ({std:.3f})'.format(
                name = metric, 
                mean = cross_val_scores[metric].mean(), 
                std = cross_val_scores[metric].std()
            )
        )
        
        logger.info('plotting learning curve for [{}] classifier with respect to metric [{}]'.format(name, metric))
        
        train_sizes, train_scores, test_scores = learning_curves[metric]
        
        fig, ax = plt.subplots()

        ax.plot(train_sizes, train_scores, label='Trainings set {}-score'.format(metric))
        ax.plot(train_sizes, test_scores, label='Test set {}-score'.format(metric))

        plt.title('Learning Curve of {} classifer'.format(name))

        plt.xlabel('Training set size')
        plt.ylabel('Score')

        plt.xlim(train_sizes[0], train_sizes[-1])

        plt.legend(loc='best')

        filename = '{name}_{metric}.png'.format(name=clf.__class__.__name__, metric=metric)

        logger.info('saving to file [{}]'.format(filename))

        plt.savefig(filename)
        
    continue
    
    print classification_report(y_test, pred, target_names=semeval_all.target_names)
    print confusion_matrix(y_test, pred)
    
    
    
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
    
