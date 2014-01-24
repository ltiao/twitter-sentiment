#!/usr/bin/env python
# -*- coding: utf-8 -*-

# import os,sys,inspect
# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(currentdir)
# sys.path.insert(0,parentdir) 

# Setup Logging
from settings import setup_logging
from logging import getLogger

setup_logging()
logger = getLogger('dataset')

from analyzer import preprocess, tokenize

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer

from sklearn.preprocessing import LabelEncoder
from sklearn.datasets.base import Bunch

from sklearn.base import BaseEstimator, TransformerMixin

import numpy as np
import scipy as sp

def load_semeval_vectorized(subtask='b', subset='all'):
    try:
        from pymongo import MongoClient
    except ImportError:
        raise ImportError('pymongo must be installed to retrieve data from MongoDB')

    client = MongoClient()
    db = client.twitter_database
    db_labeled_tweets = db.labeled_tweets

    if subtask == 'a':
        raise NotImplementedError('SemEval-2013 Task 2 Subtask A data not yet supported')
    elif subtask == 'b':
        
        q = {
            u'text': {'$exists': True}, 
            u'class.overall': {'$exists': True}
        }

        tweets = list(db_labeled_tweets.find(q))

        vect = CountVectorizer(tokenizer=tokenize, preprocessor=preprocess)#, ngram_range=(1, 2))
        le = LabelEncoder()
        
        if subset == 'all':
            data = vect.fit_transform(tweet.get(u'text') for tweet in tweets)
            target = le.fit_transform(
                [u'neutral' if tweet['class']['overall'] in ('neutral', 'objective', 'objective-OR-neutral') else tweet['class']['overall'] for tweet in tweets]
            )
            docs = tweets
        elif subset == 'train':
            training_tweets = [tweet for tweet in tweets if tweet['class']['training']]
            data = vect.fit_transform(tweet.get(u'text') for tweet in training_tweets)
            target = le.fit_transform(
                [u'neutral' if tweet['class']['overall'] in ('neutral', 'objective', 'objective-OR-neutral') else tweet['class']['overall'] for tweet in training_tweets]
            )
            docs = training_tweets
        elif subset == 'test':
            training_tweets = [tweet for tweet in tweets if tweet['class']['training']]
            vect.fit(tweet.get(u'text') for tweet in training_tweets)
            le.fit(
                [u'neutral' if tweet['class']['overall'] in ('neutral', 'objective', 'objective-OR-neutral') else tweet['class']['overall'] for tweet in training_tweets]
            )
            testing_tweets = [tweet for tweet in tweets if not tweet['class']['training']]
            data = vect.transform(tweet.get(u'text') for tweet in testing_tweets)
            target = le.transform(
                [u'neutral' if tweet['class']['overall'] in ('neutral', 'objective', 'objective-OR-neutral') else tweet['class']['overall'] for tweet in testing_tweets]
            )
            docs = testing_tweets
        else:
            raise ValueError("'{}' is not a valid subset: should be one of ['train', 'test', 'all']".format(subset))
            
    else:
        raise ValueError("'{}' is not a valid subtask: should be one of ['a', 'b']".format(subtask))

    return Bunch(
            data = data,
            target = target,
            target_names = le.classes_,
            label_encoder = le,
            vectorizer = vect,
            docs = np.asarray(docs)
        )

def load_semeval(subtask='b', subset='all'):
    try:
        from pymongo import MongoClient
    except ImportError:
        raise ImportError('pymongo must be installed to retrieve data from MongoDB')

    client = MongoClient()
    db = client.twitter_database
    db_labeled_tweets = db.labeled_tweets

    if subtask == 'a':
        raise NotImplementedError('SemEval-2013 Task 2 Subtask A data not yet supported')
    elif subtask == 'b':
        
        q = {
            u'text': {'$exists': True}, 
            u'class.overall': {'$exists': True}
        }
    
        if subset in ('train', 'test'):
            q[u'class.training'] = (subset == 'train')
        elif subset != 'all':
            raise ValueError("'{}' is not a valid subset: should be one of ['train', 'test', 'all']".format(subset))

        return db_labeled_tweets.find(q)
    else:
        raise ValueError("'{}' is not a valid subtask: should be one of ['a', 'b']".format(subtask))

class TweetVectorizer(BaseEstimator, TransformerMixin):
    
    def __init__(self, count_vectorizer=None, dict_vectorizer=None):
        self.count_vect = self.build_count_vectorizer(count_vectorizer)
        self.dict_vect = self.build_dict_vectorizer(dict_vectorizer)
        if self.count_vect is None and self.dict_vect is None:
            raise ValueError('Must provide at least one vectorizer.')

    # Identity function - to support subclassing
    def build_count_vectorizer(self, count_vect):
        return count_vect
        
    # Identity function - to support subclassing
    def build_dict_vectorizer(self, dict_vect):
        return dict_vect

    def fit(self, X, y=None):
        if self.count_vect is not None:
            self.count_vect.fit(x.get(u'text') for x in X)
            self.feature_names_ = self.count_vect.get_feature_names()
        else:
            self.feature_names_ = []
            
        if self.dict_vect is not None:
            self.dict_vect.fit(self._features_dict(x) for x in X)
            self.feature_names_.extend(self.dict_vect.get_feature_names())

        return self

    def transform(self, X, y=None):
        if self.count_vect is not None:
            X1 = self.count_vect.transform(x.get(u'text') for x in X)
            
        if self.dict_vect is not None:
            X2 = self.dict_vect.transform(self._features_dict(x) for x in X)
        
        try:
            return sp.sparse.hstack((X1, X2))
        except UnboundLocalError:
            try: return X1
            except UnboundLocalError: return X2
        
    def inverse_transform(self, X):
        raise NotImplementedError('Does not support inverse transform yet.')

    def get_feature_names(self):
        return self.feature_names_
        
    def get_count_vectorizer(self):
        return self.count_vect
        
    def get_dict_vectorizer(self):
        return self.dict_vect

    def _features_dict(self, tweet):
        if self.count_vect is not None:
            analyzer = self.count_vect.build_analyzer()
        else:
            analyzer = lambda s: s.split()
        tweet_text = tweet.get(u'text')
        tokens = analyzer(tweet_text)
        is_reply = tweet.get(u'in_reply_to_status_id', None) is not None
        num_tokens = len(tokens)
        return dict(
            is_reply = is_reply,
            num_tokens = num_tokens,
        )

if __name__ == '__main__':
    
    twitter_data = load_semeval(subtask='b', subset='all')
    
    vec1 = CountVectorizer(tokenizer=tokenize, preprocessor=preprocess)
    vec2 = DictVectorizer()
    le = LabelEncoder()
    
    vec = TweetVectorizer(dict_vectorizer=vec2, count_vectorizer=vec1)
    X = vec.fit_transform(list(twitter_data[:10]))
    print X.shape
    print X.todense()
    print vec.get_feature_names()
    
    exit(0)
    
    import matplotlib.pyplot as plt
    import numpy as np

    tweets = load_semeval(subtask='b', subset='all')
    
    bincount = np.bincount(tweets.target)
    
    n = bincount.shape[0]
    
    ind = np.arange(n)  # the x locations for the groups
    width = 0.5         # the width of the bars
    
    #plt.xkcd()
    
    fig, ax = plt.subplots()
    
    rects = ax.bar(ind, bincount, width, align='center', facecolor='#9999ff')
    
    ax.set_ylabel('Frequency')
    ax.set_xlabel('Class')
    ax.set_title('Class frequency distribution')
    ax.set_xticks(ind)
    ax.set_xticklabels(tweets.target_names)
    ax.set_axisbelow(True)
    ax.yaxis.grid()
    
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., height+100, '{}'.format(int(height)), ha='center')
    
    plt.savefig('temp.png')