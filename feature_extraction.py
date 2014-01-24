#!/usr/bin/env python
# -*- coding: utf-8 -*-

# TODO: 
#   * Documentation
#   * Doctests
#   * Implement Inverse Transform
#   * Unit tests

# Setup Logging
from settings import setup_logging
from logging import getLogger
setup_logging()
logger = getLogger('dataset')

from sklearn.base import BaseEstimator, TransformerMixin

import numpy as np
import scipy as sp

class CountDictCombinedVectorizer(BaseEstimator, TransformerMixin):
    
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
            self.count_vect.fit(self._string_generator(X))
            self.feature_names_ = self.count_vect.get_feature_names()
        else:
            self.feature_names_ = []
            
        if self.dict_vect is not None:
            self.dict_vect.fit(self._features_dicts_generator(X))
            self.feature_names_.extend(self.dict_vect.get_feature_names())

        return self

    def transform(self, X, y=None):
        if self.count_vect is not None:
            X1 = self.count_vect.transform(self._string_generator(X))
            
        if self.dict_vect is not None:
            X2 = self.dict_vect.transform(self._features_dicts_generator(X))

        # This is succinct but hacky in some sense..
        try:
            return sp.sparse.hstack((X1, X2))
        except UnboundLocalError:
            try: return X1
            except UnboundLocalError: return X2
    
    # TODO
    def inverse_transform(self, X):
        raise NotImplementedError('Inverse Transform not implemented')

    def get_feature_names(self):
        return self.feature_names_
        
    def get_count_vectorizer(self):
        return self.count_vect
        
    def get_dict_vectorizer(self):
        return self.dict_vect

    def _string_generator(self, X):
        return (self.string_value(x) for x in X)

    def _features_dicts_generator(self, X):
        return (self.features_dict(x) for x in X)

    def string_value(self, x):
        raise NotImplementedError('String value not defined')

    def features_dict(self, x):
        raise NotImplementedError('Feature extraction dictionary not defined')
        
class TweetVectorizer(CountDictCombinedVectorizer):

    def string_value(self, x):
        return x.get(u'text')

    def features_dict(self, x):
        if self.count_vect is not None:
            analyzer = self.count_vect.build_analyzer()
        else:
            from analyzer import preprocess, tokenize
            analyzer = lambda s: tokenize(preprocess(s))
        tweet_text = x.get(u'text', '')
        tokens = analyzer(tweet_text)
        
        
        
        return dict(
            # true_label = u'neutral' if x['class']['overall'] in ('neutral', 'objective', 'objective-OR-neutral') else x['class']['overall'],
            # is_reply = x.get(u'in_reply_to_status_id', None) is not None,
            num_tokens = len(tokens),
            char_len = 
            # retweet_count = x.get(u'retweet_count'),
            # favorite_count = x.get(u'favorite_count')
        )

if __name__ == '__main__':

    from data import load_semeval
    from analyzer import preprocess, tokenize

    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction import DictVectorizer
    
    twitter_data = load_semeval(subtask='b', subset='all')

    vec = TweetVectorizer(dict_vectorizer=DictVectorizer(), count_vectorizer=CountVectorizer(tokenizer=tokenize, preprocessor=preprocess))
    
    import pprint
    
    for x in twitter_data[:10]:
        pprint.pprint(x)
        pprint.pprint(vec.features_dict(x))
        print
        
    exit(0)
    X = vec.fit_transform(list(twitter_data[:10]))
    
    print X.shape
    print vec.get_feature_names()