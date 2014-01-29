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


import numpy as np
import scipy as sp

from sklearn.base import BaseEstimator, TransformerMixin

from string import punctuation
import regex

from analyzer import preprocess, tokenize
from pprint import pprint, pformat
from nltk.data import load

sentence_tokenizer = load('tokenizers/punkt/english.pickle')

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

        # This is succinct but also quite hacky in some sense..
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

def pattern_features_dict(pattern, string, prefix=None, check_match=False, count_matches=False, n_start=None):
    # Just another way of asking: isinstance(pattern, regex._pattern_type)
    try:
        matches = list(pattern.finditer(string))
    except AttributeError:
        matches = list(regex.finditer(pattern, string))
    
    logger.debug('{}:\n\tPattern: {}\n\tMatches: {}'.format(prefix, pattern, [mo.group() for mo in matches]))
    
    if prefix is None:
        prefix = pattern

    result = {}
    
    if check_match:
        result['match_exists'] = (matches != [])
    
    if count_matches:
        result['match_count'] = len(matches)
        
    if n_start is not None:
        try:
            result['match_start'] = matches[n_start].start()
        except IndexError:
            # default position if there was no match is defined as -1. 
            # Could also use something like 140.
            result['match_start'] = -1
    
    for k in result.keys():
        result['_'.join([prefix, k])] = result.pop(k)

    return result

class TweetVectorizer(CountDictCombinedVectorizer):

    def string_value(self, x):
        return x.get(u'text')

    def features_dict(self, x):
        if self.count_vect is not None:
            analyzer = self.count_vect.build_analyzer()
        else:
            analyzer = lambda s: tokenize(preprocess(s))
        
        tweet_text = x.get(u'text', '')
        
        word_tokens = analyzer(tweet_text)
        sent_tokens = sentence_tokenizer.tokenize(tweet_text)
        
        # logger.debug(tweet_text)
        # logger.debug(pformat(word_tokens))
        # logger.debug(pformat(sent_tokens))

        hashtags = x.get(u'entities', {}).get(u'hashtags', [])

        features_dict_ = dict(
            # true_label = u'neutral' if x['class']['overall'] in ('neutral', 'objective', 'objective-OR-neutral') else x['class']['overall'],
            # is_reply = x.get(u'in_reply_to_status_id', None) is not None,
            char_count = len(tweet_text),
            word_count = len(word_tokens),
            sentence_count = len(sent_tokens),
            retweet_count = x.get(u'retweet_count'),
            favorite_count = x.get(u'favorite_count'),
            hashtags_count = len(hashtags)
        )

        d = {
            'question_marks': {
                'pattern': r'\?',
                'count_matches': True,
            },
            'colons': {
                'pattern': r':\s*\w',
                'count_matches': True,
                'n_start': 0,
            },
            'repeated': {
                'pattern': r'(\w)\1{2,}',
                'count_matches': True,
            },
            'all_caps': {
                'pattern': r'\b[A-Z]{2,}\b',
                'count_matches': True,
            },
            'punctuation_marks': {
                'pattern': r'[{0}]'.format(regex.escape(punctuation)),
                'count_matches': True,
            }
        }

        for k in d:
            features_dict_.update(pattern_features_dict(string=tweet_text, prefix=k, **d[k]))
        
        return features_dict_

if __name__ == '__main__':

    from data import load_semeval
    from analyzer import preprocess, tokenize

    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction import DictVectorizer
    
    twitter_data = load_semeval(subtask='b', subset='all')

    vec = TweetVectorizer(dict_vectorizer=DictVectorizer(), count_vectorizer=CountVectorizer(tokenizer=tokenize, preprocessor=preprocess))
        
    for x in twitter_data[:10]:
        features = vec.features_dict(x)
        pprint(x)
        pprint(features)
        print
        
    exit(0)
    X = vec.fit_transform(list(twitter_data[100:200]))
    
    print X.shape
    print vec.get_feature_names()