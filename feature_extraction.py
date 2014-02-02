#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Louis Tiao <Louis.Tiao@csiro.au>
#
# License: ?

# TODO:
#   * Documentation
#   * Doctests
#   * Unit tests

# Setup Logging
from settings import setup_logging
from logging import getLogger
setup_logging()
logger = getLogger('dataset')

import numpy as np
import scipy as sp

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline

from pprint import pprint, pformat
from string import punctuation
from nltk.data import load
from nltk import pos_tag
import regex

from analyzer import preprocess, tokenize

sentence_tokenizer = load('tokenizers/punkt/english.pickle')
upenn_tagset = load('help/tagsets/upenn_tagset.pickle')

def prefix_dict_keys(d, prefix):
    for k in d.keys():
        d['_'.join((prefix, k))] = d.pop(k)
    return d

def match_start(m, n):
    try:
        return m[n].start()
    except IndexError:
        return -1

def pattern_features_dict(pattern, string, **fn_names):
    # Just another way of asking: isinstance(pattern, regex._pattern_type)
    try:
        matches = list(pattern.finditer(string))
    except AttributeError:
        matches = list(regex.finditer(pattern, string))

    return dict((fn_name, fn(matches)) for fn_name, fn in fn_names.items())

class ExtractorPipeline(Pipeline):

    def __init__(self, get_names_from=None, *args, **kwargs):
        self.get_names_from = get_names_from
        super(ExtractorPipeline, self).__init__(*args, **kwargs)

    def get_feature_names(self):
        if self.get_names_from is not None:
            trans = self.named_steps.get(self.get_names_from)
        else:
            _, trans = self.steps[-1]
        return trans.get_feature_names()

class TweetTextExtractor(BaseEstimator, TransformerMixin):

    def fit(self, X=None, y=None):
        return self

    def transform(self, X, y=None):
        return [self._extract_text(x) for x in X]

    def _extract_text(self, x):
        return x.get(u'text', '')

class FeaturesDictExtractor(BaseEstimator, TransformerMixin):

    # TODO: Should the feature activation 
    # options be defined as kwargs in fit instead?
    def __init__(self, features=None):
        if features is not None:
            # TODO: Ensure iterable
            self.features = features
        else:
            self.features = self.get_all_possible_feature_names()

    def get_all_possible_feature_names(self):
        """the suffix (*) of all callable attributes of the form 'feature__*'"""
        return [attr.split('__', 1)[1] for attr in dir(self) if callable(getattr(self, attr)) and attr.split('__', 1)[0] == 'feature']

    def fit(self, X=None, y=None):
        return self

    def transform(self, X, y=None):
        return [self._extract_features(self.preprocess(x)) for x in X]

    def preprocess(self, x):
        """Identity function. 
        To be overriden by subclass
        """
        return x

    # TODO: fix so that the feature names are 
    # the same as those defined in self.features
    # rather than having to return a dict that
    # defines the key name again. (The key issue
    # here is support for feature__* functions with
    # multiple outputs)
    def _extract_features(self, x):
        features = {}
        for feature in self.features:
            feature_func = getattr(self, 'feature__{fn_name}'.format(fn_name=feature), lambda x: {})
            features.update(feature_func(x))
        return features

class TweetFeaturesExtractor(FeaturesDictExtractor):

    def preprocess(self, x):
        text = x.get(u'text', '')
        
        x['word_tokens'] = tokenize(preprocess(text))
        x['sentence_tokens'] = sentence_tokenizer.tokenize(text)
        
        return x
    
    def feature__char_count(self, x):
        return dict(char_count=len(x.get(u'text', '')))
    
    def feature__is_reply(self, x):
        return dict(is_reply=x.get(u'in_reply_to_status_id', None) is not None)
    
    def feature__word_count(self, x):
        return dict(word_count=len(x.get(u'word_tokens', [])))
        
    def feature__char_count(self, x):
        return dict(char_count=len(x.get(u'text', '')))
    
    def feature__sentence_count(self, x):
        return dict(word_count=len(x.get(u'word_tokens', [])))
    
    def feature__hashtag_count(self, x):
        return dict(hashtag_count=len(x.get(u'entities', {}).get(u'hashtags', [])))

    def feature__retweet_count(self, x):
        return dict(retweet_count=x.get(u'retweet_count', 0))
                
    def feature__favorite_count(self, x):
        return dict(favorite_count=x.get(u'favorite_count', 0))

    def feature__question_marks(self, x):
        return prefix_dict_keys(
            pattern_features_dict(
                r'\?', 
                x.get(u'text', ''), 
                count=lambda m: len(m)
            ), 
            'question_mark'
        )

    def feature__colons(self, x):
        return prefix_dict_keys(
            pattern_features_dict(
                r':\s*\w', 
                x.get(u'text', ''), 
                count=lambda m: len(m), 
                start_pos=lambda m: match_start(m, 0)
            ), 
            'colons'
        )
        
    def feature__repeated(self, x):
        return prefix_dict_keys(
            pattern_features_dict(
                r'(\w)\1{2,}', 
                x.get(u'text', ''), 
                count=lambda m: len(m), 
            ), 
            'repeated'
        )
    
    def feature__all_caps(self, x):
        return prefix_dict_keys(
            pattern_features_dict(
                r'\b[A-Z]{2,}\b', 
                x.get(u'text', ''), 
                count=lambda m: len(m), 
            ), 
            'all_caps'
        )
        
    def feature__start_caps(self, x):
        return prefix_dict_keys(
            pattern_features_dict(
                r'\b[A-Z][A-Za-z]+\b', 
                x.get(u'text', ''), 
                count=lambda m: len(m), 
            ), 
            'start_caps'
        )
        
    def feature__punctuation_marks(self, x):
        return prefix_dict_keys(
            pattern_features_dict(
                r'[{0}]'.format(regex.escape(punctuation)),
                x.get(u'text', ''), 
                count=lambda m: len(m), 
            ), 
            'punctuation_marks'
        )

# TODO: Construct as a Pipeline like others defined
# here but is not as trivial as `fit_transform` does 
# not accept argument `X` by default.
class TweetLabelEncoder(LabelEncoder):

    def fit(self, y):
        return super(TweetLabelEncoder, self).fit(self.y_new(y))

    def transform(self, y):
        return super(TweetLabelEncoder, self).transform(self.y_new(y))
    
    def fit_transform(self, y):
        return super(TweetLabelEncoder, self).fit_transform(self.y_new(y))
    
    def y_new(self, y):
        return [self._normalize(self._target(a)) for a in y]

    def _target(self, a):
        return a['class']['overall']

    def _normalize(self, a):
        return u'neutral' if a in ('neutral', 'objective', 'objective-OR-neutral') else a

def make_text_extractor():
    text_steps = [
        ('text_extract', TweetTextExtractor()),
        ('count_vect',
         CountVectorizer(tokenizer=tokenize, preprocessor=preprocess))
    ]
    return ExtractorPipeline(steps=text_steps)


def make_feature_extractor():
    feature_steps = [
        ('features_extract', TweetFeaturesExtractor()),
        ('dict_vect', DictVectorizer())
    ]
    return ExtractorPipeline(steps=feature_steps)


def make_combined_extractor():
    extractors = [
        ('text', make_text_extractor()),
        ('features', make_feature_extractor())
    ]
    return FeatureUnion(extractors)

if __name__ == '__main__':

    from data import load_semeval

    twitter_data = load_semeval(subtask='b', subset='all')

    combined_vec = make_combined_extractor()

    for x in twitter_data[:10]:
        _, feature_extractor = combined_vec.transformer_list[1]
        features = feature_extractor.named_steps[
            'features_extract']._features(x)
        pprint(x)
        pprint(features)
        print

    exit(0)

    X = combined_vec.fit_transform(list(twitter_data[:10]))

    print X.shape
    print combined_vec.get_feature_names()
