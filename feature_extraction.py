#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Louis Tiao <Louis.Tiao@csiro.au>
#
# License: ?

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

from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin

from pprint import pprint, pformat
from string import punctuation
from nltk.data import load
import regex

from analyzer import preprocess, tokenize

sentence_tokenizer = load('tokenizers/punkt/english.pickle')

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
        return [self._text(x) for x in X]
    
    def _text(self, x):
        return x.get(u'text', '')

class TweetFeaturesExtractor(BaseEstimator, TransformerMixin):
        
    def fit(self, X=None, y=None):
        return self
    
    def transform(self, X, y=None):
        return [self._features(x) for x in X]
    
    def _features(self, x):
        analyzer = lambda s: tokenize(preprocess(s))
        
        tweet_text = x.get(u'text', '')
        
        word_tokens = analyzer(tweet_text)
        sent_tokens = sentence_tokenizer.tokenize(tweet_text)
        hashtags = x.get(u'entities', {}).get(u'hashtags', [])

        features_dict_ = dict(
            is_reply = x.get(u'in_reply_to_status_id', None) is not None,
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
            features_dict_.update(self._pattern_features_dict(string=tweet_text, prefix=k, **d[k]))
        
        return features_dict_

    def _pattern_features_dict(self, pattern, string, prefix=None, check_match=False, count_matches=False, n_start=None):
        # Just another way of asking: isinstance(pattern, regex._pattern_type)
        try:
            matches = list(pattern.finditer(string))
        except AttributeError:
            matches = list(regex.finditer(pattern, string))

        # logger.debug('{}:\n\tPattern: {}\n\tMatches: {}'.format(prefix, pattern, [mo.group() for mo in matches]))

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
        
def make_text_extractor():
    text_steps = [
        ('text_extract', TweetTextExtractor()),
        ('count_vect', CountVectorizer(tokenizer=tokenize, preprocessor=preprocess))
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
        features = feature_extractor.named_steps['features_extract']._features(x)
        pprint(x)
        pprint(features)
        print
        
    exit(0)
    
    X = combined_vec.fit_transform(list(twitter_data[:10]))

    print X.shape
    print combined_vec.get_feature_names()