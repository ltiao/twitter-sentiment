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
from sklearn.externals import six

from pprint import pprint, pformat
from string import punctuation
from collections import Counter

from nltk.data import load
from nltk import pos_tag
import regex

from analyzer import TweetPreprocessor, TweetTokenizer

preprocess = TweetPreprocessor()
tokenize = TweetTokenizer()

sentence_tokenizer = load('tokenizers/punkt/english.pickle')
upenn_tagset = load('help/tagsets/upenn_tagset.pickle')

class Bunch(dict):

    def __init__(self, *args, **kwargs):
        self.update(*args, **kwargs)
        self.__dict__ = self

def match_start(match_objects, n):
    try:
        return match_objects[n].start()
    except IndexError:
        return -1

def ends_with(match_objects):
    try:
        match = match_objects[-1]
    except IndexError:
        return False
    return match.end() == len(match.string)

MATCH_FUNCS = dict(
    exists = lambda m: m != [],
    count = lambda m: len(m),
    start_pos = lambda m: match_start(m, 0),
    ends_with = ends_with
)

def get_match_func(func):
    if isinstance(func, six.string_types):
        try:
            func = MATCH_FUNCS[func]
        except KeyError:
            raise ValueError('{0} is not a valid scoring value. Valid options are {1}'.format(func, sorted(MATCH_FUNCS.keys())))        
    return func

def prefix_dict_keys(d, prefix):
    for k in d.keys():
        d['_'.join((prefix, k))] = d.pop(k)
    return d

def pattern_features_dict(pattern, string, *fn_names, **funcs):
    # Just another way of asking: isinstance(pattern, regex._pattern_type)
    try:
        matches = list(pattern.finditer(string))
    except AttributeError:
        matches = list(regex.finditer(pattern, string))

    result = {}   
    
    for fn_name, fn in funcs.items():
        result[fn_name] = fn(matches)
    
    for fn_name in fn_names:
        fn = get_match_func(fn_name)
        result[fn_name] = fn(matches)

    return result
  
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
        return [self.extract_text(x) for x in X]

    def extract_text(self, x):
        return x.get(u'text', '')

# Precompile regex patterns for efficiency
REGEX_PATTERNS = dict(
    question_mark = regex.compile(r'\?+'),
    exclamation_mark = regex.compile(r'!+'),
    colons = regex.compile(r':\s*\w'),
    repeated_chars = regex.compile(r'(\w)\1{2,}'),
    begins_caps = regex.compile(r'\b[A-Z][A-Za-z]+\b'),
    all_caps = regex.compile(r'\b[A-Z]{2,}\b')
)

class TweetWrapper(Bunch):

    def __init__(self, *args, **kwargs):
        super(TweetWrapper, self).__init__(*args, **kwargs)
        self.word_tokens = tokenize(preprocess(self.get(u'text', '')))  
        self.sent_tokens = sentence_tokenizer.tokenize(self.get(u'text', ''))
        self.pos_tag = pos_tag(self.word_tokens)

    def extract(self, *features):
        result = {}
        for feature_name in features:
            feature = getattr(self, feature_name, {})
            if isinstance(feature, dict):
                feature = prefix_dict_keys(feature, feature_name)
                result.update(feature)
            else:
                result[feature_name] = feature
        return result

    @property
    def pos_count(self):
        pos_freq = Counter(tag for _, tag in self.pos_tag)
        return dict(('{}'.format(pos), pos_freq.get(pos, 0)) for pos in upenn_tagset.keys())

    @property
    def is_reply(self):
        return self.get(u'in_reply_to_status_id', None) is not None

    @property
    def char_count(self):
        return len(self.get(u'text', ''))

    @property
    def word_count(self):
        return len(self.word_tokens)
        
    @property
    def sentence_count(self):
        return len(self.sent_tokens)

    @property
    def hashtag_count(self):
        return len(self.get(u'entities', {}).get(u'hashtags', []))

    @property
    def retweet_count(self):
        return self.get(u'retweet_count', 0)

    @property
    def favorite_count(self):
        return self.get(u'favorite_count', 0)

    @property
    def question_mark(self):
        return pattern_features_dict(REGEX_PATTERNS['question_mark'], self.text, 'count', 'start_pos')

    @property
    def exclamation_mark(self):
        return pattern_features_dict(REGEX_PATTERNS['exclamation_mark'], self.text, 'count', 'start_pos')

    @property
    def colons(self):
        return pattern_features_dict(REGEX_PATTERNS['colons'], self.text, 'count', 'start_pos')

    @property
    def repeated_chars(self):
        return pattern_features_dict(REGEX_PATTERNS['repeated_chars'], self.text, 'count')

    @property
    def begins_caps(self):
        return pattern_features_dict(REGEX_PATTERNS['begins_caps'], self.text, 'count')

    @property
    def all_caps(self):
        return pattern_features_dict(REGEX_PATTERNS['all_caps'], self.text, 'count')

class TweetFeaturesExtractor(BaseEstimator, TransformerMixin):

    def __init__(self, features=None):
        self.features = features

    def fit(self, X=None, y=None):
        return self

    def transform(self, X, y=None):
        return [TweetWrapper(x).extract(*self.features) for x in X]

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
