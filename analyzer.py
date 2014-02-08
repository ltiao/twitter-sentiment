#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Using `regex` instead of the built-in `re` to
# support variable-length look-behind assertions.
# I decided to do this because Java's regular
# expression engine supports this by default and
# I eventually want to extract all regular expressions
# patterns to a single YAML file that can be used for 
# both preprocessing (normalization) and also tokenization
# in both Python and Java (and other languages) as-is. 
# (similar to UA-Parser: https://github.com/tobie/ua-parser)

import regex as re # http://xkcd.com/1171/
from htmlentitydefs import name2codepoint

from nltk.tokenize import RegexpTokenizer
from nltk.tokenize.api import TokenizerI
from nltk.internals import convert_regexp_to_nongrouping
from nltk import pos_tag

from collections import defaultdict, OrderedDict

from sklearn.base import BaseEstimator, TransformerMixin

__all__ = ['TweetTokenizer', 'TweetPreprocessor']

def repl_html_entity(m):
    entity = m.group('entity') # get the entity name or number
    try:
        # if integer
        codepoint = int(entity)       
    except ValueError: 
        # not integer - it must be named and therefore 
        # in name2codepoint (i.e. codepoint is never None)
        codepoint = name2codepoint.get(entity)
    # if codepoint > 16**2, or for some other 
    # reason we cannot encode, just leave as-is
    try:
        return unichr(codepoint)
    except ValueError:
        return m.group()

REGEXES = defaultdict(dict)

REGEXES['html_entity']['regex'] = re.compile('&#?(?P<entity>\d+|{0});'.format('|'.join(name2codepoint)))
REGEXES['html_entity']['repl'] = repl_html_entity

# TODO: Unicode support
REGEXES['url']['regex'] = re.compile(
    r"""[a-z][a-z0-9+\-.]*://                                       # Scheme
    ([a-z0-9\-._~%!$&'()*+,;=]+@)?                                  # User
    (?P<host>[a-z0-9\-._~%]+                                        # Named Host
    |\[[a-f0-9:.]+\]                                                # IPv6 host
    |\[v[a-f0-9][a-z0-9\-._~%!$&'()*+,;=:]+\][a-z0-9+&@#/%=~_|$])   # IPvFuture host
    (:[0-9]+)?                                                      # Port
    (/[a-z0-9\-._~%!$&'()*+,;=:@]+[a-z0-9+&@#/%=~_|$])*/?           # Path
    (\?[a-z0-9\-._~%!$&'()*+,;=:@/?]*[a-z0-9+&@#/%=~_|$])?          # Query
    (\#[a-z0-9\-._~%!$&'()*+,;=:@/?]*[a-z0-9+&@#/%=~_|$])?          # Fragment
    """, re.IGNORECASE | re.VERBOSE)
REGEXES['url']['repl'] = 'URL'

# TODO: Trailing latin accent, symbol and URL detection
REGEXES['mention']['regex'] = re.compile(
    ur"""
    (?<=[^a-zA-Z0-9_!#\$%&*@＠]|^|[Rr][Tt]:?)
    [@＠]
    [a-zA-Z0-9_]{1,20}
    (\/[a-zA-Z][a-zA-Z0-9_\-]{0,24})?
    """, re.UNICODE | re.VERBOSE | re.IGNORECASE)
REGEXES['mention']['repl'] = '@MENTION'

REGEXES['repeated_chars']['regex'] = re.compile(r'(\w)\1{2,}')
REGEXES['repeated_chars']['repl'] = r'\1\1\1'

REGEXES['emoticons']['regex'] = re.compile(
    r"""
    [<>]?
    [:;=8]                      # eyes
    [\-o\*\']?                  # optional nose
    [\)\]\(\[DpP/\:\}\{@\|\\]   # mouth      
    |
    [\)\]\(\[d/\:\}\{@\|\\]     # mouth
    [\-o\*\']?                  # optional nose
    [:;=8]                      # eyes
    [<>]?
    """, re.UNICODE | re.VERBOSE)
    
REGEXES['hashtag']['regex'] = re.compile(r'\#+[\w_]+[\w\'_\-]*[\w_]+')
    
REGEXES['words']['regex'] = re.compile(
    r"""
    [a-z][a-z'\-_]+[a-z]        # Words with apostrophes or dashes.
    |
    [+\-]?\d+[,/.:-]\d+[+\-]?   # Numbers, including fractions, decimals.
    |
    [\w_]+                      # Words without apostrophes or dashes.
    |
    \.(?:\s*\.){1,}             # Ellipsis dots. 
    |
    \S                          # Everything else that isn't whitespace.
    """, re.UNICODE | re.VERBOSE | re.IGNORECASE)

negation_words = [
    'never',
    'no',
    'nothing',
    'nowhere',
    'noone',
    'none',
    'not',
    'havent',
    'hasnt',
    'hadnt',
    'cant',
    'couldnt',
    'shouldnt',
    'wont',
    'wouldnt',
    'dont',
    'doesnt',
    'didnt',
    'isnt',
    'arent',
    'aint',
    'n\'t',
    'hardly',
    'far from',
]

end_punctuation = ['.', ':', ';', '!', '?']

REGEXES['negation']['regex'] = re.compile(
    r"""
    (?<=({0})\s+)
    [^{1}]*
    (?=[{1}]|$)
    """.format(
        '|'.join(re.escape(word) for word in negation_words), 
        r''.join(re.escape(punc) for punc in end_punctuation)
    ), 
    flags = re.IGNORECASE | re.VERBOSE
)

REGEXES['negation']['repl'] = lambda mo: ' '.join('{}_NEG'.format(token) for token in mo.group().split())

class TweetPreprocessor(object):

    def __init__(self, *steps):
        if steps:
            self.steps = steps
        else:
            self.steps = ['html_entity', 'url', 'mention', 'repeated_chars']
    
    def __call__(self, text):
        return self.preprocess(text)

    def preprocess(self, text):
        for k in self.steps:
            repl = REGEXES[k]['repl']
            text = REGEXES[k]['regex'].sub(repl, text)
        return text

class TweetTokenizer(TokenizerI):

    def __init__(self, pos_tagging=False):
        self.pos_tagging = pos_tagging
        pattern = ur'|'.join(REGEXES[k]['regex'].pattern for k in ('url', 'emoticons', 'mention', 'hashtag', 'words'))
        nongrouping_pattern = convert_regexp_to_nongrouping(pattern)
        self._regexp = re.compile(nongrouping_pattern, flags=re.UNICODE | re.MULTILINE | re.VERBOSE | re.IGNORECASE)

    def __call__(self, text):
      return self.tokenize(text)

    def tokenize(self, text):
        tokens = self._regexp.findall(text)
        if self.pos_tagging:
            return ['_'.join(tagged) for tagged in pos_tag(tokens)]
        else:
            return tokens
            