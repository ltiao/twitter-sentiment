#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.metrics import confusion_matrix
from itertools import product

y_true, y_pred = np.array([0, 1, 0, 2, 1, 1]), np.array([0, 0, 0, 2, 1, 2])
confusion = np.array([[np.logical_and(y_true==r, y_pred==c) for c in xrange(3)] for r in xrange(3)])

print np.sum(confusion, axis=2)
print confusion_matrix(y_true, y_pred)

print y_pred[confusion[0, 0]]

exit(0)

print y_true < 2

print confusion_matrix(y_true, y_pred)

from scipy.sparse import coo_matrix
from sklearn.utils.multiclass import unique_labels

print np.asarray(
    coo_matrix(
        (
            np.ones(
                y_true.shape[0], dtype=np.int
            ), 
            (y_true, y_pred)
        ), 
        shape=(3, 3)
    ).todense()
)


exit(0)


import re
from nltk.internals import convert_regexp_to_nongrouping

from htmlentitydefs import name2codepoint

HTML_ENTITY_REGEX = re.compile('&#?(?P<entity>\d+|{0});'.format('|'.join(name2codepoint)))

URL_REGEX = re.compile(
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



MENTION_REGEX = re.compile(
    ur"""
    ([^a-zA-Z0-9_!#\$%&*@＠]|^|RT:?)
    [@＠]
    [a-zA-Z0-9_]{1,20}
    (\/[a-zA-Z][a-zA-Z0-9_\-]{0,24})?
    """, re.UNICODE | re.VERBOSE)

REPEATED_CHAR_REGEX = re.compile(r'(\w)\1{2,}')

def decode_html_entities(s):

    def repl_func(m):
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
            
    return HTML_ENTITY_REGEX.sub(repl_func, s)

def normalize_urls(s, incl_domain=False):
    
    def repl_func(m):
        return '[URL-{0}]'.format(m.group('host'))

    if incl_domain:
        repl = '[URL]'
    else:
        repl = repl_func
    return URL_REGEX.sub(repl, s)

def normalize_repeated_chars(s):
    return REPEATED_CHAR_REGEX.sub(r'\1\1\1', s)

def normalize_mentions(s):
    
    def repl_func(m):
        return m.group(1) + '[@MENTION]'
    
    return MENTION_REGEX.sub(repl_func, s)

test = u"Check @_@ this ＠test sieee f@@k sh@#t eete @tiao http://www.google.com out http://t.co/FNkPfmii- it's great lolololol hahahahahah"

print test
print normalize_urls(test)
print normalize_repeated_chars(test)
print normalize_mentions(test)

exit(0)

from pymongo import MongoClient

client = MongoClient()

db = client.twitter_database
db_labeled_tweets = db.labeled_tweets

twts = db_labeled_tweets.find({u'text': re.compile(r'\d+')})

for twt in twts:
    print twt[u'text']
    print decode_html_entities(twt[u'text'])
    print

print decode_html_entities('&#12211;')

exit(0)


pattern_re = re.compile(convert_regexp_to_nongrouping(pattern), re.IGNORECASE | re.VERBOSE)

print test
print pattern_re.findall(test)
print pattern_re.sub('{URL}', test)

exit(0)

import twitter_text

extractor = twitter_text.extractor.Extractor(test)

print extractor.extract_urls()

class Replacer(object):

	def __init__(self, replacements):
		self.replacements = replacements
		self.locator = re.compile('|'.join(replacements.values()))

	def __call__(self, doc):
		return self.replace(doc)

	def _replace_with(self, match):
		print dir(match.group())
		return self.replacements[match.group()]

	def replace(self, doc):
		return self.locator.sub(self._replace_with, doc)
		
class MultiRegex(object):

    def __init__(self, substitutions, *args, **kwargs):
        """
        compile a disjunction of regexes, in order
        """
        self._substitutions = substitutions
        self._regex_compiled = re.compile("|".join(self._substitutions), *args, **kwargs)

	def __call__(self, s):
		return self.sub(s)

    def sub(self, s):
        return self._regex_compiled.sub(self._sub, s)

    def _sub(self, mo):
        '''
        determine which partial regex matched, and
        dispatch on self accordingly.
        '''
        groupdict = mo.groupdict()
        print mo.groups()
        return '#TEMP#'
        for k in groupdict:
            if groupdict.get(k):
                sub = getattr(self, k)
                if callable(sub):
                    return sub(mo)
                return sub
        raise AttributeError, 'nothing captured, matching sub-regex could not be identified'



r = MultiRegex((r'(\s\w\w\w\s)', r'(\s\w\w\w\w\s)'))
print test
print r.sub(test)