#!/usr/bin/env python
# -*- coding: utf-8 -*-

import regex
from htmlentitydefs import name2codepoint

HTML_ENTITY_REGEX = regex.compile('&#?(?P<entity>\d+|{0});'.format('|'.join(name2codepoint)))

URL_REGEX = regex.compile(
    r"""[a-z][a-z0-9+\-.]*://                                       # Scheme
    ([a-z0-9\-._~%!$&'()*+,;=]+@)?                                  # User
    (?P<host>[a-z0-9\-._~%]+                                        # Named Host
    |\[[a-f0-9:.]+\]                                                # IPv6 host
    |\[v[a-f0-9][a-z0-9\-._~%!$&'()*+,;=:]+\][a-z0-9+&@#/%=~_|$])   # IPvFuture host
    (:[0-9]+)?                                                      # Port
    (/[a-z0-9\-._~%!$&'()*+,;=:@]+[a-z0-9+&@#/%=~_|$])*/?           # Path
    (\?[a-z0-9\-._~%!$&'()*+,;=:@/?]*[a-z0-9+&@#/%=~_|$])?          # Query
    (\#[a-z0-9\-._~%!$&'()*+,;=:@/?]*[a-z0-9+&@#/%=~_|$])?          # Fragment
    """, regex.IGNORECASE | regex.VERBOSE)

MENTION_REGEX = regex.compile(
    ur"""
    (?<=[^a-zA-Z0-9_!#\$%&*@＠]|^|RT:?)
    [@＠]
    [a-zA-Z0-9_]{1,20}
    (\/[a-zA-Z][a-zA-Z0-9_\-]{0,24})?
    """, regex.UNICODE | regex.VERBOSE | regex.IGNORECASE)

REPEATED_CHAR_REGEX = regex.compile(r'(\w)\1{2,}')


class TwitterTextPreprocessor(object):
    
    def __init__(self):
        pass

    def __call__(self, s):
        return self.preprocess(s)

    def preprocess(self, s):
        func_list = (self.decode_html_entities, self.normalize_urls, self.normalize_mentions, self.normalize_repeated_chars)
        return reduce(lambda x, y: y(x), func_list, s)

    def decode_html_entities(self, s):

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

    def normalize_urls(self, s, incl_domain=False):
        
        def repl_func(m):
            return '[URL-{0}]'.format(m.group('host'))

        if incl_domain:
            repl = repl_func
        else:
            repl = '[URL]'
        return URL_REGEX.sub(repl, s)

    def normalize_mentions(self, s, repl_func='@MENTION'):
        return MENTION_REGEX.sub(repl_func, s)

    def normalize_repeated_chars(self, s):
        return REPEATED_CHAR_REGEX.sub(r'\1\1\1', s)

if __name__ == '__main__':

    print reduce(lambda x, y: '{0}({1})'.format(y, x), ['f', 'g', 'h'], 'x')

    preprocessor = TwitterTextPreprocessor()

    from pymongo import MongoClient

    client = MongoClient()

    db = client.twitter_database
    db_labeled_tweets = db.labeled_tweets

    for tweet in db_labeled_tweets.find({u'text': {'$exists': True}}):
        text = tweet.get(u'text')
        print tweet.get(u'_id')
        print text
        # print decode_html_entities(text)
        # print normalize_urls(text)
        # print normalize_repeated_chars(text)
        # print normalize_mentions(text)
        # print normalize_urls(normalize_repeated_chars(normalize_mentions(decode_html_entities(text))))
        print preprocessor(text).encode('utf-8')
        print