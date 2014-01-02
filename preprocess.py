#!/usr/bin/env python
# -*- coding: utf-8 -*-

from htmlentitydefs import name2codepoint
import re

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
    (?:(?<![a-zA-Z0-9_!#\$%&*@＠])|(?<=^)|(?<=RT:)|(?<=RT))
    [@＠]
    [a-zA-Z0-9_]{1,20}
    (\/[a-zA-Z][a-zA-Z0-9_\-]{0,24})?
    """, re.UNICODE | re.VERBOSE)

REPEATED_CHAR_REGEX = re.compile(r'(\w)\1{2,}')

class Processor(object):
    
    def __init__(self):
        pass

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
        repl = repl_func
    else:
        repl = '[URL]'
    return URL_REGEX.sub(repl, s)

def normalize_repeated_chars(s):
    return REPEATED_CHAR_REGEX.sub(r'\1\1\1', s)

def normalize_mentions(s):
    
    def repl_func(m):
        return '[@MENTION]'
    
    return MENTION_REGEX.sub(repl_func, s)

if __name__ == '__main__':
    test = u"Check @_@ this ＠test sieee f@@k sh@#t eete @tiao http://www.google.com out http://t.co/FNkPfmii- it's great lolololol hahahahahah"

    print test
    print normalize_urls(test)
    print normalize_repeated_chars(test)
    print normalize_mentions(test)
    
    from pymongo import MongoClient

    client = MongoClient()

    db = client.twitter_database
    db_labeled_tweets = db.labeled_tweets
    
    for tweet in db_labeled_tweets.find({u'text': {'$exists': True}}):
        text = tweet.get(u'text')
        print text
        print decode_html_entities(text)
        print normalize_urls(text)
        print normalize_repeated_chars(text)
        print normalize_mentions(text)
        print normalize_urls(normalize_repeated_chars(normalize_mentions(decode_html_entities(text) )))
        print
        
    m = re.search('(?<=abc)def', 'abcdef def defdef ab')
    print m.group()