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
import regex 
from htmlentitydefs import name2codepoint
from nltk.tokenize import RegexpTokenizer
from nltk.internals import convert_regexp_to_nongrouping

HTML_ENTITY_REGEX = regex.compile('&#?(?P<entity>\d+|{0});'.format('|'.join(name2codepoint)))

# TODO: Unicode support
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

# TODO: Trailing latin accent, symbol and URL detection
MENTION_REGEX = regex.compile(
    ur"""
    (?<=[^a-zA-Z0-9_!#\$%&*@＠]|^|RT:?)
    [@＠]
    [a-zA-Z0-9_]{1,20}
    (\/[a-zA-Z][a-zA-Z0-9_\-]{0,24})?
    """, regex.UNICODE | regex.VERBOSE | regex.IGNORECASE)

REPEATED_CHAR_REGEX = regex.compile(r'(\w)\1{2,}')

class MultiSub(object):

    def __init__(self, subs, *args, **kwargs):
        self.subs = subs
        self.regex = regex.compile(
            r'|'.join(
                '({pattern})'.format(pattern=convert_regexp_to_nongrouping(key)) for key in subs
            ), *args, **kwargs
        )

    def _repl(self, m):
        repl = self.subs.values()[m.lastindex-1]
        if callable(repl):
            return repl(m)
        return repl 

    def sub(self, string, *args, **kwargs):
        return self.regex.sub(self._repl, string, *args, **kwargs)

a = MultiSub(subs={r'something': 'nothing', r'\s\w\w\w\s': lambda m: 'test'}, flags=regex.UNICODE | regex.VERBOSE | regex.IGNORECASE)

print a.sub('this is something lol smh')

exit(0)
def decode_html_entities(string, repl=None, count=0):

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
            
    if repl is None: 
        repl = repl_func

    return HTML_ENTITY_REGEX.sub(repl, string, count)
    
def normalize_urls(string, repl='URL', count=0, incl_host=False):
    
    if incl_host:
        repl = lambda m: '-'.join(('URL', m.group('host')))
        
    return URL_REGEX.sub(repl, string, count)
    
def normalize_mentions(string, repl='@MENTION', count=0):
    return MENTION_REGEX.sub(repl, string, count)

def normalize_repeated_chars(string, repl=r'\1\1\1', count=0):
    return REPEATED_CHAR_REGEX.sub(repl, string, count)

print normalize_urls('this is a quick test http://ltiao.github.io#test- hahaha!', incl_host=False)