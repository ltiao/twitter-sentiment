#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
from nltk.tokenize import RegexpTokenizer

REGEX_CONSTANTS = {
    'at_signs':             ur'[@\uff20]',
    'utf_chars':            ur'a-z0-9_\u00c0-\u00d6\u00d8-\u00f6\u00f8-\u00ff',
    'spaces':               ur'[\u0020\u00A0\u1680\u180E\u2002-\u202F\u205F\u2060\u3000]',
    'list_pre_chars':       ur'(?:[^a-z0-9_]|^)',
    'list_end_chars':       ur'(?:[a-z0-9_]{1,20})(?:/[a-z][a-z0-9\x80-\xFF-]{0,79})?',
    'pre_chars':            ur'(?:[^/"\':!=]|^|\:)',
    'domain_chars':         ur'([\.-]|[^\s_\!\.\/])+\.[a-z]{2,}(?::[0-9]+)?',
    'path_chars':           ur'(?:[\.,]?[{utf_chars}!\*\'\(\);:=\+\$/\%#\[\]\-_,~@])',
    'query_chars':          ur'[a-z0-9!\*\'\(\);:&=\+\$/%#\[\]\-_\.,~]',
    'path_ending_chars':    ur'[{utf_chars}\)=#/]',
    'query_ending_chars':   ur'[a-z0-9_&=#]',
}

# print HASHTAG_REGEX.encode('utf-8')

class TwitterTokenizer(RegexpTokenizer):

    # class variables
    REGEX_STRINGS = {
        # Most of the Western emoticons: http://en.wikipedia.org/wiki/List_of_emoticons#Western
        # Based on Christopher Pott's Tutorial: http://sentiment.christopherpotts.net/tokenizing.html
        # TODO: *   Add support for '<3' '</3' and fix co-occurrences such as '83' 
        #           which is currently matched and plausibly an emoticon but isn't really.
        #       * Phone numbers
        #       * Dates
        'emoticons': ur"""
                [<>]?                             # optional hat or frown
                [:;=8xX]                          # eyes
                [\-o\*\'\^cっ]?                    # optional nose
                [\)\]\(\[dDpP/\:\{{}}@\|\\3cC<>]   # mouth      
                |                                 # disjunction: the reverse orientation
                [\)\]\(\[dD/\:\{{}}@\|\\<>]    
                [\-o\*\'\^cっ]?                    
                [:=8xX]                     
                [<>]?
            """,
        'hashtags': ur"""
                (?:^|[^0-9A-Z&/]+)
                [#\uff03]
                [0-9A-Z_]*[A-Z_]+[{utf_chars}]*
            """,
        'usernames': ur"""
              \B
              {at_signs}
              {list_end_chars}
            """,
        'replies': ur"""
                ^(?:{spaces})*
                {at_signs}
                [a-z0-9_]{{1,20}}
                .*
            """,
        'lists': ur"""
                {list_pre_chars}
                {at_signs}+
                {list_end_chars}
            """,
        'nonwhitespace': ur"""
                (?:[a-z][a-z'\-_]+[a-z])       # Words with apostrophes or dashes.
                |
                (?:[+\-]?\d+[,/.:-]\d+[+\-]?)  # Numbers, including fractions, decimals.
                |
                (?:[\w_]+)                     # Words without apostrophes or dashes.
                |
                (?:\.(?:\s*\.){{1,}})            # Ellipsis dots. 
                |
                (?:\S)                         # Everything else that isn't whitespace.
            """,
        # Omit URLs for now.
        # N.B. The following does not work with unicode yet.
         'url': ur'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    }

    def __init__(self):
        pattern = ur'|'.join(TwitterTokenizer.REGEX_STRINGS.values()).format(**REGEX_CONSTANTS)
        RegexpTokenizer.__init__(self, pattern=pattern, flags=re.UNICODE | re.MULTILINE | re.VERBOSE | re.IGNORECASE)

    def __call__(self, text):
      return self.tokenize(text)

if __name__ == "__main__":
    from twitter_text.regex import REGEXEN
    
    from pymongo import MongoClient
    client = MongoClient()
    db = client.twitter_database
    db_tweets = db.tweets

    twokenizer = TwitterTokenizer()

    print twokenizer('#yolo this is a @louistiao SOMETHING test :) text http://example.com/test/foo_123.jpg')
    print twokenizer('big url: http://blah.com:8080/path/to/here?p=1&q=abc,def#posn2 #ahashtag http://t.co/FNkPfmii-')

    from nltk.internals import convert_regexp_to_nongrouping

    print REGEXEN['valid_url'].pattern.encode('utf-8')

    print re.compile(convert_regexp_to_nongrouping(REGEXEN['valid_url'].pattern)).findall('big http://blah.com:8080/path/to/here?p=1&q=abc,def#posn2 #ahashtag http://t.co/FNkPfmii-')

    print convert_regexp_to_nongrouping(REGEXEN['valid_url'].pattern).encode('utf-8')

    print REGEXEN['valid_tco_url'].pattern.encode('utf-8')

    exit(0)
    for tweet in db_tweets.find(
            {
                u'text': 
                {
                    '$exists': True,
                    # '$regex': ':\)'
                }
            }
        ):
        print tweet.get(u'text')
        print twokenizer.tokenize(tweet.get(u'text'))