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

REPEATED_CHAR_REGEX = re.compile(r'(.)\1{2,}')

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

def normalise_repeated_chars(s):
    return REPEATED_CHAR_REGEX.sub(r'\1\1\1', s)

test = u"Check this sieeeeete http://www.google.com out http://t.co/FNkPfmii- it's great"

print test
print normalize_urls(test)
print normalise_repeated_chars(test)

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