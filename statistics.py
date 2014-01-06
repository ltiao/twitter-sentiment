from pymongo import MongoClient

import re

client = MongoClient()
db = client.twitter_database
db_labeled_tweets = db.labeled_tweets

EMOTICONS_REGEX = re.compile(r"""
    [<>]?                       # optional hat/brow
    [:;=8]                      # eyes
    [\-o\*\']?                  # optional nose
    [\)\]\(\[dDpP/\:\}\{@\|\\]  # mouth      
    |                           # reverse orientation
    [\)\]\(\[dDpP/\:\}\{@\|\\]  # mouth
    [\-o\*\']?                  # optional nose
    [:;=8]                      # eyes
    [<>]?                       # optional hat/brow
    """, re.VERBOSE)

r'[@\|\\/]'

HEART_REGEX = re.compile(r'(?:<|&lt;)3')
HEARTBREAK_REGEX = re.compile(r'(?:<|&lt;)/3')

SMILEY_REGEX = re.compile(r"""
    <?                  # optional hat, brow considered evil
    [:;=8]              # eyes
    [\-o\*\']?          # optional nose
    [\)\]\}DpP]         # mouth      
    |                   # reverse orientation
    [\(\[\{]            # mouth      
    [\-o\*\']?          # optional nose
    [:;=8]              # eyes
    >?                  # optional hat, brow considered evil
    """, re.VERBOSE)

FROWNIE_REGEX = re.compile(r"""
    [<>]?           # optional hat/brow
    [:;=8]          # eyes
    [\-o\*\',]?      # optional nose
    [\(\[\{]  # mouth    
    |               # reverse orientation
    \B
    [\)\]\}]  # mouth      
    [\-o\*\',]?      # optional nose
    [:;=8]          # eyes
    [<>]?           # optional hat/brow
    """, re.VERBOSE)

import numpy as np
import scipy.stats

def contingency_table(emoticons):
    classes = {
        'Positive': 'positive',
        'Negative': 'negative',
        'Neutral': {
            '$in': ['neutral', 'objective', 'objective-OR-neutral']
        }
    }
    obs = np.array([[db_labeled_tweets.find({u'text': emoticons[a], u'class.overall': classes[cls_]}).count() for cls_ in classes] for a in emoticons])
    row_format ="{:>19}" * (len(classes) + 1)
    print 'Contingency Table:'
    print
    print row_format.format("", *classes)
    for heart, row in zip(emoticons, obs):
        print row_format.format(heart, *row)
    print
    print 'Chi-squared test of independence:'
    print
    print '\tPearson Chi-squared: {0} | DF: {2} | P-Value: {1}'.format(*scipy.stats.chi2_contingency(obs))
    print 50 * '-'

contingency_table({
        'Happy emoticons': SMILEY_REGEX,
        'Sad/angry emoticons': FROWNIE_REGEX
    })

contingency_table({
        'Heart (<3)': HEART_REGEX,
        'Heartbroken (</3)': HEARTBREAK_REGEX
    })

exit(0)

print 'Usable tweets'
print db_labeled_tweets.find({u'text': {'$exists': True}}).count()

for tweet in db_labeled_tweets.find({u'text': FROWNIE_REGEX}):
    print tweet.get(u'text')
    print