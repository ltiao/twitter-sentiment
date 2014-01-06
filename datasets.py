#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tokenizer import TwitterTokenizer
from preprocess import TwitterTextPreprocessor

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

class Bunch(dict):
    """Container object for datasets: dictionary-like object that
       exposes its keys as attributes."""

    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self.__dict__ = self

def load_semeval(subtask='b', subset='all'):
	try:
		from pymongo import MongoClient
	except ImportError:
		raise ImportError('pymongo must be installed to retrieve data from MongoDB')

	client = MongoClient()
	db = client.twitter_database
	db_labeled_tweets = db.labeled_tweets

	q = {
		u'text': {'$exists': True}, 
		u'class.overall': {'$exists': True}
	}

	if subtask == 'a':
		raise NotImplementedError('SemEval-2013 Task 2 Subtask A data not yet supported')
	elif subtask == 'b':
		if subset in ('train', 'test'):
			q[u'class.training'] = (subset == 'train')
		elif subset != 'all':
			raise ValueError("'{}' is not a valid subset: should be one of ['train', 'test', 'all']".format(subset))	
	else:
		raise ValueError("'{}' is not a valid subtask: should be one of ['a', 'b']".format(subtask))

	tweets = list(db_labeled_tweets.find(q))
	
	if not tweets:
		raise RuntimeError('Could not retrieve any data with query {}'.format(q))

	tweets_target = [u'neutral' if tweet['class']['overall'] in ('neutral', 'objective', 'objective-OR-neutral') else tweet['class']['overall'] for tweet in tweets]

	vect = CountVectorizer(tokenizer=TwitterTokenizer(), preprocessor=TwitterTextPreprocessor())
	le = LabelEncoder()

	return Bunch(
			data = vect.fit_transform(tweet.get(u'text') for tweet in tweets),
			target = le.fit_transform(tweets_target),
			target_names = le.classes_
		)


if __name__ == '__main__':
	tweets = load_semeval(subtask='b', subset='train')
	print tweets.data.shape
	print tweets.target_names