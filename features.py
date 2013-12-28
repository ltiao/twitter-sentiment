from tokenize import TwitterTokenizer
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(tokenizer=TwitterTokenizer())

if __name__ == "__main__":
	from pymongo import MongoClient
	client = MongoClient()
	db = client.twitter_database
	db_tweets = db.tweets

	analyzer = vectorizer.build_analyzer()
	print analyzer("This is a text document to analyze.")

	print vectorizer.fit_transform(t.get(u'text') for t in db_tweets.find({u'text': {'$exists': True}}))
	print vectorizer.get_feature_names()