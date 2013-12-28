import csv, pprint
from pymongo import MongoClient
from bson.objectid import ObjectId

client = MongoClient()

db = client.twitter_database
db_labeled_tweets = db.labeled_tweets

# Datasets

tasks = ['a', 'b']
dataset_file_modifiers = ['', '.dev']

dataset_filename = 'tweeti-{0}.dist.tsv'

for task in tasks:
    for modifier in dataset_file_modifiers:
        print 'Reading [{}]...'.format(dataset_filename.format(task+modifier))
        with open(dataset_filename.format(task+modifier)) as tsv:
            if task == 'a':
                update =  {
					'$addToSet':
					{
						u'class.expressions': {}
					}
				}
                for tweet_id_str, user_id_str, start_str, end_str, label in csv.reader(tsv, dialect='excel-tab'):
                    tweet_id, start, end = map(int, (tweet_id_str, start_str, end_str))
                    tweet = db_labeled_tweets.find_one({u'_id': tweet_id, u'text': {'$exists': True}})
                    if tweet:
                        update['$addToSet'][u'class.expressions'] = {
                            u'indices': (start, end),
                            u'expression': ' '.join(tweet.get(u'text', '').split()[start:end+1]),
                            u'class': label
                        }
                        print tweet.get(u'text', '')
                        pprint.pprint(update)
                        print db_labeled_tweets.update({u'_id': tweet_id}, update)
                        print
            else: 
                for tweet_id_str, user_id_str, label in csv.reader(tsv, dialect='excel-tab'):
                    tweet_id = int(tweet_id_str)
                    print 'Updating tweet {0} with class {1}'.format(tweet_id, label)
                    print db_labeled_tweets.update({u'_id': tweet_id}, {'$set': {u'class.overall': label}})