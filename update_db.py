import csv, math, pprint, logging.config
import twitter
from pymongo import MongoClient
from settings import LOGGING
import time, datetime
from collections import Counter
from utils import robust

# Logging

logging.config.dictConfig(LOGGING)
logger = logging.getLogger('twitter')

# Twitter

CONSUMER_KEY = 'yPLz4cxR6iuqRRN1TptnVg'
CONSUMER_SECRET = 'R5JoXqVsR9ExGqUh4jvNBMvTjQJzemFSphgprZbVg'
OAUTH_TOKEN = '2215208869-avz1Tg2A2bjyll7KdMg9GTg3E4fvVWJNjdpuvW2'
OAUTH_TOKEN_SECRET = 'SaABeAmn1zEX6HOcfNEjKT6VLB7apF30ky0X3qsT3wJhn'

auth = twitter.oauth.OAuth(OAUTH_TOKEN, OAUTH_TOKEN_SECRET, CONSUMER_KEY, CONSUMER_SECRET)

twitter_api = twitter.Twitter(auth=auth)

# Data Persistence

client = MongoClient()
db = client.twitter_database
db_tweets = db.tweets

# Datasets

tasks = ['a', 'b']
dataset_file_modifiers = ['', '.dev']

dataset_filename = 'tweeti-{task}.dist.tsv'

def rate_limit_status_string():
    rate_limit = twitter_api.application.rate_limit_status(resources='statuses')
    show_status_limit = rate_limit[u'resources'][u'statuses'][u'/statuses/show/:id']
    return 'Limit: {lim} | Remaining: {rem} | Reset: {reset}'.format(
                lim = show_status_limit[u'limit'],
                rem = show_status_limit[u'remaining'],
                reset = datetime.datetime.fromtimestamp(show_status_limit[u'reset'])
            )

# Start

logger.info(rate_limit_status_string())

# pprint.pprint(list(db_tweets.find()))

for task in tasks:
    for modifier in dataset_file_modifiers:
        task += modifier
        logger.info('Reading [{}]...'.format(dataset_filename.format(task=task)))
        with open(dataset_filename.format(task=task)) as tsv:
            c = Counter(row[0] for row in csv.reader(tsv, dialect='excel-tab'))
            num_uniq = len(c)
            num_lines = len(list(c.elements()))
            logger.info('Found {} lines and {} unique'.format(num_lines, num_uniq))
        with open(dataset_filename.format(task=task)) as tsv:
            rate_limit_reset = twitter_api.application.rate_limit_status(resources='statuses')[u'resources'][u'statuses'][u'/statuses/show/:id'][u'reset']
            for i, row in enumerate(csv.reader(tsv, dialect='excel-tab')):
                tweet_id = int(row[0])
                logger.info('[{}] Processing line {}/{} (tweet: {}): {:.2%} complete...'.format(dataset_filename.format(task=task), i, num_lines, tweet_id, i/float(num_lines)))
                if db_tweets.find_one({'_id': tweet_id}): 
                    logger.info('\t- tweet {} has already been retrieved and stored.'.format(tweet_id))
                    continue

                logger.info('\t- querying Twitter API...')
                tweet = robust(twitter_api.statuses.show, rate_limit_reset=rate_limit_reset, id=tweet_id, trim_user=True)

                if tweet is not None:
                    try:
                        rate_limit_reset = float(tweet.headers.getheader('x-rate-limit-reset'))
                        logger.info('** Limit: {lim} | Remaining: {rem} | Reset: {reset} **'.format(
                            lim = tweet.headers.getheader('x-rate-limit-limit'),
                            rem = tweet.headers.getheader('x-rate-limit-remaining'),
                            reset = datetime.datetime.fromtimestamp(rate_limit_reset)
                        ))
                        tweet[u'_id'] = tweet.pop(u'id')
                        logger.info('\t- storing in MongoDB...')
                        db_tweets.save(tweet)
                    except AttributeError:
                        logger.info('\t- storing status code in MongoDB...')
                        db_tweets.save({u'_id': tweet_id, u'code': tweet})
                    except:
                        logger.exception('\t- Unforeseen Error Occurred')
                    else:
                        logger.info('Successfully processed!')
