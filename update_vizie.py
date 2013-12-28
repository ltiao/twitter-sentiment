import csv, json, pprint, requests, time, datetime, logging.config
from settings import LOGGING
from pymongo import MongoClient
from collections import Counter, defaultdict

WINDOW_SIZE = 15*60 # in seconds
RATE_LIMIT = 180 # per window

delay = WINDOW_SIZE/float(RATE_LIMIT) + 1 # delay with small buffer

# Logging

logging.config.dictConfig(LOGGING)
logger = logging.getLogger('twitter')

# Data Persistence

client = MongoClient()
db = client.twitter_database
db_vizie_ref = db.vizie_reference

# Datasets

tasks = ['a', 'b']
dataset_file_modifiers = ['', '.dev']

dataset_filename = 'tweeti-{task}.dist.tsv'

# Vizie

ACCESS_TOKEN = '262d4ec27de6440738c9e237864bef'

BASE_URL = 'https://www.vizie.csiro.au/debug2/api'

DATA_URL = BASE_URL + '/data'
USER_URL = BASE_URL + '/user'

payload = {
	'accessToken': ACCESS_TOKEN,
	'type': 'api',
	'api': 'twitter'
}

for task in tasks:
	for modifier in dataset_file_modifiers:
		task += modifier
		logger.info('Reading [{}]...'.format(dataset_filename.format(task=task)))
		with open(dataset_filename.format(task=task)) as tsv:
			c = Counter(row[0] for row in csv.reader(tsv, dialect='excel-tab'))
			num_uniq = len(c)
			num_lines = len(list(c.elements()))
			logger.info('Found {} unique lines and a total of {} lines'.format(num_uniq, num_lines))
		c = defaultdict(int)
		with open(dataset_filename.format(task=task)) as tsv:
			for i, row in enumerate(csv.reader(tsv, dialect='excel-tab')):
				payload['id'] = row[0]
				logger.info('[{}] Processing line {} (tweet: {})...'.format(dataset_filename.format(task=task), i, payload['id']))
				if c[payload['id']]: 
					logger.info('[{}] \tAlready processed.'.format(dataset_filename.format(task=task)))
					continue

				if db_vizie_ref.find_one({'_id': payload['id'], 'vizie_post_id': {'$exists': True}}): 
					logger.info('[{}] \tAlready been retrieved and stored.'.format(dataset_filename.format(task=task)))
					continue

				logger.info('[{}] \tImporting tweet {} to Vizie...'.format(dataset_filename.format(task=task), payload['id']))			
				r = requests.post(DATA_URL + '/posts/import', data=payload)
				logger.info('[{}] \t\tPOST request sent to [{}]'.format(dataset_filename.format(task=task), r.url))
				logger.debug('[{}] \t\twith parameters {}'.format(dataset_filename.format(task=task), json.dumps(payload)))
				
				logger.debug('[{}] \t\t\tstatus code: {} | response: {}'.format(dataset_filename.format(task=task), r.status_code, r.text))
				logger.debug('[{}] \t\t\t{}'.format(dataset_filename.format(task=task), r.headers))

				try:
					json_response = r.json()
				except ValueError, e:
					json_response = {}
					logger.info(e)

				result = {u'_id': payload['id'], u'code': r.status_code}

				if r.status_code == 500:
					if json_response.get('error') == 'post could not be retrieved from the Twitter API':
						logger.info('[{}] \t\t\tTweet is no longer available.'.format(dataset_filename.format(task=task)))
						result['message'] = 'Tweet is no longer available'
					else:
						logger.exception('[{}] \t\t\tUnexpected Error Occurred!'.format(dataset_filename.format(task=task)))
				elif r.status_code == 200:
					logger.info('[{}] \t\t\tmessage: {message} | vizie_post_id: {post id}\t\t\t '.format(dataset_filename.format(task=task), **json_response))
					result = {
						u'_id': payload['id'],
						u'vizie_post_id': json_response['post id'],
						u'message': json_response['message']
					}
				else:
					logger.exception('[{}] \t\t\tUnexpected Error Occurred!'.format(dataset_filename.format(task=task)))

				logger.info('[{}] \t\tCompleted POST request'.format(dataset_filename.format(task=task)))

				logger.info('[{}] \t\tSaving to MongoDB...'.format(dataset_filename.format(task=task)))
				db_vizie_ref.save(result)
				logger.info('[{}] \t\tSaved!'.format(dataset_filename.format(task=task)))

				c[payload['id']] += 1
				logger.info('[{}] Successfully processed line {}/{} ({:.2f}% complete)'.format(dataset_filename.format(task=task), i, num_lines-1, (i)/float(num_lines-1)*100))
				
				logger.info('[{}] Sleeping for {} seconds...'.format(dataset_filename.format(task=task), delay))
				time.sleep(delay)
