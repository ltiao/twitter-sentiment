import sys
import time
from urllib2 import URLError
from httplib import BadStatusLine

import logging.config
from settings import LOGGING

import datetime
import twitter

# Logging

logging.config.dictConfig(LOGGING)
logger = logging.getLogger('twitter')

def robust(twitter_api_func, retries=10, wait_time=2, rate_limit_reset=None, *args, **kwargs):
    
    def delay(wait_time=wait_time):
        logger.info('\t- Sleeping for {} seconds ({:.2f} minutes) (Until {})...'.format(wait_time, wait_time/float(60), datetime.datetime.now() + datetime.timedelta(seconds=wait_time)))
        time.sleep(wait_time)

    logger.info('\t- calling resource... retries left: {} | waiting time: {}'.format(retries, wait_time))
    if retries < 1 or wait_time > 1024: # This is rather redundant. Only one of these conditions is required...
        return

    try:
        result = twitter_api_func(*args, **kwargs)
    except twitter.TwitterHTTPError, e:
        if e.e.code == 429:
            logger.info('\t- rate limit exceeded.')
            if rate_limit_reset:
                reset_time = datetime.datetime.fromtimestamp(rate_limit_reset)
                sleep_seconds = (reset_time-datetime.datetime.now()).seconds + 5 # give a 5 second buffer in case we're forced to sleep again
                delay(sleep_seconds)
            else:
                delay()
            logger.info('\t- waking up now. Trying again...')
            return robust(twitter_api_func, retries-1, wait_time*1.5, *args, **kwargs)
        elif e.e.code in (500, 502, 503, 504):
            delay()
            logger.info('\t- waking up now. Trying again...')
            return robust(twitter_api_func, retries-1, wait_time*1.5, *args, **kwargs)
        else:
            if e.e.code == 404:
                logger.info('\tNot found')
            elif e.e.code == 403:
                logger.info('\tForbidden')
            else:
                logger.exception('\tSome other unforeseen error occurred.')
            return e.e.code
    except URLError, e:
        delay()
        logger.info('\tWaking up now. Trying again...')
        return robust(twitter_api_func, retries-1, wait_time*2, *args, **kwargs)
    except:
        logger.exception('\tSome other unforeseen error occurred.')
        return
    else:
        return result