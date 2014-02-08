#!/usr/bin/env python
# -*- coding: utf-8 -*-

# import os,sys,inspect
# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(currentdir)
# sys.path.insert(0,parentdir) 

# Setup Logging
from settings import setup_logging
from logging import getLogger

setup_logging()
logger = getLogger('dataset')

from analyzer import TweetPreprocessor, TweetTokenizer
from feature_extraction import make_text_extractor, Bunch
from csv import reader

from collections import namedtuple

import numpy as np
import scipy as sp
import csv
import yaml

class SMS(namedtuple('SMS', ['sms_id_str', 'user_id_str', 'label', 'text'])):

    def _asdict(self):
        'Return a new OrderedDict which maps field names to their values'
        return dict(zip(self._fields, self))

def load_sms(subtask='b'):
    
    if subtask == 'a':
        raise NotImplementedError('SemEval-2013 Task 2 Subtask A data not yet supported')
    elif subtask == 'b':
        
        filename = 'data/cache/sms_b.yml'
        
        try:
            with open(filename, 'r') as infile:
                result = yaml.load(infile)
        except IOError:        
            provided_filename = 'data/2download/test/gold/sms-test-gold-B.tsv'
            with open(provided_filename, 'r') as tsv:
                result = [sms._asdict() for sms in map(SMS._make, csv.reader(tsv, dialect='excel-tab'))]
            
            with open(filename, 'w+') as outfile:
                yaml.dump(result, outfile, default_flow_style=False)

        return result
    else:
        raise ValueError("'{}' is not a valid subtask: should be one of ['a', 'b']".format(subtask))
    

def load_semeval(subtask='b', subset='all'):
    try:
        from pymongo import MongoClient
    except ImportError:
        raise ImportError('pymongo must be installed to retrieve data from MongoDB')

    client = MongoClient()
    db = client.twitter_database
    db_labeled_tweets = db.labeled_tweets

    if subtask == 'a':
        raise NotImplementedError('SemEval-2013 Task 2 Subtask A data not yet supported')
    elif subtask == 'b':
        
        q = {
            u'text': {'$exists': True}, 
            u'class.overall': {'$exists': True}
        }
    
        if subset in ('train', 'test'):
            q[u'class.training'] = (subset == 'train')
        elif subset != 'all':
            raise ValueError("'{}' is not a valid subset: should be one of ['train', 'test', 'all']".format(subset))

        return db_labeled_tweets.find(q)
    else:
        raise ValueError("'{}' is not a valid subtask: should be one of ['a', 'b']".format(subtask))

def load_semeval_vectorized(vect=make_text_extractor(), subtask='b', subset='all'):
    
    data = list(load_semeval(subtask, subset))
    
    if subset == 'train':
        return
    
    

def train_test_concat(X_train, X_test, y_train, y_test):
    # TODO: Throw exception or normalise on XOR case
    if sp.sparse.issparse(X_train) and sp.sparse.issparse(X_test):
        vstack = sp.sparse.vstack
    else:
        vstack = np.vstack

    return vstack((X_train, X_test)), np.concatenate((y_train, y_test))

def train_test_cv_generator(X_train, X_test, y_train, y_test):
    test_start_index = X_train.shape[0]
    test_end_index = test_start_index + X_test.shape[0]
    # not a very exciting generator. only has one item
    yield (np.arange(test_start_index), np.arange(test_start_index, test_end_index))

if __name__ == '__main__':
    
    import matplotlib.pyplot as plt
    import numpy as np

    tweets = load_semeval(subtask='b', subset='all')
    
    bincount = np.bincount(tweets.target)
    
    n = bincount.shape[0]
    
    ind = np.arange(n)  # the x locations for the groups
    width = 0.5         # the width of the bars
    
    #plt.xkcd()
    
    fig, ax = plt.subplots()
    
    rects = ax.bar(ind, bincount, width, align='center', facecolor='#9999ff')
    
    ax.set_ylabel('Frequency')
    ax.set_xlabel('Class')
    ax.set_title('Class frequency distribution')
    ax.set_xticks(ind)
    ax.set_xticklabels(tweets.target_names)
    ax.set_axisbelow(True)
    ax.yaxis.grid()
    
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., height+100, '{}'.format(int(height)), ha='center')
    
    plt.savefig('temp.png')