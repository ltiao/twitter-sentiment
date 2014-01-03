#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import random
import yaml
import re

from preprocess import TwitterTextPreprocessor


class TestTwitterPreprocessor(unittest.TestCase):
    
    def setUp(self):
        with open('twitter-text-conformance/extract.yml', 'r') as extractor_tests_infile:
            self.extractor_tests = yaml.load(extractor_tests_infile)
        self.preprocessor = TwitterTextPreprocessor() 

    # TODO: Dynamically generate test method instead of making assertions in a loop in one test method,
    # using nosetests or some other method described in:
    # http://stackoverflow.com/questions/32899/how-to-generate-dynamic-parametrized-unit-tests-in-python
    def test_mentions(self):
        for test in self.extractor_tests['tests']['mentions']:
            text = test['text']
            if test['expected']:
                expected_text = re.sub('|'.join(re.escape(exp) for exp in test['expected']), 'MENTION', text)
            else:
                expected_text = text
            print self.preprocessor.normalize_mentions(text)
            print expected_text
            print
            self.assertEqual(self.preprocessor.normalize_mentions(text), expected_text)

class KnownGood(unittest.TestCase):
    def __init__(self, test):
        super(KnownGood, self).__init__()
        self.text = test['text']
        if test['expected']:
            self.expected_text = re.sub('|'.join(re.escape(exp) for exp in test['expected']), 'MENTION', self.text)
        else:
            self.expected_text = self.text
        self.preprocessor = TwitterTextPreprocessor() 
        
    def runTest(self):
        self.assertEqual(self.preprocessor(self.text), self.expected_text)
        
def suite():
    suite = unittest.TestSuite()
    with open('twitter-text-conformance/extract.yml', 'r') as extractor_tests_infile:
        extractor_tests = yaml.load(extractor_tests_infile)
    suite.addTests(KnownGood(test) for test in extractor_tests['tests']['mentions'])
    return suite

if __name__ == '__main__':
    #unittest.main()
    unittest.TextTestRunner().run(suite())
