#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import random
import yaml
import re

from preprocess import TwitterTextPreprocessor

class TestTwitterMentionsNormalizer(unittest.TestCase):
    def __init__(self, test):
        super(TestTwitterMentionsNormalizer, self).__init__()
        self.test = test
        self.preprocessor = TwitterTextPreprocessor()

    def runTest(self):
        text = self.test['text']
        if self.test['expected']:
            expected_text = re.sub('|'.join(re.escape(exp) for exp in self.test['expected']), 'MENTION', text)
        else:
            expected_text = text
        self.assertEqual(self.preprocessor.normalize_mentions(text), expected_text)

if __name__ == '__main__':
    mentions_test_suite = unittest.TestSuite()
    with open('twitter-text-conformance/extract.yml', 'r') as extractor_tests_infile:
        mentions_test_suite.addTests(TestTwitterMentionsNormalizer(test) for test in yaml.load(extractor_tests_infile).get('tests', {}).get('mentions'))
    unittest.TextTestRunner().run(mentions_test_suite)
