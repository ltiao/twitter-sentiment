import unittest
import sys

sys.path.append('/Users/louistiao/Dropbox/Projects/CSIRO/twitter_sentiment_dev')

from feature_extraction import make_combined_extractor

class TestFeatureExtraction(unittest.TestCase):

    def setUp(self):
        self.vec = make_combined_extractor()
        _, self.feature_extractor = self.vec.transformer_list[1]
        self.features_dict = self.feature_extractor.named_steps['features_extract']._features
    
    def test_all_caps(self):
        features = self.features_dict({u'text': 'WHY ARE YOU YELLING AT ME!!??'})
        self.assertIn('all_caps_match_count', features)
        self.assertEquals(features['all_caps_match_count'], 6)

    def test_some_caps(self):
        features = self.features_dict({u'text': 'The word I isn\'t really as emphatic as THIS so shouldn\'t be included'})
        self.assertIn('all_caps_match_count', features)
        self.assertEquals(features['all_caps_match_count'], 1)
        
    def test_no_caps(self):
        features = self.features_dict({u'text': 'There really isn\'t anything exciting about this piece of text'})
        self.assertIn('all_caps_match_count', features)
        self.assertEquals(features['all_caps_match_count'], 0)

if __name__ == '__main__':
    unittest.main()