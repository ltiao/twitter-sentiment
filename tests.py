#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import unittest

from tokenize import TwitterTokenizer

class TestSequenceFunctions(unittest.TestCase):

    def setUp(self):
        self.seq = range(10)

    def test_shuffle(self):
        # make sure the shuffled sequence does not lose any elements
        random.shuffle(self.seq)
        self.seq.sort()
        self.assertEqual(self.seq, range(10))

        # should raise an exception for an immutable sequence
        self.assertRaises(TypeError, random.shuffle, (1,2,3))

    def test_choice(self):
        element = random.choice(self.seq)
        self.assertTrue(element in self.seq)

    def test_sample(self):
        with self.assertRaises(ValueError):
            random.sample(self.seq, 20)
        for element in random.sample(self.seq, 5):
            self.assertTrue(element in self.seq)

class TestTwitterTokenizer(unittest.TestCase):
    
    def setUp(self):
        self.twokenizer = TwitterTokenizer()

    # TODO: Add everything from http://en.wikipedia.org/wiki/List_of_emoticons (including Eastern emoticons)
    def test_emoticons(self):
        strings = (
            u':-) :) :o) :] :3 :c) :> =] 8) =) :} :^) :っ)',
            u':-D :D 8-D 8D x-D xD X-D XD =-D =D =-3 =3', # omitted: B^D
            u'>:[ :-( :( :-c :c :-< :っC :< :-[ :[ :{',
        )
        for string in strings:
            tokens = self.twokenizer.tokenize(string)
            self.assertEqual(string, ' '.join(tokens))

    def test_hashtag(self):
        string = u'@soem83thing @test'
        tokens = self.twokenizer.tokenize(string)
        self.assertEqual(string, ' '.join(tokens))


if __name__ == '__main__':
    unittest.main()