import re
import os
import gc
import sys
import math
import time
import pickle
import random
import pprint
import unicodedata
import collections
import numpy as np
from stop_words import get_stop_words
import readers.utils as utils
from readers.Mention import Mention
from readers.config import Config
from readers.vocabloader import VocabLoader

pp = pprint.PrettyPrinter()

stop_words = get_stop_words('en')

class TestWordVocabOverlap(object):
    def __init__(self, config, vocabloader):
        self.test_mentions = utils.make_mentions_from_file(config.ace_mentions_file)

        (self.word2idx, self.idx2word) = vocabloader.getWordVocab()

        self.numTestDataTokens = 0
        self.testTokensVocabOverlap = 0

        self.testDataWordSet = set()

        self.getTestWordVocabOverlap()

    def getTestWordVocabOverlap(self):
        for m in self.test_mentions:
            words = m.sent_tokens
            self.testDataWordSet.update(words)
            self.numTestDataTokens += len(words)
            for word in words:
                if word in self.word2idx:
                    self.testTokensVocabOverlap += 1
        print("Total Tokens in Test Data : {}".format(self.numTestDataTokens))
        print("Test Tokens w Vocab Overlap : {}".format(self.testTokensVocabOverlap))
        percOverlap = 100*float(self.testTokensVocabOverlap)/float(self.numTestDataTokens)
        print("Percentage Overlap : {}".format(percOverlap))

        inter = self.testDataWordSet.intersection(set(self.word2idx.keys()))
        percVocabInter = 100*float(len(inter))/float(len(self.testDataWordSet))
        print("Test Data Vocab Size : {}".format(len(self.testDataWordSet)))
        print("Test Data Vocab w Vocab Overlap : {}".format(percVocabInter))


if __name__=='__main__':
    config = Config("configs/vocab_config.ini")
    vocabloader = VocabLoader(config)
    a = TestWordVocabOverlap(config, vocabloader)
