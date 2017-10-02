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

class coherenceCount(object):
    def __init__(self, config, vocabloader):
        self.tr_mens_dir = config.train_mentions_dir
        self.tr_mens_files = utils.get_mention_files(self.tr_mens_dir)

        print("Loading Coherence String Dicts")
        (coh2idx, idx2coh) = utils.load(config.cohstring_vocab_pkl)
        (cohG92idx, idx2cohG9) = utils.load(config.cohstringG9_vocab_pkl)

        print("Coherence Stirng set Size : {}, cnt >= 10 size : {}".format(
          len(idx2coh), len(idx2cohG9)))


        self.testDataCountCohLessMens(config.val_mentions_file, cohG92idx)
        self.testDataCountCohLessMens(config.test_mentions_file, cohG92idx)

        self.testDataCountCohLessMens(config.ace_mentions_file, cohG92idx)

        self.testDataCountCohLessMens(config.aida_inkb_dev_file, cohG92idx)
        self.testDataCountCohLessMens(config.aida_inkb_test_file, cohG92idx)
        self.testDataCountCohLessMens(config.aida_inkb_train_file, cohG92idx)
    #end-init


    def testDataCountCohLessMens(self, mens_file, coh2idx):
        print(mens_file)
        mens = utils.make_mentions_from_file(mens_file)
        total_mens = len(mens)
        cohless_mens = self.countCohlessMens(mens, coh2idx)
        print("Total Mens : {}  Coherence Less Mentions : {}".format(
          total_mens, cohless_mens))

    def countCohlessMens(self, mens, coh2idx):
        cohless_cnt = 0
        for m in mens:
            cohless = True
            for c in m.coherence:
                if c in coh2idx:
                    cohless = False
                    break
            if cohless:
                cohless_cnt += 1
        return cohless_cnt

    def countTrainCohlessMens(self, coh2idx):
        totalMentions = 0
        cohless = 0
        for (i,mens_file) in enumerate(self.tr_mens_files):
            print("File Num : {}".format(i))
            file = os.path.join(self.tr_mens_dir, mens_file)
            mens = utils.make_mentions_from_file(file)
            cohless += self.countCohlessMens(mens, coh2idx)
            totalMentions += len(mens)

        return (totalMentions, cohless)


if __name__=='__main__':
    config = Config("configs/wcoh_config.ini")
    vocabloader = VocabLoader(config)
    a = coherenceCount(config, vocabloader)
