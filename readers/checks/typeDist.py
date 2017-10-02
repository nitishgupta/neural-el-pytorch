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

class typeCountDistribution(object):
    def __init__(self, config, vocabloader):
        self.tr_mens_dir = config.train_mentions_dir
        self.tr_mens_files = utils.get_mention_files(self.tr_mens_dir)

        (self.label2idx, self.idx2label) = vocabloader.getLabelVocab()

        # Initializing with zero count
        self.typeCount = {}
        for t in self.label2idx.keys():
            self.typeCount[t] = 0

        self.makeLabelCount()

    def addToTypeCount(self, mens):
        for m in mens:
            types = m.types
            for t in types:
                self.typeCount[t] += 1

    def convertTypeCountToFraction(self, decSorted, numMens):
        typeCount = []
        for (t, c) in decSorted:
            typeCount.append((t, float(c)/numMens))
        return typeCount


    def makeLabelCount(self):
        totalMentions = 0
        for (i,mens_file) in enumerate(self.tr_mens_files):
            print("File Num : {}".format(i))
            file = os.path.join(self.tr_mens_dir, mens_file)
            mens = utils.make_mentions_from_file(file)
            self.addToTypeCount(mens)
            totalMentions += len(mens)
        decSorted = utils.decrSortedDict(self.typeCount)
        decSorted = self.convertTypeCountToFraction(decSorted, totalMentions)
        print("Total Mentions : {}".format(totalMentions))
        pp.pprint(decSorted)


if __name__=='__main__':
    config = Config("configs/vocab_config.ini")
    vocabloader = VocabLoader(config)
    a = typeCountDistribution(config, vocabloader)
