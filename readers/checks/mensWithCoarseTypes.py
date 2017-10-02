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

coarsetypes = set(["location", "person", "organization", "event"])

class CoarseTypeMentions(object):
    def __init__(self, config, vocabloader):
        print("Loading mentions ... ")
        self.mens_file = config.ace_mentions_file
        self.mens = utils.make_mentions_from_file(self.mens_file)
        (l2idx, idx2l) = vocabloader.getLabelVocab()

        print(coarsetypes)
        for t in coarsetypes:
            print(l2idx[t])

        print(idx2l[0])

        self.countMentionsWithCoarse()


    def countMentionsWithCoarse(self):
        mentionsWithCoarse = 0
        hascoarse = False
        for m in self.mens:
            types = m.types
            for t in types:
                if t in coarsetypes:
                    hascoarse = True
            if hascoarse:
                mentionsWithCoarse += 1
            hascoarse = False
        percMentionsWithCoarse = (float(mentionsWithCoarse)/len(self.mens))*100
        print("Total Mentions : {}. Perc mentions with Coarse type : {:.4f}%".format(
              len(self.mens), percMentionsWithCoarse))


if __name__=='__main__':
    config = Config("configs/vocab_config.ini")
    vocabloader = VocabLoader(config)
    a = CoarseTypeMentions(config, vocabloader)
