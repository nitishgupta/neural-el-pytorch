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
import readers.utils as utils
from readers.Mention import Mention
from readers.config import Config
from readers.vocabloader import VocabLoader

pp = pprint.PrettyPrinter()

class TestTrainEntityCoherenceOverlap(object):
    def __init__(self, config, vocabloader, testfile):
        self.test_mentions = utils.make_mentions_from_file(testfile)
        self.tr_mens_dir = config.train_mentions_dir
        self.tr_mens_files = utils.get_mention_files(self.tr_mens_dir)

        (self.knwid2idx, self.idx2knwid) = vocabloader.getKnwnWidVocab()

        # Wid2Wikititle Map
        self.wid2WikiTitle = vocabloader.getWID2Wikititle()
        print(" [#] Size of Wid2Wikititle: {}".format(len(self.wid2WikiTitle)))

        # {wiktitle: set(coherence_words)}
        self.train_entities = {}

        # {wiktitle: List of set(coherence_words) for each mention
        self.test_entities = {}

        self.encontext_pr = {}

        self.getTestEntitiesCoherenceWords()
        self.getTrainEntitiesCoherenceWords()
        self.computeEntityCoherencePR(self.train_entities, self.test_entities)

    def getContextWordsForMention(self, m):
        words = [w for w in m.sent_tokens if w not in stop_words]
        return words

    def getTestEntitiesCoherenceWords(self):
        for m in self.test_mentions:
            if m.wid in self.knwid2idx:
                if not m.wikititle in self.test_entities:
                    self.test_entities[m.wikititle] = []
                words = m.coherence
                self.test_entities[m.wikititle].append(set(words))
        print("Total Test Mentions Entities : {}".format(len(self.test_entities)))

    def getTrainEntitiesCoherenceWords(self):
        for (i,mens_file) in enumerate(self.tr_mens_files):
            print("File Num : {}".format(i))
            file = os.path.join(self.tr_mens_dir, mens_file)
            mens = utils.make_mentions_from_file(file)
            for m in mens:
                wt = m.wikititle
                if wt in self.test_entities:
                    if not wt in self.train_entities:
                        self.train_entities[wt] = set()
                    words = m.coherence
                    self.train_entities[wt].update(words)
        print("Total Train Entities : {}".format(len(self.train_entities)))

    def computeEntityCoherencePR(self, tr_dict, test_dict):
        avg_p = 0.0
        avg_r = 0.0
        for wt in tr_dict:
            test_words_list = test_dict[wt]
            train_words = tr_dict[wt]
            p = 0.0
            r = 0.0
            f1 = 0.0
            for test_words in test_words_list:
                # test_words is a set of words for one mention
                inter = test_words.intersection(train_words)
                p += float(len(inter))/float(len(test_words))
                r += float(len(inter))/float(len(train_words))
            p = p/float(len(test_words_list))
            r = r/float(len(test_words_list))
            if not (p == 0 and r == 0):
                f1 = 2*p*r/(p+r)
            else:
                f1 = 0.0
            self.encontext_pr[wt] = (p, r, f1)

            avg_p += p
            avg_r += r
        #
        avg_p = avg_p/float(len(tr_dict))
        avg_r = avg_r/float(len(tr_dict))
        #pp.pprint(self.encontext_pr)
        print("Num of intersection entities : {}".format(len(tr_dict)))
        print("P : {}, R : {}".format(avg_p, avg_r))

    def intersections(self):
        trintersect = self.train_entities.intersection(set(self.knwid2idx.keys()))
        trvalintersection = self.train_entities.intersection(self.coldval_entities)

        print("Train en vocab intersection : {}".format(len(trintersect)))
        print("Train - Cold Val En intersection : {}".format(len(trvalintersection)))


if __name__=='__main__':
    configpath = "configs/wcoh_config.ini"
    config = Config(configpath)
    vocabloader = VocabLoader(config)
    a = TestTrainEntityCoherenceOverlap(config, vocabloader,
                                        testfile=config.test_mentions_file)
