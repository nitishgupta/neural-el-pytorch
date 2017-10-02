import re
import os
import gc
import sys
import math
import time
import pickle
import random
import unicodedata
import collections
import numpy as np
import readers.utils as utils
from readers.Mention import Mention
from readers.config import Config
from readers.vocabloader import VocabLoader

class TrainColdValEntityIntersection(object):
    def __init__(self, config, vocabloader):

        self.new_knw_wid_vocab = "/save/ngupta19/wikipedia/wiki_mentions/wcoh/vocab/new/new_knwn_wid_vocab.pkl"

        (self.knwid2idx, self.idx2knwid) = utils.load(self.new_knw_wid_vocab)

        newfile = "/save/ngupta19/wikipedia/wiki_mentions/wcoh/newmentions.txt"
        self.new_mentions = utils.make_mentions_from_file(newfile)

        self.coldWIDS = set()

    def updateKnwWidVocab(self):
        print("Old : {} Old : {}".format(len(self.knwid2idx), len(self.idx2knwid)))
        for m in self.new_mentions:
            if m.wid not in self.knwid2idx:
                self.idx2knwid.append(m.wid)
                self.knwid2idx[m.wid] = len(self.idx2knwid) - 1

        print("new : {} new : {}".format(len(self.knwid2idx), len(self.idx2knwid)))
        utils.save(self.new_knw_wid_vocab, (self.knwid2idx, self.idx2knwid))

    def findColdEntitiesInTest(self, test_mentions_file):
        test_mentions = utils.make_mentions_from_file(test_mentions_file)
        coldMentions = 0
        for m in test_mentions:
            if m.wid not in self.knwid2idx:
                self.coldWIDS.add(m.wid)
                coldMentions += 1

        print("Total Mentions : {}".format(len(test_mentions)))
        print("Cold Mentions : {}".format(coldMentions))
        print("Cold WID Set Size : {}".format(len(self.coldWIDS)))



    def writeNewKnownMentions(self):
        outf = open(self.outfile, 'w')

        print("Priocessing : {}".format(self.cold_val1))
        with open(self.cold_val1, 'r') as f:
            lines = f.readlines()
            for line in lines:
                m = line.strip()
                wid = m.split("\t")[1]
                if wid in self.coldWIDS:
                    outf.write(m)
                    outf.write("\n")

        print("Priocessing : {}".format(self.cold_val1))
        with open(self.cold_val2, 'r') as f:
            lines = f.readlines()
            for line in lines:
                m = line.strip()
                wid = m.split("\t")[1]
                if wid in self.coldWIDS:
                    outf.write(m)
                    outf.write("\n")

        outf.close()




if __name__=='__main__':
    config = Config("configs/all_mentions_config.ini")
    vocabloader = VocabLoader(config)
    a = TrainColdValEntityIntersection(config, vocabloader)

    a.findColdEntitiesInTest(config.aida_inkb_dev_file)
    a.findColdEntitiesInTest(config.aida_inkb_test_file)
    a.findColdEntitiesInTest(config.ace_mentions_file)
    a.findColdEntitiesInTest(config.wikidata_inkb_test_file)
    a.findColdEntitiesInTest(config.msnbc_inkb_test_file)

    print(utils.get_mention_files(config.train_mentions_dir))

    #a.updateKnwWidVocab()
