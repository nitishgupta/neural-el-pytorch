import re
import os
import gc
import sys
import time
import math
import pickle
import random
import pprint
import unicodedata
import configparser
import collections
import numpy as np
import readers.utils as utils
from readers.config import Config

class VocabLoader(object):
    def __init__(self, config):
        self.initialize_all_dicts()
        self.config = config

    def initialize_all_dicts(self):
        (self.gword2idx, self.gidx2word) = (None, None)
        (self.label2idx, self.idx2label) = (None, None)
        (self.knwid2idx, self.idx2knwid) = (None, None)
        self.trval_cands_dict = None
        self.wid2Wikititle = None
        self.wid2TypeLabels = None
        (self.test_knwen_cwikis, self.test_allen_cwikis) = (None, None)
        self.glove2vec = None
        self.knownwid2descvecs = None
        self.crosswikis_pruned = None

    def getWordVocab(self):
        if self.word2idx == None or self.idx2word == None:
            if not os.path.exists(self.config.word_vocab_pkl):
                print("Word Vocab PKL missing")
                sys.exit()
            (self.word2idx, self.idx2word) = utils.load(self.config.word_vocab_pkl)
        return (self.word2idx, self.idx2word)

    def getLabelVocab(self):
        if self.label2idx == None or self.idx2label == None:
            if not os.path.exists(self.config.label_vocab_pkl):
                print("Label Vocab PKL missing")
                sys.exit()
            (self.label2idx, self.idx2label) = utils.load(self.config.label_vocab_pkl)
        return (self.label2idx, self.idx2label)

    def getKnwnWidVocab(self):
        if self.knwid2idx == None or self.idx2knwid == None:
            if not os.path.exists(self.config.kwnwid_vocab_pkl):
                print("Known Entities Vocab PKL missing")
                sys.exit()
            (self.knwid2idx, self.idx2knwid) = utils.load(self.config.kwnwid_vocab_pkl)
        return (self.knwid2idx, self.idx2knwid)

    def getTrainValCandidateDict(self):
        if self.trval_cands_dict == None:
            if not os.path.exists(self.config.trval_kwnidx_cands_pkl):
                print("Train Validation Candidate Dict missing")
                sys.exit()
            self.trval_cands_dict = utils.load(self.config.trval_kwnidx_cands_pkl)
        return self.trval_cands_dict

    def getTestKnwEnCwiki(self):
        if self.test_knwen_cwikis == None:
            if not os.path.exists(self.config.test_kwnen_cwikis_pkl):
                print("Test Known Entity CWikis Dict missing")
                sys.exit()
            self.test_knwen_cwikis = utils.load(self.config.test_kwnen_cwikis_pkl)
        return self.test_knwen_cwikis

    def getTestAllEnCwiki(self):
        if self.test_allen_cwikis == None:
            if not os.path.exists(self.config.test_allen_cwikis_pkl):
                print("Test All Entity CWikis Dict missing")
                sys.exit()
            self.test_allen_cwikis = utils.load(self.config.test_allen_cwikis_pkl)
        return self.test_allen_cwikis

    def getWID2Wikititle(self):
        if self.wid2Wikititle == None:
            if not os.path.exists(self.config.widWiktitle_pkl):
                print("wid2Wikititle pkl missing")
                sys.exit()
            self.wid2Wikititle = utils.load(self.config.widWiktitle_pkl)
        return self.wid2Wikititle

    def getWID2TypeLabels(self):
        if self.wid2TypeLabels == None:
            if not os.path.exists(self.config.wid2typelabels_vocab_pkl):
                print("wid2TypeLabels pkl missing")
                sys.exit()
            self.wid2TypeLabels = utils.load(self.config.wid2typelabels_vocab_pkl)
        return self.wid2TypeLabels

    def loadGloveVectors(self):
        if self.glove2vec == None:
            if not os.path.exists(self.config.glove_pkl):
                print("Glove_Vectors_PKL doesnot exist")
                sys.exit()
            self.glove2vec = utils.load(self.config.glove_pkl)
        return self.glove2vec

    def getGloveWordVocab(self):
        if self.gword2idx == None or self.gidx2word == None:
            if not os.path.exists(self.config.glove_word_vocab_pkl):
                print("Glove Word Vocab PKL missing")
                sys.exit()
            (self.gword2idx, self.gidx2word) = utils.load(self.config.glove_word_vocab_pkl)
        return (self.gword2idx, self.gidx2word)

    def loadKnownWIDDescVecs(self):
        if self.knownwid2descvecs == None:
            if not os.path.exists(self.config.knownwid2descvectors):
                print("Known WIDS Description Vectors PKL missing")
                sys.exit()
            self.knownwid2descvecs = utils.load(self.config.knownwid2descvectors)
        return self.knownwid2descvecs

    def loadPrunedCrosswikis(self):
        if self.crosswikis_pruned == None:
            if not os.path.exists(self.config.crosswikis_pruned_pkl):
                print("Crosswikis Pruned Does not exist.")
                sys.exit()
            self.crosswikis_pruned = utils.load(
                self.config.crosswikis_pruned_pkl)
        return self.crosswikis_pruned

if __name__=='__main__':
    config = Config("configs/wcoh_config.ini")
    a = VocabLoader(config)
    a.loadWord2Vec()
