import re
import os
import gc
import sys
import math
import pickle
import random
import unicodedata
import collections
import numpy as np
import readers.utils as utils
import time

start_word = "<s>"
end_word = "<eos>"

class Mention(object):
    def __init__(self, mention_line, trreader):
        ''' mention_line : Is the string line stored for each mention
        mid wid wikititle start_token end_token surface tokenized_sentence all_types
        '''
        mention_line = mention_line.strip()
        split = mention_line.split("\t")
        (self.mid, self.wid, self.wikititle) = split[0:3]
        self.start_token = int(split[3])
        self.end_token = int(split[4])
        self.surface = split[5]
        self.sent_tokens = [start_word]
        self.sent_tokens.extend(split[6].split(" "))
        self.sent_tokens.append(end_word)
        self.types = split[7].split(" ")

        assert self.end_token <= (len(self.sent_tokens) - 1), "Line : %s" % mention_line
    #enddef
#endclass

class TestKnownEntityCount(object):
    def __init__(self, test_mentions_file, word_vocab_pkl,
                 label_vocab_pkl, knwn_wid_vocab_pkl):
        self.start_word = start_word
        self.end_word = end_word
        self.unk_word = 'unk' # In tune with word2vec
        self.unk_wid = "<unk_wid>"
        self.tr_sup = 'tr_sup'
        self.tr_unsup = 'tr_unsup'

        if not (os.path.exists(word_vocab_pkl) and os.path.exists(label_vocab_pkl)
                and os.path.exists(knwn_wid_vocab_pkl)) :
            print("Atleast one vocab not found. Run vocabs.py before running model.")
            sys.exit()

        # Word VOCAB
        print("[#] Loading word vocab ... ")
        (self.word2idx, self.idx2word) = utils.load(word_vocab_pkl)
        self.num_words = len(self.idx2word)
        print(" [#] Word vocab loaded. Size of vocab : {}".format(self.num_words))

        # Label Vocab
        print("[#] Loading label vocab ... ")
        (self.label2idx, self.idx2label) = utils.load(label_vocab_pkl)
        self.num_labels = len(self.idx2label)
        print(" [#] Label vocab loaded. Number of labels : {}".format(self.num_labels))

        # Known WID Vocab
        print("[#] Loading Known Entities Vocab : ")
        (self.knwid2idx, self.idx2knwid) = utils.load(knwn_wid_vocab_pkl)
        self.num_knwn_entities = len(self.idx2knwid)
        print(" [#] Loaded. Num of known wids : {}".format(self.num_knwn_entities))

        # Crosswikis
        #print("[#] Loading training/val crosswikis dictionary ... ")
        #self.crosswikis_dict = utils.load_crosswikis(trval_crosswikis_pkl)

        print("[#] Test Mentions File : {}".format(test_mentions_file))

        print("[#] Loading test mentions ... ")
        self.test_mentions = self._make_mentions_from_file(test_mentions_file)
        self.num_test_mentions = len(self.test_mentions)
        print( "[#] Test Mentions : {}".format(self.num_test_mentions))

        print("\n[#] LOADING COMPLETE")

    #*******************      END __init__      *********************************

    def get_mention_files(self, mentions_dir):
        mention_files = []
        for (dirpath, dirnames, filenames) in os.walk(mentions_dir):
            mention_files.extend(filenames)
            break
        #endfor
        random.shuffle(mention_files)
        return mention_files
    #enddef

    def _make_mentions_from_file(self, mens_file):
        with open(mens_file, 'r') as f:
            mention_lines = f.read().strip().split("\n")
        mentions = []
        for line in mention_lines:
            mentions.append(Mention(line, self))
        return mentions
    #enddef

    def count_known_entities(self):
        ''' Data : wikititle \t mid \t wid \t start \t end \t tokens \t labels
        '''
        num_knwn_entity_mentions = 0
        knwn_entities = set()
        for m in self.test_mentions:
            if m.wid in self.knwid2idx:
                num_knwn_entity_mentions += 1
                knwn_entities.add(m.wid)
            else:
                print("{}, {}".format(m.wid, m.sent_tokens))

        print("Total Mentions : {}, Known Entity Mentions : {}".format(
              self.num_test_mentions, num_knwn_entity_mentions))
    #enddef

    def convert_word2idx(self, word):
        if word in self.word2idx:
            return self.word2idx[word]
        else:
            return self.word2idx[self.unk_word]
    #enddef

if __name__ == '__main__':
    sttime = time.time()
    b = TestKnownEntityCount(
      test_mentions_file="/save/ngupta19/datasets/ACE/mentions.txt",
      word_vocab_pkl="/save/ngupta19/wikipedia/wiki_mentions/vocab/word_vocab.pkl",
      label_vocab_pkl="/save/ngupta19/wikipedia/wiki_mentions/vocab/label_vocab.pkl",
      knwn_wid_vocab_pkl="/save/ngupta19/wikipedia/wiki_mentions/vocab/knwn_wid_vocab.pkl")
    stime = time.time()

    b.count_known_entities()
