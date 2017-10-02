import re
import os
import gc
import sys
import math
import time
import pickle
import gensim
import random
import unicodedata
import collections
import numpy as np
from readers import utils
from readers.config import Config
from readers.Mention import Mention

start_word = "<s>"
end_word = "<eos>"

class VocabBuilder(object):
    def __init__(self, config, widWikititle_file, widLabel_file,
                 word_threshold=1):

        '''Given training data, makes word vocab, glove word vocab,
           doc_mentions vocab, type lables vocab, known_wid vocab,
           wid2Wikititle
        '''
        self.start_word = start_word
        self.end_word = end_word
        self.unk_word = 'unk' # In tune with word2vec
        self.unk_wid = "<unk_wid>"

        self.tr_mens_dir = config.train_mentions_dir
        self.tr_mens_files = utils.get_mention_files(self.tr_mens_dir)
        self.num_tr_mens_files = len(self.tr_mens_files)
        print("[#] Training Mention Files : {} files".format(
            self.num_tr_mens_files))

        print("[#] Validation Mentions File : {}".format(
            config.val_mentions_file))

        tr_data_vocabs_exist = self.check_train_data_vocabs_exist(
                config.word_vocab_pkl, config.label_vocab_pkl,
                config.kwnwid_vocab_pkl, config.cohstring_vocab_pkl,
                config.cohstringG1_vocab_pkl)

        if not tr_data_vocabs_exist:
            print("[#] Loading pretrained word2vec embeddings .. ")
            self.word2vec = gensim.models.Word2Vec.load_word2vec_format(
                    config.word2vec_bin_gz, binary=True)
            self.word2vec.init_sims(replace=True)

            print("All/Some Training Vocabs do not exist. Making ... ")
            self.make_training_data_vocabs(
                self.tr_mens_dir, self.tr_mens_files,
                config.word_vocab_pkl, config.label_vocab_pkl,
                config.kwnwid_vocab_pkl, config.cohstring_vocab_pkl,
                config.cohstringG1_vocab_pkl, config.cohstringG9_vocab_pkl,
                word_threshold)

        if not os.path.exists(config.widWiktitle_pkl):
            print(" [#] Making wid2Wikititle Map")
            wid2Wikititle = self.make_widWikititleMap(widWikititle_file)
            utils.save(config.widWiktitle_pkl, wid2Wikititle)
            print(" [#] Done. Size : {}".format(len(wid2Wikititle)))

        if not os.path.exists(config.wid2typelabels_vocab_pkl):
            print(" [#] Making wid2Types Map")
            wid2types = self.make_wid2TypesMap(widLabel_file)
            utils.save(config.wid2typelabels_vocab_pkl, wid2types)
            print(" [#] Done. Size : {}".format(len(wid2types)))

        if not os.path.exists(config.glove_word_vocab_pkl):
            print(" [#] Makign GloVe Word Vocabs")
            glove2vec = utils.load(config.glove_pkl)
            print("   [#] Glove embeddings loaded. Size: {}".format(
                len(glove2vec)))
            (glove_word2idx,
             glove_idx2word) = self.make_glovewordvocab(glove2vec)
            utils.save(config.glove_word_vocab_pkl,
                       (glove_word2idx, glove_idx2word))
    # end-vocab-init

    def check_train_data_vocabs_exist(self, word_vocab_pkl, label_vocab_pkl,
                                      knwn_wid_vocab_pkl, coh_vocab_pkl,
                                      cohG1_vocab_pkl):
        if (os.path.exists(word_vocab_pkl)
            and os.path.exists(label_vocab_pkl)
            and os.path.exists(knwn_wid_vocab_pkl)
            and os.path.exists(coh_vocab_pkl)
            and os.path.exists(cohG1_vocab_pkl)):
            return True
        else:
            return False

    def make_widWikititleMap(self, widWikititle_file):
        wid2Wikititle = {self.unk_wid:self.unk_wid}
        with open(widWikititle_file, 'r') as f:
            lines = f.read().strip().split("\n")

        for line in lines:
            s = line.strip().split("\t")
            wid2Wikititle[s[0]] = s[1]

        print("Size of wid2Wikititle : {}".format(len(wid2Wikititle)))
        return wid2Wikititle

    def make_wid2TypesMap(self, widLabel_file):
        wid2Types = {}
        with open(widLabel_file, 'r') as f:
            lines = f.read().strip().split("\n")
        for line in lines:
            ssplit = line.strip().split("\t")
            types = ssplit[1].strip().split(" ")
            wid = ssplit[0].strip()
            wid2Types[wid] = types
        return wid2Types

    def make_glovewordvocab(self, glove2vec):
        word2idx = {}
        idx2word = []
        for word in glove2vec:
            self.add_to_vocab(word2idx, idx2word, word)

        if "unk" in glove2vec:
            print("UNK PRESENT")
        print("Make Word2Idx done. Size : {}".format(len(idx2word)))
        return (word2idx, idx2word)

    def add_to_vocab(self, element2idx, idx2element, element):
        if element not in element2idx:
            idx2element.append(element)
            element2idx[element] = len(idx2element) - 1

    def make_training_data_vocabs(self, tr_mens_dir, tr_mens_files,
                                  word_vocab_pkl, label_vocab_pkl,
                                  knwn_wid_vocab_pkl, coh_vocab_pkl,
                                  cohG1_vocab_pkl, cohG2_vocab_pkl,
                                  threshold):

        print("Building training vocabs : ")
        word_count_dict = {}
        coh_count_dict = {}
        idx2word = [self.unk_word]
        word2idx = {self.unk_word:0}
        idx2label = []
        label2idx = {}
        idx2knwid = [self.unk_wid]
        knwid2idx = {self.unk_wid:0}
        idx2coh = [self.unk_word]
        coh2idx = {self.unk_word:0}
        idx2cohG1 = [self.unk_word]
        cohG12idx = {self.unk_word:0}
        idx2cohG2 = [self.unk_word]
        cohG22idx = {self.unk_word:0}

        files_done = 0
        for file in tr_mens_files:
            mens_fpath = os.path.join(tr_mens_dir, file)
            mentions = utils.make_mentions_from_file(mens_file=mens_fpath)
            for mention in mentions:
                for typel in mention.types:
                    self.add_to_vocab(element2idx=label2idx,
                                      idx2element=idx2label,
                                      element=typel)
                for token in mention.sent_tokens:
                    if token not in word_count_dict:
                        word_count_dict[token] = 0
                    word_count_dict[token] = word_count_dict[token] + 1

                for cohstring in mention.coherence:
                    if cohstring not in coh_count_dict:
                        coh_count_dict[cohstring] = 0
                    coh_count_dict[cohstring] = coh_count_dict[cohstring] + 1

                self.add_to_vocab(element2idx=knwid2idx,
                                  idx2element=idx2knwid,
                                  element=mention.wid)
            files_done += 1
            print("Files done : {}".format(files_done))
        # all-files-processed
        # WORD VOCAB
        # for word, count in word_count_dict.items():
        #       if count > threshold:
        #               self.add_to_vocab(element2idx=word2idx, idx2element=idx2word,
        #                                                 element=word)

        for word in self.word2vec.vocab:
            self.add_to_vocab(element2idx=word2idx,
                              idx2element=idx2word,
                              element=word)
        # Coherence (and greater 1) VOCAB
        for (cstr, cnt) in coh_count_dict.items():
            self.add_to_vocab(element2idx=coh2idx,
                              idx2element=idx2coh,
                              element=cstr)
            if cnt > 1:
                self.add_to_vocab(element2idx=cohG12idx,
                                  idx2element=idx2cohG1,
                                  element=cstr)
            if cnt > 9:
                self.add_to_vocab(element2idx=cohG22idx,
                                  idx2element=idx2cohG2,
                                  element=cstr)

        print(" [#] Total Words : : {}".format(len(word_count_dict)))
        print(" [#] Threhsolded word vocab. Word Vocab Size: {}".format(
            len(idx2word)))
        utils.save(word_vocab_pkl, (word2idx, idx2word))
        print(" [#] Label Vocab Size: {}".format(len(idx2label)))
        utils.save(label_vocab_pkl, (label2idx, idx2label))
        print(" [#] Known Wiki Titles Size: {}".format(len(idx2knwid)))
        utils.save(knwn_wid_vocab_pkl, (knwid2idx, idx2knwid))
        print(" [#] Coherence String Set Size: {}".format(len(idx2coh)))
        utils.save(coh_vocab_pkl, (coh2idx, idx2coh))
        print(" [#] Coherence String (cnt > 1) Size: {}".format(
            len(idx2cohG1)))
        utils.save(cohG1_vocab_pkl, (cohG12idx, idx2cohG1))
        print(" [#] Coherence String (cnt > 2) Size: {}".format(
            len(idx2cohG2)))
        utils.save(cohG2_vocab_pkl, (cohG22idx, idx2cohG2))


if __name__ == '__main__':
    c = Config("configs/wcoh_config.ini")
    b = VocabBuilder(c,
            widWikititle_file="/save/ngupta19/freebase/types_xiao/wid.WikiTitle",
            widLabel_file="/save/ngupta19/freebase/types_xiao/wid.fbtypelabels",
            word_threshold=0)
