import re
import os
import gc
import sys
import math
import pickle
import random
import gensim
import unicodedata
import collections
import numpy as np
import time
import readers.utils as utils
from readers.config import Config
from readers.Mention import Mention
from readers.vocabloader import VocabLoader

start_word = "<s>"
end_word = "<eos>"

# person: 25, event: 10, organization: 5, location: 1

class ColdDataReader(object):
    def __init__(self, config, vocabloader, train_mentions_dir,
                 val_file, test_file,
                 num_cands, batch_size,
                 strict_context=True, pretrain_wordembed=True, coherence=True,
                 wordDropoutKeep=1.0, cohDropoutKeep=1.0, window=-1,
                 glove=True):

        '''
        Reader especially for training data, but can be used for test data as
        validation and test file inputs. The requirement is that the mention candidates
        should be added to the TrValCandidateDict using readers.train.crosswikis_vocab

        DataType = 1 corresponds to val_file
        DataType = 2 corresponds to test_file
        '''
        self.config = config
        self.coldDir = config.coldDir
        self.start_word = start_word
        self.end_word = end_word
        self.unk_word = 'unk' # In tune with word2vec
        self.unk_wid = "<unk_wid>"
        self.pretrain_wordembed = pretrain_wordembed
        self.coherence = coherence
        assert 0.0 < wordDropoutKeep <= 1.0
        self.wordDropoutKeep = wordDropoutKeep
        assert 0.0 < cohDropoutKeep <= 1.0
        self.cohDropoutKeep = cohDropoutKeep
        self.glove = glove
        self.num_cands = num_cands
        self.strict_context = strict_context
        self.batch_size = batch_size

        # Word Vocab
        if not self.glove:
            (self.word2idx, self.idx2word) = vocabloader.getWordVocab()
        else:
            (self.word2idx, self.idx2word) = vocabloader.getGloveWordVocab()
        self.num_words = len(self.idx2word)
        print(" [#] Word vocab loaded. Size of vocab : {}".format(self.num_words))

        # Label Vocab
        (self.label2idx, self.idx2label) = vocabloader.getLabelVocab()
        self.num_labels = len(self.idx2label)
        print(" [#] Label vocab loaded. Number of labels : {}".format(self.num_labels))

        # Known WID Vocab
        (self.knwid2idx, self.idx2knwid) = vocabloader.getKnwnWidVocab()
        self.num_knwn_entities = len(self.idx2knwid)
        print(" [#] Loaded. Num of known wids : {}".format(self.num_knwn_entities))

        # Cold WID Vocab
        print("Loading Cold Wid Vocab")
        (self.coldwid2idx, self.idx2coldwid) = utils.load(self.config.coldwid_vocab_pkl)
        self.num_cold_entities = len(self.coldwid2idx)
        print(" [#] Loaded. Num of cold entities : {}".format(self.num_cold_entities))

        # Wid2Wikititle Map
        self.wid2WikiTitle = vocabloader.getWID2Wikititle()
        print(" [#] Total number of entities : {}".format(len(self.wid2WikiTitle)))

        # Wid2TypeLabels Map
        self.wid2TypeLabels = vocabloader.getWID2TypeLabels()
        print(" [#] Total number of Wids : {}".format(len(self.wid2TypeLabels)))

        # Coherence String Vocab
        print("Loading Coherence Strings Dicts ... ")
        (self.cohG92idx, self.idx2cohG9) = utils.load(config.cohstringG9_vocab_pkl)
        self.num_cohstr = len(self.idx2cohG9)
        print(" [#] Number of Coherence Strings in Vocab : {}".format(self.num_cohstr))



        # Known WID Description Vectors
        #self.kwnwid2descvecs = vocabloader.loadKnownWIDDescVecs()
        #print(" [#] Size of kwn wid desc vecs dict : {}".format(len(self.kwnwid2descvecs)))

        # Train / Validation Candidates
        self.trval_cands_dict = vocabloader.getTrainValCandidateDict()
        print(" [#] Size of dict : {}".format(len(self.trval_cands_dict)))

        # self.tr_mens_dir = train_mentions_dir
        # self.tr_mens_files = utils.get_mention_files(self.tr_mens_dir)
        # self.num_tr_mens_files = len(self.tr_mens_files)
        # print("[#] Training Mention Files : {} files".format(self.num_tr_mens_files))

        print("[#] Validation Mentions File : {}".format(val_file))
        print("[#] Test Mentions File : {}".format(test_file))

        self.tr_mentions = []
        self.tr_men_idx = 0
        self.num_tr_mens = 0
        self.tr_fnum = 0
        self.tr_epochs = 0

        print("[#] Pre-loading validation mentions ... ")
        self.val_mentions = utils.make_mentions_from_file(val_file, verbose=True)
        self.test_mentions = utils.make_mentions_from_file(test_file, verbose=True)
        self.val_men_idx = 0
        self.test_men_idx = 0
        self.num_val_mens = len(self.val_mentions)
        self.num_test_mens = len(self.test_mentions)
        self.val_epochs = 0
        self.test_epochs = 0
        print( "[#] Validation Mentions : {}, Test Mentions : {}".format(
              self.num_val_mens, self.num_test_mens))


        if self.pretrain_wordembed:
            stime = time.time()
            if not self.glove:
                print("[#] Loading pretrained word2vec embeddings .. ")
                self.word2vec = gensim.models.Word2Vec.load_word2vec_format(
                  config.word2vec_bin_gz, binary=True)
                self.word2vec.init_sims(replace=True)
                print(" [#] Word2Vec Vectors loaded!")
            else:
                self.word2vec = vocabloader.loadGloveVectors()
                print(" [#] Glove Vectors loaded!")
            ttime = (time.time() - stime)/float(60)
            print("  [#] Time to load vectors : {} mins".format(ttime))


        print("[#] Batch Size: %d" % self.batch_size)

        print("\n[#] LOADING COMPLETE:")
        print("[1] Train mention file \n[2] Validation mentions \n"
              "[3] Word Vocab \n[4] Label Set")

    #*******************      END __init__      *********************************

    def get_vector(self, word):
        if not self.glove:
            if word in self.word2vec.vocab:
                return self.word2vec[word]
            else:
                return self.word2vec['unk']
        if self.glove:
            if word in self.word2vec:
                return self.word2vec[word]
            else:
                return self.word2vec['unk']
    #enddef

    def _load_mentions_from_file(self, mens_dir, mens_files, num_mens_files, fnum, epochs):
        if fnum == num_mens_files: # One round already done, increase epochs
            fnum = 0
            epochs += 1
            #random.shuffle(mens_files)
        file = os.path.join(mens_dir, mens_files[fnum])
        fnum += 1
        # self.tr_mens is a list of objects of Mention class
        mens = utils.make_mentions_from_file(file)
        return (mens, mens_files, fnum, epochs)

    def load_mentions_from_file(self, data_idx=0):
        # data_idx : train = 0, validation = 1
        if data_idx==0 or data_idx=="tr":
            stime = time.time()
            (self.tr_mens, self.tr_mens_files,
             self.tr_fnum, self.tr_epochs) = self._load_mentions_from_file(
              self.tr_mens_dir, self.tr_mens_files, self.num_tr_mens_files,
              self.tr_fnum, self.tr_epochs)
            self.num_tr_mens = len(self.tr_mens)
            self.tr_men_idx = 0
            ttime = (time.time() - stime)/60.0
            print("File Number loaded : {}".format(self.tr_fnum))
            print("Loaded tr mentions. Num of mentions : {}. Time : {:.2f} mins".format(
              self.num_tr_mens, ttime))
        else:
            print("Wrong Datatype. Exiting.")
            sys.exit(0)
    #enddef

    def reset_validation(self):
        self.val_men_idx = 0
        self.test_men_idx = 0
        self.val_epochs = 0
        self.test_epochs = 0

    def _read_mention(self, data_type=0):
        # Read train mention
        if data_type == 0 or data_type=="tr":
            # If all mentions read or no ments in memory
            if self.tr_men_idx == self.num_tr_mens or self.num_tr_mens == 0:
                self.load_mentions_from_file(data_type)

            mention = self.tr_mens[self.tr_men_idx]
            self.tr_men_idx += 1
            return mention
        # Read val mention
        if data_type == 1 or data_type == "val":
            mention = self.val_mentions[self.val_men_idx]
            self.val_men_idx += 1
            if self.val_men_idx == self.num_val_mens:
                self.val_epochs += 1
                self.val_men_idx = 0
            return mention
        # Read test mention
        if data_type == 2 or data_type == "test":
            mention = self.test_mentions[self.test_men_idx]
            self.test_men_idx += 1
            if self.test_men_idx == self.num_test_mens:
                self.test_epochs += 1
                self.test_men_idx = 0
            return mention
        print("Wrong data_type arg. Quitting ... ")
        sys.exit(0)
    #enddef

    def _next_batch(self, data_type):
        ''' Data : wikititle \t mid \t wid \t start \t end \t tokens \t labels
        start and end are inclusive
        '''
        # Sentence     = s1 ... m1 ... mN, ... sN.
        # Left Batch   = s1 ... m1 ... mN
        # Right Batch  = sN ... mN ... m1
        (left_batch, right_batch) = ([], [])

        # For each mention, list of its tokens. Average token embeddings for repr.
        mention_embed_batch = []

        # Labels : Vector of 0s and 1s of size = number of labels = 113
        labels_batch = np.zeros([self.batch_size, self.num_labels])

        # Indices: In [B, CohStrs] matrix, Values: 1.0 for len(indices)
        coh_indices = []
        coh_values = []
        if self.coherence:
            coh_matshape = [self.batch_size, self.num_cohstr]
        else:
            coh_matshape = []

        # Wiki Description: [B, N=100, D=300]
        #truewid_descvec_batch = []

        # Candidate WID idxs and their cprobs
        # First element is always true wid
        (wid_idxs_batch, wid_cprobs_batch) = ([], [])

        # As these are cold entities, the first element in wid_idxs_batch will
        # correspond to unk_wid. We need to explicity pass the correct wid for
        # description and types.
        wids_batch = []

        while len(left_batch) < self.batch_size:
            batch_el = len(left_batch)
            m = self._read_mention(data_type=data_type)

            wids_batch.append(m.wid)

            start = m.start_token
            end = m.end_token

            # Labels
            for label in m.types:
                if label in self.label2idx:
                    labelidx = self.label2idx[label]
                    labels_batch[batch_el][labelidx] = 1.0
            #labels

            # Coherence Batch
            cohFound = False    # If no coherence mention is found, then add unk
            if self.coherence:
                cohidxs = []  # Indexes in the [B, NumCoh] matrix
                cohvals = []  # 1.0 to indicate presence
                for cohstr in m.coherence:
                    r = random.random()
                    if cohstr in self.cohG92idx and r < self.cohDropoutKeep:
                        cohidx = self.cohG92idx[cohstr]
                        cohidxs.append([batch_el, cohidx])
                        cohvals.append(1.0)
                        cohFound = True
                if cohFound:
                    coh_indices.extend(cohidxs)
                    coh_values.extend(cohvals)
                else:
                    cohidx = self.cohG92idx[self.unk_word]
                    coh_indices.append([batch_el, cohidx])
                    coh_values.append(1.0)

            # Left and Right context
            if self.strict_context:    # Strict Context
                left_tokens = m.sent_tokens[0:m.start_token]
                right_tokens = m.sent_tokens[m.end_token+1:][::-1]
            else:    # Context inclusive of mention surface
                left_tokens = m.sent_tokens[0:m.end_token+1]
                right_tokens = m.sent_tokens[m.start_token:][::-1]

            # Word Dropout
            left_tokens = self.wordDropout(left_tokens, self.wordDropoutKeep)
            right_tokens = self.wordDropout(right_tokens, self.wordDropoutKeep)

            if not self.pretrain_wordembed:
                left_idxs = [self.convert_word2idx(word) for word in left_tokens]
                right_idxs = [self.convert_word2idx(word) for word in right_tokens]
            else:
                left_idxs = left_tokens
                right_idxs = right_tokens

            left_batch.append(left_idxs)
            right_batch.append(right_idxs)


            ''' # Mention Tokens Batch
            mention_tokens = m.surface.split(" ")
            mention_tokens = self.wordDropout(mention_tokens, self.wordDropoutKeep)
            mention_embed_batch.append(mention_tokens)
            '''

            # Wiki Description Batch
            # Each self.kwnwid2descvecs[m.wid] - List of N=100 vectors (size=300)

            #if m.wid in self.knwid2idx:
            #  truewid_descvec_batch.append(self.kwnwid2descvecs[m.wid])
            #else:
            #  truewid_descvec_batch.append(self.kwnwid2descvecs[self.unk_wid])


            # Candidate WID_Idxs and Prior Probabilities
            cands_dict_key = (utils._getLnrm(m.surface), m.wid)
            try:
                (wid_idxs, wid_cprobs) = self.trval_cands_dict[cands_dict_key]
            except:
                wid_idxs = [self.knwid2idx[self.unk_wid]]*self.num_cands
                wid_cprobs = [0.0]*self.num_cands


            wid_idxs_batch.append(wid_idxs)
            wid_cprobs_batch.append(wid_cprobs)

            #self.print_test_batch(m, wid_idxs, wid_cprobs)
        #end batch making

        coherence_batch = (coh_indices, coh_values,
                           coh_matshape)

        return (left_batch, right_batch, wids_batch, labels_batch,
                coherence_batch, wid_idxs_batch, wid_cprobs_batch)
    #enddef

    def wordDropout(self, list_tokens, dropoutkeeprate):
        if dropoutkeeprate < 1.0:
            for i in range(0,len(list_tokens)):
                r = random.random()
                if r > dropoutkeeprate:
                    list_tokens[i] = self.unk_word
        return list_tokens

    def print_test_batch(self, mention, wid_idxs, wid_cprobs):
        print("Surface : {}  WID : {}".format(mention.surface, mention.wid))
        print(mention.sent_tokens)
        # print("WIDS : ")
        # for (idx,cprob) in zip(wid_idxs, wid_cprobs):
        #   print("WID : {}  CPROB : {}".format(self.idx2knwid[idx], cprob))
        # print()

    def _random_knwn_ents(self, knwn_ent_idx, num):
        ''' Given an entity, sample a number of random neg entities from known entity set
        knwn_ent_idx : idx of known entity for which negs are to be sampled
        num : number of negative samples needed
        '''
        neg_ents = []
        while len(neg_ents) < num:
            neg = random.randint(0, self.num_knwn_entities-1)
            if neg != knwn_ent_idx:
                neg_ents.append(neg)
        return neg_ents

    def embed_batch(self, batch):
        ''' Input is a padded batch of left or right contexts containing words
        Dimensions should be [B, padded_length]
        Output:
          Embed the word idxs using pretrain word embedding
        '''
        output_batch = []
        for sent in batch:
            word_embeddings = [self.get_vector(word) for word in sent]
            output_batch.append(word_embeddings)
        return output_batch

    def embed_mentions_batch(self, mentions_batch):
        ''' Input is batch of mention tokens as a list of list of tokens.
        Output: For each mention, average word embeddings '''
        embedded_mentions_batch = []
        for m_tokens in mentions_batch:
            outvec = np.zeros(300, dtype=float)
            for word in m_tokens:
                outvec += self.get_vector(word)
            outvec = outvec / len(m_tokens)
            embedded_mentions_batch.append(outvec)
        return embedded_mentions_batch

    def pad_batch(self, batch):
        if not self.pretrain_wordembed:
            pad_unit = self.word2idx[self.unk_word]
        else:
            pad_unit = self.unk_word
        lengths = [len(i) for i in batch]
        max_length = max(lengths)
        for i in range(0, len(batch)):
            batch[i].extend([pad_unit]*(max_length - lengths[i]))
        return (batch, lengths)

    def _next_padded_batch(self, data_type):
        (left_batch, right_batch, wids_batch,
         labels_batch, coherence_batch,
         wid_idxs_batch, wid_cprobs_batch) = self._next_batch(data_type=data_type)
        (left_batch, left_lengths) = self.pad_batch(left_batch)
        (right_batch, right_lengths) = self.pad_batch(right_batch)
        if self.pretrain_wordembed:
            left_batch = self.embed_batch(left_batch)
            right_batch = self.embed_batch(right_batch)
            #mention_batch = self.embed_mentions_batch(mention_batch)

        return (left_batch, left_lengths, right_batch, right_lengths, wids_batch,
                labels_batch, coherence_batch, wid_idxs_batch, wid_cprobs_batch)
    #enddef

    def convert_word2idx(self, word):
        if word in self.word2idx:
            return self.word2idx[word]
        else:
            return self.word2idx[self.unk_word]
    #enddef

    def next_train_batch(self):
        return self._next_padded_batch(data_type=0)

    def next_val_batch(self):
        return self._next_padded_batch(data_type=1)

    def next_test_batch(self):
        return self._next_padded_batch(data_type=2)

if __name__ == '__main__':
    sttime = time.time()
    batch_size = 1
    num_batch = 1000
    #configpath = "configs/allnew_mentions_config.ini"
    configpath = "configs/cold_mentions_config.ini"
    config = Config(configpath)
    vocabloader = VocabLoader(config)
    b = TrainingDataReader(config=config,
                           vocabloader=vocabloader,
                           train_mentions_dir=config.train_mentions_dir,
                           val_file=config.cold_mentions_file,
                           test_file=config.aida_inkb_train_file,
                           num_cands=30,
                           batch_size=batch_size,
                           strict_context=False,
                           pretrain_wordembed=False,
                           coherence=False,
                           wordDropoutKeep=0.6,
                           glove=False)

    stime = time.time()

    i = 0
    total_instances = 0
    while b.tr_epochs < 1 and b.val_epochs < 1 and b.test_epochs < 1:
    #for i in range(0, num_batch):
        (left_batch, left_lengths,
         right_batch, right_lengths, truewid_descvec_batch,
         labels_batch, coherence_batch,
         wid_idxs_batch, wid_cprobs_batch) = b._next_padded_batch(data_type=1)
        total_instances += len(left_batch)
        print(left_batch)
        print(wid_idxs_batch)
        print(wid_cprobs_batch)

        # left_widxs = left_batch[0]
        # rght_widxs = right_batch[0]
        # wid_idxs = wid_idxs_batch[0]
        # left_words = [b.idx2word[idx] for idx in left_widxs]
        # rght_words = [b.idx2word[idx] for idx in rght_widxs]
        # wids = [b.idx2knwid[idx] for idx in wid_idxs]
        # titles = [b.widWikititle[wid] for wid in wids]

        # print(left_words)
        # print(rght_words)
        # print(titles)
        # print("\n")


        if i%100 == 0:
            etime = time.time()
            t=etime-stime
            print("{} done. Time taken : {} seconds".format(i, t))

        i += 1
    #endfor
    print(len(truewid_descvec_batch))
    print(len(truewid_descvec_batch[0]))
    print(len(truewid_descvec_batch[0][0]))
    etime = time.time()
    t=etime-stime
    tt = etime - sttime
    print("Total Instances : {}".format(total_instances))
    print("Batching time (in secs) to make %d batches of size %d : %7.4f seconds" % (i, batch_size, t))
    print("Total time (in secs) to make %d batches of size %d : %7.4f seconds" % (i, batch_size, tt))
