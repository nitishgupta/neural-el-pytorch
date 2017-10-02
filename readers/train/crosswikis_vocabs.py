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

start_word = "<s>"
end_word = "<eos>"

class CrossWikisVocabs(object):
    def __init__(self, config, vocabloader):
        ''' Used to make pruned crosswikis dict and candidate dictionary
        for training and validation data

        train_val_cwikis_pkl : Slice of crosswikis for surfaces in train/val (NOT USED)

        train_val_cwikis_cands_pkl: Train/Val data only contain known entities
        This dict acts as pre-cache of mention candidates.
        key   : (LNRM(surface), WID)
        Value : ([Candidate_IDXs], [CProbs])
        Candidate_Idxs : The first idx is the true wid_idx, rest are candidates
        Padded with Unk_Wid_Idx(=0) if less than number of candidates needed.
        '''
        self.config = config
        train_mentions_dir = config.train_mentions_dir
        val_mentions_file = config.val_mentions_file
        test_mentions_file = config.test_mentions_file

        tr_mens_files = utils.get_mention_files(train_mentions_dir)
        self.numc = 30
        (self.knwid2idx, self.idx2knwid) = vocabloader.getKnwnWidVocab()
        self.wid2WikiTitle = vocabloader.getWID2Wikititle()

        if not os.path.exists(config.trval_kwnidx_cands_pkl):
            self.crosswikis_dict = vocabloader.loadPrunedCrosswikis()
            print("Crosswikis Loaded. Size: {}".format(
                len(self.crosswikis_dict)))

            print("Size of known entities: {}".format(len(self.knwid2idx)))
            print("Making Train/Validation/Test CWiki Candidates.\n"
                  "{Key:(surface, wid), V: ([CandWids], [PriorProbs])")
            train_val_candidates_dict = self.make_train_val_candidatesDict(
                    train_mentions_dir, tr_mens_files,
                    val_mentions_file, test_mentions_file)
            utils.save(config.trval_kwnidx_cands_pkl,
                       train_val_candidates_dict)
            print("Train/Val Candidates Dict Saved")
            sys.exit(0)
        else:
            print("Train/Val CWiki Candidates already exists")
            trval_cand_dict = utils.load(train_val_cwikis_cands_pkl)
            print("Loaded dict")
            key = ('barackobama', '534366')
            (candidates, probs) = (trval_cand_dict[key][0],
                                   trval_cand_dict[key][1])
            candidates = [self.idx2knwid[wididx] for wididx in candidates]
            candidates = [self.wid2WikiTitle[wid] for wid in candidates]

            print((key, candidates, probs))

            # obama = utils._getLnrm("United Nations")
            # key = (obama, 534366)
            # print("Candidates for Barack Obama")
            # print(trval_cand_dict[key])
    # end-vocab-init

    def _getCandidatesForSurfaceWid(self, surface, wid):
        # Though all training mentions true wid is in known set,
        # checking here again makes this function versatile to other mentions

        # First candidate is the true entity
        if wid in self.knwid2idx:
            candidates = [self.knwid2idx[wid]]
        else:
            candidates = [self.knwid2idx["<unk_wid>"]]
        cprobs = [0.0]
        # Adding candidates for complete surface
        surfacelnrm = utils._getLnrm(surface)
        if surfacelnrm in self.crosswikis_dict:
            # Get list of (c_wid,cprobs)
            (wids, probs) = self.crosswikis_dict[surfacelnrm]
            for (c, p) in zip(wids, probs):
                if c == wid:
                    cprobs[0] = p
                elif c in self.knwid2idx:
                    candidates.append(self.knwid2idx[c])
                    cprobs.append(p)
        '''
        # If candidates not complete
        if len(candidates) < self.numc:
            surface_tokens = surface.split(" ")
            token_num = 0
            # For each token find and add candidates
            while (len(candidates) < self.numc and
                   token_num < len(surface_tokens)):
                tokenlnrm = utils._getLnrm(surface_tokens[token_num])
                if tokenlnrm in self.crosswikis_dict:
                    # List of (c,cprobs)
                    c_cprobs = self.crosswikis_dict[tokenlnrm]
                    for (c,p) in c_cprobs:
                        if (c != wid and c in self.knwid2idx and
                                self.knwid2idx[c] not in candidates):

                            candidates.append(self.knwid2idx[c])
                            cprobs.append(p)
                #token
                token_num += 1
            #alltokensend
        '''
        candidates = candidates[0:self.numc]
        cprobs = cprobs[0:self.numc]
        assert len(candidates) == len(cprobs)
        num_cands = len(candidates)
        remain = self.numc - num_cands
        candidates.extend([self.knwid2idx["<unk_wid>"]]*remain)
        cprobs.extend([0.0]*remain)

        assert len(candidates) == self.numc
        assert len(cprobs) == self.numc
        assert candidates[0] == self.knwid2idx[wid]
        for i in range(1, len(candidates)):
            assert candidates[i] != self.knwid2idx[wid]
            assert candidates[i] < len(self.idx2knwid)

        return (candidates, cprobs)

    def _addCandidatesForMentions(self, mentions, cwiki_dict):
        for m in mentions:
            assert m.wid in self.knwid2idx, "Wid not in knwid2idx!!!"
            key = (utils._getLnrm(m.surface), m.wid)
            if key not in cwiki_dict:
                (candidates, cprobs) = self._getCandidatesForSurfaceWid(
                    m.surface, m.wid)
                cwiki_dict[key] = (candidates, cprobs)

    def make_train_val_candidatesDict(self, tr_mens_dir, tr_mens_files,
                                      val_mentions_file, test_mentions_file):
        '''Make a dictionary for training and validation mention candidates
                Note: All training and validation mentions are for
                Known entities
                First element in candidates is the true entity. If it is not
                a crosswiki candidate, then corresponding c_prob = 0.0
                If <30 candidates, then rest is padded unkwid_idx and cprob=0.0
                Can be used for candidate statistics - see candidate_stats.py

        Such data_structure allows for faster training.
        Dict:
                Key: (LNRM(m.surface), m.wid)
                Val: ([true_wid_idx, c_idx1, c_idx2, .., unkwididx], [cprobs])
        '''

        print("Adding mentions of training data")
        cwiki_dict = {}
        files_done = 0
        for file in tr_mens_files:
            mens_fpath = os.path.join(tr_mens_dir, file)
            mentions = utils.make_mentions_from_file(mens_file=mens_fpath)
            self._addCandidatesForMentions(mentions, cwiki_dict)
            files_done += 1
            print("Training Files done : {}".format(files_done))
            print("CWiki Dict Size : {}".format(len(cwiki_dict)))

        print(" Adding mentions of validation data")
        mentions = utils.make_mentions_from_file(mens_file=val_mentions_file)
        self._addCandidatesForMentions(mentions, cwiki_dict)

        print(" Adding mentions of test data")
        mentions = utils.make_mentions_from_file(mens_file=test_mentions_file)
        self._addCandidatesForMentions(mentions, cwiki_dict)

        return cwiki_dict

    def _addCandidatesForAdditionalMentions(self, mentions, cwiki_dict):
        for m in mentions:
            key = (utils._getLnrm(m.surface), m.wid)
            if key not in cwiki_dict:
                if m.wid in self.knwid2idx:
                    (candidates,
                     cprobs) = self._getCandidatesForSurfaceWid(m.surface,
                                                                m.wid)
                    cwiki_dict[key] = (candidates, cprobs)
                else:
                    candidates = [self.knwid2idx["<unk_wid>"]]*self.numc
                    cprobs = [0.0]*self.numc
                    cwiki_dict[key] = (candidates, cprobs)

    def updateTrValCandDict(self, trValCandDict_pkl, crosswikis_pkl,
                            knwn_wid_vocab_pkl, *args):
        if not os.path.exists(trValCandDict_pkl):
            print("Train/Val CWiki Candidates Dict doesn't exist")
            sys.exit()

        print("Updating TrValKwnCandDict for : ")

        print("Loading trvalCandsDict ... ")
        candsDict = utils.load(trValCandDict_pkl)
        print("TrValCandDictSize : {}".format(len(candsDict)))
        self.crosswikis_dict = utils.load_crosswikis(crosswikis_pkl)
        print("Loading known wid2idx dict")
        (self.knwid2idx, self.idx2knwid) = utils.load(knwn_wid_vocab_pkl)
        print("Adding candidates for additional mentions")

        datasetsToUpdate = args
        for dataset in datasetsToUpdate:
            test_file = dataset
            print(test_file)
            mentions = utils.make_mentions_from_file(mens_file=test_file)
            self._addCandidatesForAdditionalMentions(mentions, candsDict)
            print("Size now : {}".format(len(candsDict)))

        utils.save(trValCandDict_pkl, candsDict)
        print("TrValCandDictSize : {}".format(len(candsDict)))


if __name__ == '__main__':
    config = Config("configs/config.ini")
    vocabloader = VocabLoader(config)
    b = CrossWikisVocabs(config=config, vocabloader=vocabloader)
