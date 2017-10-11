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


class TestCandidateDictionary(object):
    def __init__(self, config, vocabloader):
        ''' Used to make pruned crosswikis dict and candidate dictionary
        for training and validation data

        train_val_cwikis_pkl : Slice of crosswikis for surfaces in train/val

        train_val_cwikis_cands_pkl: Train/Val data only contain known entities
        This dict acts as pre-cache of mention candidates.
        key   : (LNRM(surface), WID)
        Value : ([Candidate_IDXs], [CProbs])
        Candidate_Idxs : The first idx is the true wid_idx, rest are candidates
        Padded with Unk_Wid_Idx(=0) if less than number of candidates needed.
        '''
        self.config = config

        self.numc = 30
        (self.knwid2idx, self.idx2knwid) = vocabloader.getKnwnWidVocab()
        self.wid2WikiTitle = vocabloader.getWID2Wikititle()

        if not os.path.exists(config.test_kwnen_cands_pkl):
            print("Test Candidates Dictionary does not exist. Making new ...")
            self.test_kwnen_cands_dict = {}
        else:
            self.test_kwnen_cands_dict = vocabloader.getTestCandidateDict()
            print("Test Candidates Dictionary exists. Size:{}".format(
                len(self.test_kwnen_cands_dict)))

        print("Loading Crosswikis")
        self.crosswikis_dict = vocabloader.loadPrunedCrosswikis()
        print("Crosswikis Loaded. Size: {}".format(
            len(self.crosswikis_dict)))

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
            # Get list of (c_wid, cprobs)
            (wids, probs) = self.crosswikis_dict[surfacelnrm]
            for (c, p) in zip(wids, probs):
                if c == wid:
                    cprobs[0] = p
                elif c in self.knwid2idx:
                    candidates.append(self.knwid2idx[c])
                    cprobs.append(p)

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

    def make_test_candidates(self, test_file):
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
        cands_dict = {}
        mentions = utils.make_mentions_from_file(mens_file=test_file)
        self._addCandidatesForMentions(mentions, cands_dict)

        return cands_dict

    def updateTestCandsDict(self, test_file):

        print("Updating Test Candidates Dict. Size:{}\n"
              "Key:(surface, wid), V: ([CandWids], [PriorProbs])".format(
                len(self.test_kwnen_cands_dict)))
        print("Test File: {}".format(test_file))
        test_cands_dict = self.make_test_candidates(test_file)

        self.test_kwnen_cands_dict.update(test_cands_dict)

        utils.save(self.config.test_kwnen_cands_pkl,
                   self.test_kwnen_cands_dict)
        print("Train/Val Candidates Dict Saved. Size:{}".format(
            len(self.test_kwnen_cands_dict)))


if __name__ == '__main__':
    config = Config("configs/config.ini")
    vocabloader = VocabLoader(config)
    b = TestCandidateDictionary(config=config, vocabloader=vocabloader)

    b.updateTestCandsDict(test_file=config.aida_kwn_dev_file)
    b.updateTestCandsDict(test_file=config.aida_kwn_test_file)

    sys.exit(0)
