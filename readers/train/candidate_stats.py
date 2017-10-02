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
from readers.config import Config
from readers.Mention import Mention
from readers.vocabloader import VocabLoader


start_word = "<s>"
end_word = "<eos>"

class CandidateStats(object):
    def __init__(self, config, vocabloader, num_cands):
        self.unk_wid = "<unk_wid>"

        self.g_num_wids = 0
        self.g_num_elements = 0

        # Known WID Vocab
        # Known WID Vocab
        (self.knwid2idx, self.idx2knwid) = vocabloader.getKnwnWidVocab()
        self.num_kwn_entities = len(self.idx2knwid)
        print(" [#] Loaded. Num of known wids : {}".format(self.num_kwn_entities))

        # Candidates Dict
        print("[#] Loading training/val crosswikis candidate dict ... ")
        self.trval_cands_dict = vocabloader.getTrainValCandidateDict()

        print("[#] Training Mentions Dir : {}".format(config.train_mentions_dir))
        self.tr_mens_dir = config.train_mentions_dir
        self.tr_mens_files = utils.get_mention_files(self.tr_mens_dir)
        self.num_tr_mens_files = len(self.tr_mens_files)
        print(" [#] Training Mention Files : {} files".format(self.num_tr_mens_files))

        print("[#] Validation Mentions File : {}".format(config.val_mentions_file))
        print("[#] Test Mentions File : {}".format(config.test_mentions_file))

        print("[#] Pre-loading validation mentions ... ")
        self.val_mentions = utils.make_mentions_from_file(config.val_mentions_file)
        self.test_mentions = utils.make_mentions_from_file(config.test_mentions_file)
        self.num_val_mens = len(self.val_mentions)
        self.num_test_mens = len(self.test_mentions)
        print( "[#] Validation Mentions : {}, Test Mentions : {}".format(
              self.num_val_mens, self.num_test_mens))

        self.num_cands = num_cands

        print("\n[#] LOADING COMPLETE:")
    #*******************      END __init__      *********************************

    def allCandidateStats(self):
        print("Computing Training Stats")
        tr_mens = 0
        tr_recallAt1 = 0
        tr_recallAt30 = 0
        tr_no_cands = 0
        tr_cands = 0
        #widNotInCands = set()
        widNotInCands = []

        for tr_file in self.tr_mens_files:
            print("Reading file : {}".format(tr_file))
            mens = self._load_mentions_from_file(self.tr_mens_dir, tr_file)
            tr_mens += len(mens)
            (rAt1, rAt30, noCands, num_cands) = self.get_cand_stats(mens, widNotInCands)
            tr_recallAt1 += rAt1
            tr_recallAt30 += rAt30
            tr_no_cands += noCands
            tr_cands += num_cands

        tr_recallAt1 = float(tr_recallAt1)/tr_mens
        tr_recallAt30 = float(tr_recallAt30)/tr_mens
        tr_avg_cands = float(tr_cands)/tr_mens

        print("Total Training Mentions : {}, Recall @ 1 : {} "
              "Recall @ 30 : {} No Candidates : {} Avg. Num of Cands : {}".format(
                tr_mens, tr_recallAt1, tr_recallAt30, tr_no_cands, tr_avg_cands))
        print("Correct WID not in Crosswikis candidates : {}".format(
          len(widNotInCands)))

        v_mens = len(self.val_mentions)
        #v_widNotInCands = set()
        v_widNotInCands = []
        (v_recallAt1, v_recallAt30, v_noCands, num_cands) = self.get_cand_stats(
          self.val_mentions, v_widNotInCands)
        v_recallAt1 = float(v_recallAt1)/v_mens
        v_recallAt30 = float(v_recallAt30)/v_mens
        v_numCands = float(num_cands)/v_mens

        print("Total Validation Mentions : {}, Recall @ 1 : {} "
              "Recall @ 30 : {}, No Cands : {} Avg Num of Cands : {}".format(
                v_mens, v_recallAt1, v_recallAt30, v_noCands, v_numCands))
        print("Correct WID not in Crosswikis candidates : : {}".format(len(v_widNotInCands)))
    #enddef

    def get_cand_stats(self, mentions, widNotInCands):
        recallAt1 = 0
        recallAt30 = 0
        no_cands = 0
        num_cands = 0
        found = False
        for m in mentions:
            found = False
            candidates = self.get_fuzzy_candidates(m)
            num_cands += len(candidates)
            if len(candidates) > 0:
                if candidates[0] == m.wid:
                    recallAt1 += 1
                cand_wids = set(candidates)
                if m.wid in cand_wids:
                    recallAt30 += 1
                    found = True
                if found == False:
                    widNotInCands.append((m.surface, m.wid))
                    #widNotInCands.add((m.surface, m.wid))
            else:
                no_cands += 1
        return (recallAt1, recallAt30, no_cands, num_cands)

    def get_candidates(self, mention):
        candidates = []
        # Fill num_cands now
        surface = utils._getLnrm(mention.surface)
        if surface in self.crosswikis_dict:
            cands = self.crosswikis_dict[surface][0:self.num_cands]
            for c in cands:
                candidates.append(c[0])

        return candidates

    def get_fuzzy_candidates(self, mention):
        candidates = []
        surface_tokens = mention.surface.split(" ")
        surfacelnrm = utils._getLnrm(mention.surface)
        if surfacelnrm in self.crosswikis_dict:
            cands = self.crosswikis_dict[surfacelnrm][0:self.num_cands]
            for c in cands:
                candidates.append(c[0])
        token_num = 0
        extra_cands = set()
        while len(extra_cands) < (self.num_cands-len(candidates)) and token_num < len(surface_tokens):
            surfacetoken = utils._getLnrm(surface_tokens[token_num])
            if surfacetoken in self.crosswikis_dict:
                cands = self.crosswikis_dict[surfacetoken][0:(self.num_cands-len(candidates))]
                for c in cands:
                    extra_cands.add(c[0])
            token_num += 1
        candidates.extend(list(extra_cands))
        return candidates

    def knwnCandsDictStats(self):
        total_keys = len(self.trval_cands_dict)
        recallAt30 = 0
        widNotInCands = 0
        numCands = 0
        for key in self.trval_cands_dict:
            surface = key[0]
            wid = key[1]
            [candidates, cprobs] = self.trval_cands_dict[key]
            if cprobs[0] > 0.0:
                recallAt30 += 1
            else:
                widNotInCands += 1
            for p in cprobs:
                if p > 0.0:
                    numCands += 1
        avgNumCands = float(numCands)/total_keys
        print("Total Keys : {}  Recall@30 : {}  WidNotinCand : {} AvgNumCands : {}".format(
          total_keys, recallAt30, widNotInCands, avgNumCands))
    #enddef

    def _knwnEntity_candidate_stats(self, mentions):
        ''' Get candidate stats for mentions from pre-cached candidate dict
          But method is generic, can/should be used whenever candiidates as read as
          candidates are desired in training i.e.
          Candidates: [true_candidx, cand1_idx, cand2_idx, ..., unk_wid_idx]
          cprobs:     [cprobs]
        '''
        num_mentions = len(mentions)
        recallAt30 = 0
        recallAt1 = 0
        numCands = 0
        noCandsForMention = 0
        for m in mentions:
            nocand = True
            true_wid_idx = self.knwid2idx[m.wid]
            surface = utils._getLnrm(m.surface)
            key = (surface, m.wid)
            (candidates, cprobs) = self.trval_cands_dict[key]
            (r1, r30, numC, noC) = utils.getCandStatsForMention(cprobs)
            recallAt1+= r1
            recallAt30+= r30
            numCands += numC
            noCandsForMention += noC
        return (num_mentions, recallAt1, recallAt30, noCandsForMention, numCands)

    def validationKnwnCandsStats(self):
        print("Computing Validation Known Candidate Stats")
        (nummens, rAt1, rAt30, noC, numC) = self._knwnEntity_candidate_stats(self.val_mentions)
        recallAt1 = rAt1/float(nummens)
        recallAt30 = rAt30/float(nummens)
        numCands = numC/float(nummens)
        noCands = noC
        candsButNotCorr = nummens - rAt30 - noC

        print("Validation Known Candidates Stats : ")
        print("Number of mentions: {}".format(nummens))
        print("Recall @ 1 : {}".format(recallAt1))
        print("Recall @ 30 : {}".format(recallAt30))
        print("No Cands : {}".format(noCands))
        print("Correct WID not in Cands : {}".format(candsButNotCorr))
        print("Num of Cands : {}".format(numCands))

    def testKnwnCandsStats(self):
        print("Computing Test Known Candidate Stats")
        (nummens, rAt1, rAt30, noC, numC) = self._knwnEntity_candidate_stats(self.test_mentions)
        recallAt1 = rAt1/float(nummens)
        recallAt30 = rAt30/float(nummens)
        numCands = numC/float(nummens)
        noCands = noC
        candsButNotCorr = nummens - rAt30 - noC

        print("Test Known Candidates Stats : ")
        print("Number of mentions: {}".format(nummens))
        print("Recall @ 1 : {}".format(recallAt1))
        print("Recall @ 30 : {}".format(recallAt30))
        print("No Cands : {}".format(noCands))
        print("Correct WID not in Cands : {}".format(candsButNotCorr))
        print("Num of Cands : {}".format(numCands))

    def trainKnwnCandsStats(self):
        print("Computing Training Known Candidate Stats")
        recallAt1 = 0
        recallAt30 = 0
        numCands = 0
        noCands = 0
        numMens = 0

        for tr_file in self.tr_mens_files:
            print("Reading file : {}".format(tr_file))
            mens = self._load_mentions_from_file(self.tr_mens_dir, tr_file)
            (numm, rAt1, rAt30, noC, numC) = self._knwnEntity_candidate_stats(mens)
            numMens += numm
            numCands += numC
            recallAt30 += rAt30
            recallAt1 += rAt1
            noCands += noC
        #
        candsButNotCorr = numMens - recallAt30 - noCands
        recallAt1 = recallAt1/float(numMens)
        recallAt30 = recallAt30/float(numMens)
        noCands = noCands/float(numMens)
        numCands = numCands/float(numMens)


        print("Training Known Candidates Stats : ")
        print("Number of mentions: {}".format(numMens))
        print("Recall @ 1 : {}".format(recallAt1))
        print("Recall @ 30 : {}".format(recallAt30))
        print("No Cands : {}".format(noCands))
        print("Correct WID not in Cands : {}".format(candsButNotCorr))
        print("Num of Cands : {}".format(numCands))
    #enddef


if __name__ == '__main__':
    sttime = time.time()
    batch_size = 1000
    num_batch = 1000
    configpath = "configs/wcoh_config.ini"
    config = Config(configpath)
    vocabloader = VocabLoader(config)
    b = CandidateStats(config=config, vocabloader=vocabloader, num_cands=30)

    stime = time.time()

    #b.allCandidateStats()
    #b.knwnCandsDictStats()
    b.validationKnwnCandsStats()
    b.testKnwnCandsStats()
    #b.trainKnwnCandsStats()
    sys.exit()
