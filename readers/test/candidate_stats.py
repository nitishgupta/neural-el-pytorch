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

class CandidateStats(object):
    def __init__(self, config, vocabloader, test_mentions_file, num_cands=30):
        self.unk_wid = "<unk_wid>"

        # Known WID Vocab
        (self.knwid2idx, self.idx2knwid) = vocabloader.getKnwnWidVocab()
        self.num_kwn_entities = len(self.idx2knwid)
        print(" [#] Loaded. Num of known wids : {}".format(self.num_kwn_entities))

        # Crosswikis
        print("[#] Loading training/val crosswikis dictionary ... ")
        self.test_knwen_cwikis = vocabloader.getTestKnwEnCwiki()
        self.test_allen_cwikis = vocabloader.getTestAllEnCwiki()

        print("[#] Test Mentions File : {}".format(test_mentions_file))

        print("[#] Pre-loading test mentions ... ")
        print("Test Mentions : {}".format(test_mentions_file))
        self.mentions = utils.make_mentions_from_file(test_mentions_file)
        self.num_mens = len(self.mentions)
        print( "[#] Test Mentions : {}".format(self.num_mens))

        self.num_cands = num_cands

    #*******************      END __init__      *********************************

    def allCandidateStats(self):
        print("Computing Training Stats")
        mens = 0
        #widNotInCands = set()
        widNotInCands = []

        mens = len(self.mentions)
        ((v_recallAt1, v_recallAt30, v_noCands, v_numCands),
         (v_recallAt1_inkb, v_recallAt30_inkb, v_noCands_inkb, v_numCands_inkb),
         inKbMens) = self.get_cand_stats(self.mentions)
        v_recallAt1 = float(v_recallAt1)/mens
        v_recallAt30 = float(v_recallAt30)/mens
        v_numCands = float(v_numCands)/mens

        v_recallAt1_inkb = float(v_recallAt1_inkb)/inKbMens
        v_recallAt30_inkb = float(v_recallAt30_inkb)/inKbMens
        v_numCands_inkb = float(v_numCands_inkb)/inKbMens



        print("Total Mentions  : {}".format(len(self.mentions)))
        print("For All Mentions : {}, Recall @ 1 : {} "
              "Recall @ 30 : {}, No Cands : {} Avg Num of Cands : {}".format(
                mens, v_recallAt1, v_recallAt30, v_noCands, v_numCands))
        print("Mentions with Entity in Known KB : {}".format(inKbMens))
        print("For Mentions in Known KB : {}, Recall @ 1 : {} "
              "Recall @ 30 : {}, No Cands : {} Avg Num of Cands : {}".format(
                inKbMens, v_recallAt1_inkb, v_recallAt30_inkb, v_noCands_inkb, v_numCands_inkb))
    #enddef

    def get_cand_stats(self, mentions):
        recallAt1 = 0
        recallAt30 = 0
        noCands = 0
        numCands = 0
        recallAt1_inkb = 0
        recallAt30_inkb = 0
        noCands_inkb = 0
        numCands_inkb = 0

        notInKB = 0
        for m in mentions:
            nocand = True
            (candidates, cprobs) = self.make_candidates_cprobs(m, useKnownEntitesOnly=False)
            (candidates_inkb, cprobs_inkb) = self.make_candidates_cprobs(m, useKnownEntitesOnly=True)
            (r1, r30, numC, noC) = utils.getCandStatsForMention(cprobs)
            (r1inkb, r30inkb, numCinkb, noCinkb) = utils.getCandStatsForMention(cprobs_inkb)
            recallAt1 += r1
            recallAt30 += r30
            numCands += numC
            noCands += noC

            recallAt1_inkb += r1inkb
            recallAt30_inkb += r30inkb
            numCands_inkb += numCinkb
            noCands_inkb += noCinkb

            if m.wid not in self.knwid2idx:
                notInKB += 1
        inKbMens = len(mentions) - notInKB
        return ((recallAt1, recallAt30, noCands, numCands),
                (recallAt1_inkb, recallAt30_inkb, noCands_inkb, numCands_inkb),
                inKbMens)


    def make_candidates_cprobs(self, m, useKnownEntitesOnly):
        # First wid_idx is true entity
        candidates = [m.wid]
        cprobs = [0.0]
        # Crosswikis to use based on Known / All entities
        if useKnownEntitesOnly:
            cwiki_dict = self.test_knwen_cwikis
        else:
            cwiki_dict = self.test_allen_cwikis

        # Fill num_cands now
        surface = utils._getLnrm(m.surface)
        if surface in cwiki_dict:
            candwids_cprobs = cwiki_dict[surface][0:self.num_cands]
            for (c,p) in candwids_cprobs:
                if c == m.wid:  # Update cprob for true if in known set
                    cprobs[0] = p
                else:
                    candidates.append(c)
                    cprobs.append(p)

        return (candidates, cprobs)


if __name__ == '__main__':
    sttime = time.time()
    configpath = "configs/allnew_mentions_config.ini"
    config = Config(configpath, verbose=False)
    vocabloader = VocabLoader(config)
    b = CandidateStats(
      config=config,
      vocabloader=vocabloader,
      test_mentions_file=config.wikidata_inkb_test_file,
      num_cands=30)

    stime = time.time()

    b.allCandidateStats()
    sys.exit()
