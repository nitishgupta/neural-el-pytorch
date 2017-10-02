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

class TestCrossWikisVocabs(object):
    def __init__(self, config, vocabloader, test_mentions_file):
        ''' Updates a test crosswikis which is the original crosswikis pruned but
        only with surfaces from test data

        There are 2 dictionaries that mare maintained :
        test_kwn_cwiki : Only has candidates that are in KnownEntity set
        test_all_cwiki : All entities from KB can be candidates (i.e. full cwikis)
        '''

        if not os.path.exists(config.test_kwnen_cwikis_pkl):
            print("Test Known Entity CWiki does not exist ... ")
            self.test_kwn_cwiki = {}
        else:
            self.test_kwn_cwiki = utils.load(config.test_kwnen_cwikis_pkl)
        print("Size of test known en cwiki : {}".format(len(self.test_kwn_cwiki)))

        if not os.path.exists(config.test_allen_cwikis_pkl):
            print("Test Data All Entity CWiki does not exist ... ")
            self.test_all_cwiki = {}
        else:
            self.test_all_cwiki = utils.load(config.test_allen_cwikis_pkl)
        print("Size of test all en cwiki : {}".format(len(self.test_all_cwiki)))

        # Known WID Vocab
        print("[#] Loading Known Entities Vocab : ")
        (self.knwid2idx, self.idx2knwid) = vocabloader.getKnwnWidVocab()
        self.num_knwn_entities = len(self.idx2knwid)
        print(" [#] Loaded. Num of known wids : {}".format(self.num_knwn_entities))

        self.crosswikis_dict = utils.load_crosswikis(config.crosswikis_pkl)
        '''
        self.test_mentions = utils.make_mentions_from_file(test_mentions_file)

        self.incrementCrossWikis(self.test_mentions)

        print("After increments ... ")
        print(" Size of test known en cwiki : {}".format(len(self.test_kwn_cwiki)))
        print(" Size of test all en cwiki : {}".format(len(self.test_all_cwiki)))

        utils.save(config.test_kwnen_cwikis_pkl, self.test_kwn_cwiki)
        utils.save(config.test_allen_cwikis_pkl, self.test_all_cwiki)
        '''

    #end-vocab-init

    def incrementCrossWikis(self, mentions):
        for m in mentions:
            wid = m.wid
            known = False
            if wid in self.knwid2idx:
                known = True
            surface = utils._getLnrm(m.surface)

            if surface in self.crosswikis_dict:
                c_cprobs = self.crosswikis_dict[surface]  # [(c,p)]
                # For All CWIKI : Add the c_cprobs as it is
                if surface not in self.test_all_cwiki:
                    self.test_all_cwiki[surface] = c_cprobs
                # For Kwn CWIKI : Prune c_cprobs, cond: c is in kwn entities
                if surface not in self.test_kwn_cwiki:
                    kwncands_cprobs = []
                    for (c,p) in c_cprobs:
                        if c in self.knwid2idx:
                            kwncands_cprobs.append((c,p))
                    if len(kwncands_cprobs) > 0:
                        self.test_kwn_cwiki[surface] = kwncands_cprobs
    #enddef

if __name__ == '__main__':
    configpath = "configs/allnew_mentions_config.ini"
    config = Config(configpath)
    vocabloader = VocabLoader(config)
    b = TestCrossWikisVocabs(
            config=config,
            vocabloader=vocabloader,
            test_mentions_file=config.aida_inkb_train_file)

    test_file = config.figer_test_file
    print(test_file)
    test_mentions = utils.make_mentions_from_file(test_file)
    b.incrementCrossWikis(test_mentions)
    print("After increments ... ")
    print(" Size of test known en cwiki : {}".format(len(b.test_kwn_cwiki)))
    print(" Size of test all en cwiki : {}".format(len(b.test_all_cwiki)))

    test_file = config.ontonotes_test_file
    print(test_file)
    test_mentions = utils.make_mentions_from_file(test_file)
    b.incrementCrossWikis(test_mentions)
    print("After increments ... ")
    print(" Size of test known en cwiki : {}".format(len(b.test_kwn_cwiki)))
    print(" Size of test all en cwiki : {}".format(len(b.test_all_cwiki)))

    '''
    print(config.aida_inkb_dev_file)
    test_mentions = utils.make_mentions_from_file(config.aida_inkb_dev_file)
    b.incrementCrossWikis(test_mentions)
    print("After increments ... ")
    print(" Size of test known en cwiki : {}".format(len(b.test_kwn_cwiki)))
    print(" Size of test all en cwiki : {}".format(len(b.test_all_cwiki)))

    print(config.aida_inkb_test_file)
    test_mentions = utils.make_mentions_from_file(config.aida_inkb_test_file)
    b.incrementCrossWikis(test_mentions)
    print("After increments ... ")
    print(" Size of test known en cwiki : {}".format(len(b.test_kwn_cwiki)))
    print(" Size of test all en cwiki : {}".format(len(b.test_all_cwiki)))

    print(config.ace_mentions_file)
    test_mentions = utils.make_mentions_from_file(config.ace_mentions_file)
    b.incrementCrossWikis(test_mentions)
    print("After increments ... ")
    print(" Size of test known en cwiki : {}".format(len(b.test_kwn_cwiki)))
    print(" Size of test all en cwiki : {}".format(len(b.test_all_cwiki)))

    print(config.msnbc_inkb_test_file)
    test_mentions = utils.make_mentions_from_file(config.msnbc_inkb_test_file)
    b.incrementCrossWikis(test_mentions)
    print("After increments ... ")
    print(" Size of test known en cwiki : {}".format(len(b.test_kwn_cwiki)))
    print(" Size of test all en cwiki : {}".format(len(b.test_all_cwiki)))

    print(config.wikidata_inkb_test_file)
    test_mentions = utils.make_mentions_from_file(config.wikidata_inkb_test_file)
    b.incrementCrossWikis(test_mentions)
    print("After increments ... ")
    print(" Size of test known en cwiki : {}".format(len(b.test_kwn_cwiki)))
    print(" Size of test all en cwiki : {}".format(len(b.test_all_cwiki)))
    '''

    utils.save(config.test_kwnen_cwikis_pkl, b.test_kwn_cwiki)
    utils.save(config.test_allen_cwikis_pkl, b.test_all_cwiki)
    print("DONE")
