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

from readers.Mention import Mention

def save(fname, obj):
    with open(fname, 'wb') as f:
        pickle.dump(obj, f, protocol=4)

def load(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)

def _getLnrm(arg):
    """Normalizes the given arg by stripping it of diacritics, lowercasing, and
    removing all non-alphanumeric characters.
    """
    arg = ''.join(
        [c for c in unicodedata.normalize('NFD', arg) if unicodedata.category(c) != 'Mn'])
    arg = arg.lower()
    arg = ''.join(
        [c for c in arg if c in set('abcdefghijklmnopqrstuvwxyz0123456789')])
    return arg

def load_crosswikis(crosswikis_pkl):
    stime = time.time()
    print("[#] Loading normalized crosswikis dictionary ... ")
    crosswikis_dict = load(crosswikis_pkl)
    ttime = (time.time() - stime)/60.0
    print(" [#] Crosswikis dictionary loaded!. Time: {0:2.4f} mins. Size : {1}".format(
    ttime, len(crosswikis_dict)))
    return crosswikis_dict
#end

def load_widSet(
    widWikititle_file="/save/ngupta19/freebase/types_xiao/wid.WikiTitle"):
    print("Loading WIDs in the KB ... ")
    wids = set()
    with open(widWikititle_file, 'r') as f:
        text = f.read()
        lines = text.strip().split("\n")
        for line in lines:
            wids.add(line.split("\t")[0].strip())

    print("Loaded all WIDs : {}".format(len(wids)))
    return wids


def measureCrossWikisEntityConverage(crosswikis_pkl):
    wids = load_widSet()
    coverageWids = set()
    crosswikis_dict = load_crosswikis(crosswikis_pkl)

    for mention in crosswikis_dict:
        for c in crosswikis_dict[mention]:
            coverageWids.add(c[0])

    entities_CommonWithKB = len(coverageWids.intersection(wids))
    entities_InKBNotInCW = len(wids.difference(coverageWids))
    entities_NotInKB = len(coverageWids.difference(wids))
    print("Total Entity Coverage of Crosswikis : {}  Entities in CW, Not in KB {}".format(
    entities_CommonWithKB, entities_NotInKB))
    print("Entities in KB but not in CW : {}".format(entities_InKBNotInCW))

def make_mentions_from_file(mens_file, verbose=False):
    stime = time.time()
    with open(mens_file, 'r') as f:
        mention_lines = f.read().strip().split("\n")
        mentions = []
        for line in mention_lines:
            mentions.append(Mention(line))
    ttime = (time.time() - stime)
    if verbose:
        print(" ## Time in loading {} mens : {} secs".format(mens_file, ttime))
    return mentions


def get_mention_files(mentions_dir):
    mention_files = []
    for (dirpath, dirnames, filenames) in os.walk(mentions_dir):
        mention_files.extend(filenames)
        break
    return mention_files


def getCandStatsForMention(cprobs):
    ''' cprobs for any mention's candidates enough to compute candidate stats
    cprobs : cprobs[0] - cprob for true entity (0.0 -> entity not in candidates)
    cprob[i] == 0.0 means unk_wid (padding) (i > 0)
    '''
    recallAt30 = 0
    recallAt1 = 0
    numCands = 0
    noCandsForMention = 0
    noCand = True

    true_cp = cprobs[0]
    maxp = true_cp
    if true_cp > 0.0:
        recallAt30 += 1
        numCands += 1
        noCand = False
    for i in range(1, len(cprobs)):
        cp = cprobs[i]
        if cp > maxp:
            maxp = cp
        if cp > 0.0:
            numCands += 1
            noCand = False
    #
    if noCand:
        noCandsForMention += 1
    if maxp == true_cp and maxp != 0.0:
        recallAt1 += 1
    return (recallAt1, recallAt30, numCands, noCandsForMention)


def decrSortedDict(dict):
    # Returns a list of tuples (key, value) in decreasing order of the values
    return sorted(dict.items(), key=lambda kv: kv[1], reverse=True)

if __name__=="__main__":
    measureCrossWikisEntityConverage("/save/ngupta19/crosswikis/crosswikis.normalized.pkl")
