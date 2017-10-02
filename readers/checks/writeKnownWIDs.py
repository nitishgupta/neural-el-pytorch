import os
import readers.utils as utils
from readers.Mention import Mention
from readers.config import Config
from readers.vocabloader import VocabLoader

config = Config("configs/all_mentions_config.ini")
vocabloader = VocabLoader(config)

(knwid2idx, idx2knwid) = vocabloader.getKnwnWidVocab()
wid2WikiTitle = vocabloader.getWID2Wikititle()

print("Known {} total {}".format(len(knwid2idx), len(wid2WikiTitle)))

widswithtext = set()
with open("/save/ngupta19/wikipedia/wiki_kb/widswithtext", 'r') as f:
    docswithtext =  f.readlines()
    for l in docswithtext:
        widswithtext.add(l.strip())

print("Total docs with text : {}".format(len(widswithtext)))

missing = 0
total = 0
for wid in knwid2idx:
    total += 1
    if not (str(wid) in widswithtext):
        print(wid)
        missing += 1

print("Known Total : {} Missing {} ".format(total, missing))


missing = 0
total = 0
for wid in wid2WikiTitle:
    total += 1
    if not (str(wid) in widswithtext):
        missing += 1

print("Total : {} Missing {} ".format(total, missing))
