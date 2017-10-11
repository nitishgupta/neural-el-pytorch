import os
import sys
import random
import numpy as np
import time
import readers.utils as utils
from readers.config import Config
from readers.Mention import Mention
from readers.vocabloader import VocabLoader

start_word = "<s>"
end_word = "<eos>"

class TrainingDataReader(object):
    def __init__(self, config, vocabloader,
                 val_file,
                 num_cands, batch_size,
                 strict_context=True, pretrain_wordembed=True,
                 wordDropoutKeep=1.0, cohDropoutKeep=1.0):

        '''
        Reader especially for training data, but can be used for test data as
        validation and test file inputs. The requirement is that the mention candidates
        should be added to the TrValCandidateDict using readers.train.crosswikis_vocab

        DataType 0/1 corresponds to train/val_file
        '''
        self.config = config
        self.batch_size = batch_size
        print("[#] Initializing Training Reader Batch Size: {}".format(
            self.batch_size))
        stime = time.time()
        self.start_word = start_word
        self.end_word = end_word
        self.unk_word = 'unk'    # In tune with glove
        self.unk_wid = "<unk_wid>"
        self.pretrain_wordembed = pretrain_wordembed
        assert 0.0 < wordDropoutKeep <= 1.0
        self.wordDropoutKeep = wordDropoutKeep
        assert 0.0 < cohDropoutKeep <= 1.0
        self.cohDropoutKeep = cohDropoutKeep
        self.num_cands = num_cands
        self.strict_context = strict_context

        # Coherence String Vocab
        (self.cohG92idx, self.idx2cohG9) = utils.load(
            config.cohstringG9_vocab_pkl)
        self.num_cohstr = len(self.idx2cohG9)
        print("[#] Coherence Loaded. Num Coherence Strings: {}".format(
            self.num_cohstr))

        self.tr_mens_dir = config.train_mentions_dir
        self.tr_mens_files = utils.get_mention_files(self.tr_mens_dir)
        self.num_tr_mens_files = len(self.tr_mens_files)
        print("[#] Training Mention Files : {} files".format(
            self.num_tr_mens_files))

        etime = time.time()
        ttime = etime - stime
        print("[#] TRAINING READER LOADING COMPLETE. "
              "Time Taken: {} secs\n".format(ttime))

    def index_coherence(self):
        print("[#] Indexing cohrence ")
        for fnum in range(0, len(self.tr_mens_files)):
            print("Processing file : {}".format(fnum))
            filepath = os.path.join(self.tr_mens_dir,
                                    self.tr_mens_files[fnum])
            outfilepath = os.path.join(self.config.newtrain_mentions_dir,
                                       self.tr_mens_files[fnum])
            outf = open(outfilepath, 'w')
            with open(filepath, 'r') as f:
                mention_lines = f.read().strip().split("\n")

            for line in mention_lines:
                split = line.strip().split("\t")
                (mid, wid, wikititle) = split[0:3]
                start_token = split[3]
                end_token = split[4]
                surface = split[5]
                sent = split[6]
                types = split[7]
                if len(split) > 8:    # If no mention surface words in coherence
                    if split[8].strip() == "":
                        coherence = []
                    else:
                        coherence = split[8].split(" ")


                cohFound = False    # If no coherence mention is found, add unk
                cohidxs = []  # Indexes in the [B, NumCoh] matrix
                for cohstr in coherence:
                    if cohstr in self.cohG92idx:
                        cohidx = self.cohG92idx[cohstr]
                        cohidxs.append(cohidx)
                        cohFound = True
                if not cohFound:
                    cohidx = self.cohG92idx[self.unk_word]
                    cohidxs.append(cohidx)

                cohIdxsStr = ""
                for cohidx in cohidxs:
                    cohIdxsStr += str(cohidx) + " "
                cohIdxsStr = cohIdxsStr.strip()

                outline = mid + "\t" + wid + "\t" + wikititle + "\t" + \
                          start_token + "\t" + end_token + "\t" + \
                          surface + "\t" + sent  + "\t" + types + "\t" + \
                          cohIdxsStr
                outline = outline.strip()
                outf.write(outline)
                outf.write("\n")

            outf.close()


if __name__ == '__main__':
    sttime = time.time()
    batch_size = 1000
    num_batch = 10
    configpath = "configs/config.ini"
    config = Config(configpath)
    vocabloader = VocabLoader(config)
    b = TrainingDataReader(config=config,
                           vocabloader=vocabloader,
                           val_file=config.aida_kwn_dev_file,
                           num_cands=30,
                           batch_size=batch_size,
                           strict_context=False,
                           pretrain_wordembed=False,
                           wordDropoutKeep=0.6,
                           cohDropoutKeep=0.4)

    b.index_coherence()
