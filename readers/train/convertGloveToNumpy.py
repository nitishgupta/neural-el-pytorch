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

# person: 25, event: 10, organization: 5, location: 1

class GloveNumpy(object):
    def __init__(self, config, vocabloader):

        '''
        Reader especially for training data, but can be used for test data as
        validation and test file inputs. The requirement is that the mention candidates
        should be added to the TrValCandidateDict using readers.train.crosswikis_vocab

        DataType 0/1 corresponds to train/val_file
        '''
        self.config = config
        self.vocabloader = vocabloader
        # Word Vocab
        (self.word2idx, self.idx2word) = vocabloader.getGloveWordVocab()
        self.num_words = len(self.idx2word)
        print("[#] WordVocab loaded. Size of vocab : {}".format(
            self.num_words))

        vtime = time.time()
        self.word2vec = vocabloader.loadGloveVectors()
        print("[#] Glove Vectors loaded!")
        ttime = (time.time() - vtime)/float(60)
        print("[#] Time to load vectors : {} mins".format(ttime))

        self.convertGloveToNumpy()
        self.glovenumpy = self.vocabloader.loadGloveNumpy()
        print("Glove numpy loaded")
        self.testGloveNumpy()

    def convertGloveToNumpy(self):
        if os.path.exists(self.config.glove_numpy_pkl):
            print("Glove numpy already exists. ")
        else:
            print("Making glove numpy")
            wordvecs = []
            for idx, word in enumerate(self.idx2word):
                wordvecs.append(self.word2vec[word])

            glovenumpy = np.array(wordvecs)
            utils.save(self.config.glove_numpy_pkl, glovenumpy)
            print("done")

    def testGloveNumpy(self):
        print("Testing glove numpy")
        word = self.idx2word[10]
        wordidx = 10
        vec1 = self.word2vec[word]
        vec2 = self.glovenumpy[wordidx]
        vec1 = np.array(vec1)
        out = np.amax(np.absolute(vec1 - vec2))
        print(out)



if __name__ == '__main__':
    sttime = time.time()
    configpath = "configs/config.ini"
    config = Config(configpath)
    vocabloader = VocabLoader(config)
    b = GloveNumpy(config=config, vocabloader=vocabloader)
