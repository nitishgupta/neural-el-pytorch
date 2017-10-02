import pprint
import configparser

pp = pprint.PrettyPrinter()

class Config(object):
    def __init__(self, paths_config, verbose=False):
        config = configparser.ConfigParser()
        config._interpolation = configparser.ExtendedInterpolation()
        config.read(paths_config)
        print(paths_config)

        c = config['DEFAULT']

        d = {}
        for k in c:
            d[k] = c[k]

        self.wiki_mentions_dir = d['wiki_mentions_dir']
        self.train_mentions_dir = d['train_mentions_dir']
        self.val_mentions_dir = d['val_mentions_dir']
        self.val_mentions_file = d['val_mentions_file']
        if 'test_mentions_dir' in d.keys():
            self.test_mentions_dir = d['test_mentions_dir']
            self.test_mentions_file = d['test_mentions_file']
        self.cold_mentions_file = d['cold_val_mentions_file']
        self.vocab_dir = d['vocab_dir']
        if 'colddir' in d.keys():
            self.coldDir = d['colddir']
        if 'coldwid_vocab_pkl' in d.keys():
            self.coldwid_vocab_pkl = d['coldwid_vocab_pkl']

        # Wid2Idx for Known Entities ~ 620K (readers.train.vocab.py)
        self.kwnwid_vocab_pkl = d['kwnwid_vocab_pkl']
        # FIGER Type label 2 idx (readers.train.vocab.py)
        self.label_vocab_pkl = d['label_vocab_pkl']
        # EntityWid: [FIGER Type Labels]
        self.wid2typelabels_vocab_pkl = d['wid_typelabels_vocab_pkl']
        # CoherenceStr2Idx at various thresholds (readers.train.vocab.py)
        if 'cohstringg9_vocab_pkl' in d.keys():
            self.cohstringG9_vocab_pkl = d['cohstringg9_vocab_pkl']

        # wid2Wikititle for whole KB ~ 3.18M (readers.train.vocab.py)
        self.widWiktitle_pkl = d['widwiktitle_pkl']

        # For training, validation and test data
        # (surface, wid) : ([truecandidx, cands_idxs], [cprobs]) (readers.train.crosswikis_vocab)
        self.trval_kwnidx_cands_pkl = d['trval_kwnidx_cands_pkl']

        # TestDatasets pruned crosswikis (readers.test.test_crosswikis_vocabs)
        self.test_allen_cwikis_pkl = d['test_allen_cwikis_pkl']
        # TestDatasets pruned crosswikis only containing known entities (readers.test.test_crosswikis_vocabs)
        self.test_kwnen_cwikis_pkl = d['test_kwnen_cwikis_pkl']

        if 'knownwid2descvectors' in d.keys():
            self.knownwid2descvectors = d['knownwid2descvectors']

        self.crosswikis_pkl = d['crosswikis_pkl']
        self.crosswikis_pruned_pkl = d['crosswikis_pruned_pkl']
        self.word2vec_bin_gz = d['word2vec_bin_gz']
        self.glove_pkl = d['glove_pkl']
        self.glove_word_vocab_pkl = d['glove_word_vocab_pkl']

        self.datasets_dir = d['datasets_dir']
        self.ace_mentions_file = d['ace_mentions_file']

        if 'ace_known_file' in d.keys():
            self.ace_known_file = d['ace_known_file']
        if 'ace_ner_file' in d.keys():
            self.ace_ner_file = d['ace_ner_file']

        if 'aida_nonnme_mens_dir' in d.keys():
            self.aida_nonnme_mens_dir = d['aida_nonnme_mens_dir']
            self.aida_nonnme_train_file = d['aida_nonnme_train_file']
            self.aida_nonnme_dev_file = d['aida_nonnme_dev_file']
            self.aida_nonnme_test_file = d['aida_nonnme_test_file']

        if 'aida_inkb_mens_dir' in d.keys():
            self.aida_inkb_mens_dir = d['aida_inkb_mens_dir']
            self.aida_inkb_train_file = d['aida_inkb_train_file']
            self.aida_inkb_dev_file = d['aida_inkb_dev_file']
            self.aida_inkb_test_file = d['aida_inkb_test_file']

        if 'aida_ner_mens_dir' in d.keys():
            self.aida_ner_dev_file = d['aida_ner_dev_file']
            self.aida_ner_test_file = d['aida_ner_test_file']

        if 'aida_kwn_dev_file' in d.keys():
            self.aida_kwn_dev_file = d['aida_kwn_dev_file']
            self.aida_kwn_test_file = d['aida_kwn_test_file']

        if 'wikidata_inkb_test_file' in d.keys():
            self.wikidata_inkb_test_file = d['wikidata_inkb_test_file']

        if 'msnbc_inkb_test_file' in d.keys():
            self.msnbc_inkb_test_file = d['msnbc_inkb_test_file']

        if 'ace2005_test_file' in d.keys():
            self.ace2005_test_file = d['ace2005_test_file']

        if 'ace2005_known_file' in d.keys():
            self.ace2005_known_file = d['ace2005_known_file']

        if 'figer_test_file' in d.keys():
            self.figer_test_file = d['figer_test_file']

        if 'ontonotes_test_file' in d.keys():
            self.ontonotes_test_file = d['ontonotes_test_file']

        if verbose:
            pp.pprint(d)

    #endinit

if __name__=='__main__':
    c = Config("configs/allnew_mentions_config.ini", verbose=True)
