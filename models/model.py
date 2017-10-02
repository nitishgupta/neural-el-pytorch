import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from models.contextencoder import ContextEncoder
from models.typeencoder import TypeEncoder

class ELModel(nn.Module):
    def __init__(self, device_id, wordvocab, cohvocab, envocab, typevocab,
                 wdim, edim, num_cands,
                 hsize, mlp_nlayers,
                 dropout, init_range,
                 mentyping, entyping, descencoding):
        super(ELModel, self).__init__()
        self.device_id=device_id
        (self.w2idx, self.idx2w) = wordvocab
        self.wordvocabsize = len(self.w2idx)
        (self.knwid2idx, self.idx2knwid) = envocab
        self.envocabsize = len(self.knwid2idx)
        (self.cohstr2idx, self.idx2cohstr) = cohvocab
        self.numcohstrs = len(self.cohstr2idx)
        (self.typ2idx, self.idx2typ) = cohvocab
        self.numtypes = len(self.typ2idx)
        self.numcands = num_cands

        self.edim = edim
        self.wdim = wdim

        self.init_range = init_range
        self.dropout = dropout

        ''' Entity Representations '''
        self.entityembeds = nn.Embedding(self.envocabsize, self.edim,
                                         sparse=True)

        # Context Encoder
        self.contextencoder = ContextEncoder(
            self.device_id, self.edim, self.wdim, self.edim, numfflayers=1,
            doccontextinsize=self.numcohstrs, docffnumlayers=1,
            contextmergeffnlayers=1, dropout=self.dropout,
            init_range=self.init_range)

        # Type Encoder
        self.typeencoder = TypeEncoder(self.device_id, self.numtypes,
                                       self.edim, self.dropout,
                                       self.init_range)

        # Desc Encoder


        self.init_weights()

    def init_weights(self):
        self.entityembeds.weight.data.uniform_(-self.init_range,
                                               self.init_range)

    def _cuda(self, m):
        if self.device_id is not None:
           return m.cuda(self.device_id)
        return m

    def forward_context(self, **kargs):
        left_context = kargs['leftb']
        right_context = kargs['rightb']
        left_lens = kargs['leftlens']
        right_lens = kargs['rightlens']
        sparse_doc_vecs = kargs['docb']
        cand_entities = kargs['wididxsb']   # [B, C]
        bs = left_context.size()[0]

        # [B, H]
        '''
        context_encoded = self.contextencoder.forward(
            left_input=left_context, right_input=right_context,
            left_lens=left_lens, right_lens=right_lens,
            doc_input=sparse_doc_vecs)
        '''
        context_encoded = Variable(torch.randn(bs, self.edim))
        context_encoded = context_encoded.cuda(0)

        # [B, T]
        mentype_probs = self.typeencoder.forward(context_encoded)

        # [B, C, H]
        cand_entities_embed = self.entityembeds(cand_entities)


        context_encoded_expanded = context_encoded.unsqueeze(1).expand(
            bs, self.numcands, self.edim)

        # [B, C]
        cand_en_scores = context_encoded_expanded.mul(
            cand_entities_embed).sum(2)

        # [B, C]
        cand_en_probs = F.softmax(cand_en_scores)

        return (cand_en_scores, cand_en_probs, mentype_probs)


    def lossfunc(self, **kargs):
        predwidscores = kargs['predwidscores']
        truewidvec = kargs['truewidvec']

        elloss = F.cross_entropy(input=predwidscores, target=truewidvec)

        return elloss
