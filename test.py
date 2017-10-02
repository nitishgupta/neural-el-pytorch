import os
import sys
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class ELModel(nn.Module):
    def __init__(self, device_id,
                 numens, num_cands,
                 edim, numwords, init_range):
        super(ELModel, self).__init__()
        self.device_id=device_id
        self.numcands = num_cands
        self.numens = numens
        self.edim = edim

        self.init_range = init_range

        ''' Entity Representations '''
        self.entityembeds = nn.Embedding(self.numens, self.edim,
                                         sparse=True)

        self.entityembeds.weight.data.uniform_(-self.init_range,
                                               self.init_range)

        self.docmat = Parameter(torch.Tensor(numwords, self.edim))


    def forward(self, **kargs):
        cand_entities = kargs['cands']   # [B, C]
        docsparse = kargs['doc']   # [B, D]
        bs = cand_entities.size()[0]

        ''' This actually comes from a network. Omitted for brevity '''
        # context_encoded = Variable(torch.randn(bs, self.edim))
        # context_encoded = context_encoded.cuda(0)

        context_encoded = torch.mm(docsparse, self.docmat)

        # [B, C, H]
        cand_entities_embed = self.entityembeds(cand_entities)

        context_encoded_expanded = context_encoded.unsqueeze(1)
        context_encoded_expanded = context_encoded_expanded.expand(
            bs, self.numcands, self.edim)

        # [B, C]
        cand_en_scores = context_encoded_expanded.mul(
            cand_entities_embed).sum(2)

        # [B, C]
        cand_en_probs = F.softmax(cand_en_scores)

        return (cand_en_scores, cand_en_probs)

    def lossfunc(self, **kargs):
        predwidscores = kargs['predwidscores']
        truewidvec = kargs['truewidvec']

        elloss = F.cross_entropy(input=predwidscores, target=truewidvec)

        return elloss


def optstep(optimizer, loss):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


if __name__=='__main__':
    numens = 600000
    numwords = 1500000
    num_cands = 30
    edim = 100
    init_range = 0.001
    bs = 1000

    model = ELModel(0, numens, num_cands, edim, numwords, init_range)
    print("Size of entity embds: {}".format(model.entityembeds.weight.size()))
    model.cuda(0)

    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.1)

    for i in range(0, 10000):
        cands = np.random.randint(low=0, high=numens, size=(bs, num_cands))
        cands = Variable(torch.LongTensor(cands))
        cands = cands.cuda(0)

        ind = [[random.randint(0, bs-1), random.randint(0, numwords-1)] for j in range(0,10000)]
        numelems =  len(ind)
        ind = torch.LongTensor(ind)
        v = torch.FloatTensor([random.random() for j in range(0, numelems)])
        docsparse = torch.sparse.FloatTensor(ind.t(), v, torch.Size([bs, numwords]))
        docsparse = Variable(docsparse.cuda(0))

        (cand_en_scores, cand_en_probs) = model(cands=cands, doc=docsparse)
        truewidvec = Variable(torch.LongTensor([0]*bs))
        truewidvec = truewidvec.cuda(0)

        loss = model.lossfunc(predwidscores=cand_en_scores,
                              truewidvec=truewidvec)
        optstep(optimizer, loss)

        print("Step : {} Loss : {}".format(i, loss))
