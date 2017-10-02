import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from torch.nn.utils.rnn import pack_padded_sequence as packseq
from torch.nn.utils.rnn import pad_packed_sequence as padseq

def buildmlp(inputdim, outputdim, hdim, nlayers, dropoutlayer):
    """Assumes dropout already applied at input."""
    layers = []
    if nlayers >= 1:
        D = inputdim
        for i in range(0, nlayers):
            layers.append(nn.Linear(D, hdim))
            layers.append(dropoutlayer)
            layers.append(nn.ReLU(inplace=True))
            # layers.append(nn.Sigmoid())
            D = hdim
        layers.append(nn.Linear(hdim, outputdim))

    else:
        layers.append(nn.Linear(inputdim, outputdim))

    return nn.Sequential(*layers)

class TypeEncoder(nn.Module):
    def __init__(self, device_id, numtypes, typedim,
                 dropout, init_range):
        super(TypeEncoder, self).__init__()
        self.device_id = device_id
        self.init_range = init_range
        self.dropoutlayer = nn.Dropout(dropout)
        self.numtypes = numtypes

        self.typeembeds = Parameter(torch.Tensor(numtypes, typedim))

        self.init_weights()

    def init_weights(self):
        self.typeembeds.data.uniform_(-self.init_range, self.init_range)

    def _cuda(self, m):
        if self.device_id is not None:
            return m.cuda(self.device_id)
        return m

    def zero_hid(self, nlayers, bs, hsize):
        hid = Variable(torch.zeros(nlayers, bs, hsize))
        hid = self._cuda(hid)
        return hid

    def forward(self, input_vecs):
        # input_vecs : [B, H], typeembeds = [T, H]
        typeembeds = self.dropoutlayer(self.typeembeds)
        inputsize = input_vecs.size()
        bs = inputsize[0]
        hsize = inputsize[1]
        input_vecs = input_vecs.unsqueeze(1).expand(bs, self.numtypes, hsize)
        typeembeds = typeembeds.unsqueeze(0).expand(bs, self.numtypes, hsize)

        type_scores = typeembeds.mul(input_vecs).sum(2)   # [B, T]
        type_probs = torch.sigmoid(type_scores)    # [B, T]

        return type_probs

    def loss(self, predTypeProb, trueTypeProb):
        typeLoss = F.binary_cross_entropy(input=predTypeProb,
                                          target=trueTypeProb)
        return typeLoss
