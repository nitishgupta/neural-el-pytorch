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

class ContextEncoder(nn.Module):
    def __init__(self, device_id, contextdim, wdim, lstmsize, numfflayers,
                 doccontextinsize, docffnumlayers, contextmergeffnlayers,
                 dropout, init_range):
        super(ContextEncoder, self).__init__()
        self.device_id = device_id
        self.dropoutlayer = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.doccontextinsize = doccontextinsize
        self.hdim = contextdim

        self.leftlstm = nn.LSTM(wdim, lstmsize, 1, batch_first=True)
        self.rightlstm = nn.LSTM(wdim, lstmsize, 1, batch_first=True)

        self.localcontextff = buildmlp(
            inputdim=2*lstmsize, outputdim=contextdim, hdim=contextdim,
            nlayers=1, dropoutlayer=self.dropoutlayer)

        # self.doccontextff = buildmlp(
        #     inputdim=doccontextinsize, outputdim=contextdim, hdim=contextdim,
        #     nlayers=docffnumlayers, dropoutlayer=self.dropoutlayer)
        self.docencodemat = Parameter(torch.Tensor(doccontextinsize,
                                                   contextdim))   # [D, H]

        self.contextmergeff = buildmlp(
            inputdim=2*contextdim, outputdim=contextdim, hdim=contextdim,
            nlayers=contextmergeffnlayers, dropoutlayer=self.dropoutlayer)

    def _cuda(self, m):
        if self.device_id is not None:
            return m.cuda(self.device_id)
        return m

    def zero_hid(self, nlayers, bs, hsize):
        hid = Variable(torch.zeros(nlayers, bs, hsize))
        hid = self._cuda(hid)
        return hid

    def forward(self, left_input, right_input, left_lens, right_lens,
                doc_input):
        left_input = self.dropoutlayer(left_input)
        right_input = self.dropoutlayer(right_input)

        left_encoded = self._getLastLSTMOutput(self.leftlstm,
                                               left_input, left_lens)
        right_encoded = self._getLastLSTMOutput(self.rightlstm,
                                                right_input, right_lens)

        locallstm_cat = torch.cat((left_encoded, right_encoded), 1)  # [B, 2H]
        locallstm_cat = self.dropoutlayer(locallstm_cat)

        local_encoded = self.localcontextff(locallstm_cat)  # [B, H]
        local_encoded = self.dropoutlayer(local_encoded)

        # doc_input : sparse [B, D] , self.docencodemat: dense [D, H]
        doc_encoded = torch.mm(doc_input, self.docencodemat)    # [B, H]

        contextff_in = torch.cat((local_encoded, doc_encoded), 1)    # [B, 2H]
        contextff_in = self.dropoutlayer(contextff_in)
        contextff_in = self.relu(contextff_in)

        context_encoded = self.contextmergeff(contextff_in)
        context_encoded = local_encoded

        return context_encoded



    def _getLastLSTMOutput(self, lstm, x, lens):
        bs = x.size()[0]
        (h0, c0) = (self.zero_hid(1, bs, lstm.hidden_size),
                    self.zero_hid(1, bs, lstm.hidden_size))

        sortedlens, sortedidxs = torch.sort(lens, dim=0, descending=True)
        x = x[sortedidxs.data]
        packed_x = packseq(x, list(sortedlens.data), batch_first=True)
        # Forward propagate RNN
        out, (h, c) = lstm(packed_x, (h0, c0))
        h = h.squeeze(0)
        h_unsort = self._cuda(Variable(h.data.new(*h.data.size())))

        soridx2d = sortedidxs.unsqueeze(1).expand(h_unsort.size())
        h_unsort = h_unsort.scatter_(0, soridx2d, h)

        return h_unsort
