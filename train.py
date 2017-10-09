import os
import sys
import time
import copy
import pprint
import select
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

from utilities import utils
from readers.train.training_reader import TrainingDataReader
from readers.train.test_reader import TestDataReader
from readers.config import Config
from readers.vocabloader import VocabLoader
from models.model import ELModel

pp = pprint.PrettyPrinter()

torch.backends.cudnn.benchmark = False

MODELTYPES = ['ELModel']

class Trainer():
    """Initializes reader, model, optimizer and trains on data.

    All parameters are initialized as global variables in __main__
    """

    def __init__(self):
        print("[#] Launching Training Job. DeviceId : {}".format(device_id))
        self.config = Config(configpath, verbose=False)
        self.vocabloader = VocabLoader(self.config)
        test_file = self.config.aida_kwn_dev_file
        valbs = 2

        self.tr_reader = TrainingDataReader(
            config=self.config,
            vocabloader=self.vocabloader,
            val_file=self.config.aida_kwn_dev_file,
            num_cands=30,
            batch_size=bs)
        self.test_reader = TestDataReader(
            config=self.config,
            vocabloader=self.vocabloader,
            val_file=test_file,
            num_cands=30,
            batch_size=valbs)

        self.wvocab_size = len(self.tr_reader.word2idx)
        self.envocab_size = len(self.tr_reader.knwid2idx)
        self.typevocab_size = len(self.tr_reader.label2idx)
        self.cohvocab_size = len(self.tr_reader.cohG92idx)

        print("[#] Word Vocab : {}, Entity Vocab: {}, Type Vocab: {} "
              "CohString Vocab : {}".format(self.wvocab_size,
                self.envocab_size, self.typevocab_size, self.cohvocab_size))

        if modeltype == 'ELModel':
            print("[#] MODEL : ELModel")
            self.model = ELModel(
                device_id=device_id,
                wordvocab=(self.tr_reader.word2idx, self.tr_reader.idx2word),
                cohvocab=(self.tr_reader.cohG92idx, self.tr_reader.idx2cohG9),
                envocab=(self.tr_reader.knwid2idx, self.tr_reader.idx2knwid),
                typevocab=(self.tr_reader.label2idx, self.tr_reader.idx2label),
                wdim=wdim, edim=endim, num_cands=30,
                hsize=endim, mlp_nlayers=1,
                dropout=dropout, init_range=init_range,
                mentyping=mentype, entyping=entype, descencoding=endesc)
        if modeltype not in MODELTYPES:
            print("Invalid modeltype : {}".format(modeltype))
            sys.exit()

        if device_id is not None:
            self.model.cuda(device_id)

        if optim == 'adam':
            print("[#] OPTIM : ADAM")
            self.optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=lr)
        elif optim == 'sgd':
            print("[#] OPTIM : SGD LR:{} Momentum:{} Nestrov:{}".format(
                lr, momentum, nestrov))
            self.optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=lr,
                momentum=momentum, nesterov=nestrov)
        else:
            print("Wrong Optimizer")
            sys.exit(0)

        timeout = 5
        print("Press any key to run (or wait {} seconds) ... ".format(timeout))
        rlist, wlist, xlist = select.select([sys.stdin], [], [], timeout)

    # @profile
    def train(self):
        # Initialize saver, model parameters (hidden inputs to lstm)
        # Load model if needed.
        self.model.train()
        start_time = time.time()
        avg_loss, avg_elloss, avg_mtypeloss = 0.0, 0.0, 0.0
        epochs = self.tr_reader.tr_epochs
        steps = 0
        ncorrect, ntotal = 0, 0
        ncorrectOA, ntotalOA = 0, 0
        ncorrectB, ntotalB = 0, 0

        bestmodel, bestval, beststep = self.model, 0.0, 0
        bestFinalVal = 0.0

        readtime, convtime, processtime = 0, 0, 0

        # while ((steps < maxsteps and bestFinalVal < 0.999) or
        #        (CURR_SWITCHES < len(CURRICULUM_ORDER) - 1)):
        while steps < maxtrsteps:
            steps += 1
            # print(curr)
            rtimestart = time.time()
            b = self.tr_reader.next_train_batch()
            (leftb, leftlens, rightb, rightlens,
             docb, typesb, wididxsb, widprobsb) = (
                b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7])
            (ind, vals, dvsize) = docb
            readtime += (time.time() - rtimestart)
            ctimestart = time.time()
            ind = torch.LongTensor(ind)
            vals = torch.FloatTensor(vals)
            docb = torch.sparse.FloatTensor(
                ind.t(), vals, torch.Size(dvsize))
            (leftb, leftlens, rightb, rightlens,
             typesb, wididxsb, widprobsb) = (
                torch.FloatTensor(leftb),
                torch.LongTensor(leftlens), torch.FloatTensor(rightb),
                torch.LongTensor(rightlens), torch.FloatTensor(typesb),
                torch.LongTensor(wididxsb), torch.FloatTensor(widprobsb))
            (leftb, leftlens, rightb, rightlens, docb,
             typesb, wididxsb, widprobsb) = utils.toCudaVariable(
                device_id,
                leftb, leftlens, rightb, rightlens, docb,
                typesb, wididxsb, widprobsb)

            truewidvec = utils.toCudaVariable(device_id,
                                              torch.LongTensor([0]*bs))[0]
            convtime += (time.time() - ctimestart)
            ptimestart = time.time()

            rets = self.model.forward_context(
                leftb=leftb, leftlens=leftlens,
                rightb=rightb, rightlens=rightlens,
                docb=docb, wididxsb=wididxsb)
            (wididxscores, wididxprobs, mentype_probs) = (rets[0], rets[1],
                                                          rets[2])

            (loss, elloss, mentype_loss) = self.model.lossfunc(
                mentype=mentype,
                predwidscores=wididxscores, truewidvec=truewidvec,
                mentype_probs=mentype_probs, mentype_trueprobs=typesb)

            self.optstep(loss)
            loss = loss.data.cpu().numpy()[0]
            elloss = elloss.data.cpu().numpy()[0]
            mentype_loss = mentype_loss.data.cpu().numpy()[0]
            avg_loss += loss
            avg_elloss += elloss
            avg_mtypeloss += mentype_loss

            processtime += (time.time() - ptimestart)

            if steps % log_interval == 0:
                totaltime = readtime + processtime + convtime
                print()
                avg_loss = utils.round_all(avg_loss/log_interval, 3)
                avg_elloss = utils.round_all(avg_elloss/log_interval, 3)
                avg_mtypeloss = utils.round_all(avg_mtypeloss/log_interval, 3)
                print("[{}, {}, rt:{:0.1f} secs ct:{:0.1f} pt:{:0.1f} "
                      "tt:{:0.1f} secs]: L:{} EL:{} MenTypL:{}".format(
                        steps, self.tr_reader.tr_epochs,
                        readtime, convtime, processtime, totaltime, avg_loss,
                        avg_elloss, avg_mtypeloss))
                readtime, convtime, processtime = 0, 0, 0
                # tracc = float(ncorrect)/float(ntotal)
                # oAtracc = float(ncorrectOA)/float(ntotalOA) if ntotalOA != 0.0 else 0.0
                # Btracc = float(ncorrectB)/float(ntotalB) if ntotalB != 0.0 else 0.0
                # avg_loss /= log_interval
                # time_elapsed = float(time.time() - start_time)/60.0
                # print("[{}, {}, {:0.1f} mins]: {}".format(
                #     steps, self.tr_reader.epochs,
                #     time_elapsed, avg_loss))
                # print("TrAcc: {} / {} : {:.3f}".format(
                #     ncorrect, ntotal, tracc))
                # print("OA : {}/{}: {}".format(ncorrectOA, ntotalOA, oAtracc))
                # print("Bool : {}/{}: {}".format(ncorrectB, ntotalB, Btracc))
                # avg_loss = 0.0
                # ntotal=0
                # ncorrect=0
                # ntotalOA = 0
                # ncorrectOA = 0
                # ntotalB = 0
                # ncorrectB = 0

            # if epochs != self.tr_reader.epochs or steps % 15000 == 0:
            if steps % 2000 == 0:
                print("Saving model: {}".format(ckptpath))
                bestmodel = copy.deepcopy(self.model)
                beststep = steps
                utils.save_checkpoint(m=bestmodel, o=self.optimizer,
                                      steps=steps, beststeps=beststep,
                                      path=ckptpath)
                # (vt, vc, va) = self.validation_performance()
                # if va > bestval:
                #     bestval = va
                #     bestmodel = copy.deepcopy(self.model)
                #     beststep = steps
                # if bestval == 0.0 and va == 0.0: # keep latest model
                #     bestval = va
                #     bestmodel = copy.deepcopy(self.model)
                #     beststep = steps
                # # Check if final curricula is reached, then update bestFinalVal
                # if CURR_SWITCHES == len(CURRICULUM_ORDER) - 1:
                #     bestFinalVal = bestval
                # print("[##] Total: {}. Correct: {}. Acc: {:0.3f} "
                #       "[B:{:.3f} E:{}]".format(vt, vc, va, bestval, beststep))
                # print("[##] Best Final Val : {}\n".format(bestFinalVal))
                # print("Saving model: {}".format(ckptpath))
                # # Saving latest model
                # bestmodel = copy.deepcopy(self.model)
                # utils.save_checkpoint(m=bestmodel, o=self.optimizer,
                #                       steps=steps, beststeps=beststep,
                #                       path=ckptpath)
                # epochs = self.tr_reader.epochs

        return (bestmodel, bestval, beststep, steps)

    def validation(self):
        # Initialize saver, model parameters (hidden inputs to lstm)
        # Load model if needed.
        self.model.train(False)
        start_time = time.time()
        avg_loss, avg_elloss, avg_mtypeloss = 0.0, 0.0, 0.0
        epochs = self.test_reader.val_epochs
        steps = 0
        ncorrect, ntotal = 0, 0
        ncorrectOA, ntotalOA = 0, 0
        ncorrectB, ntotalB = 0, 0

        bestmodel, bestval, beststep = self.model, 0.0, 0
        bestFinalVal = 0.0

        readtime, convtime, processtime = 0, 0, 0

        # while ((steps < maxsteps and bestFinalVal < 0.999) or
        #        (CURR_SWITCHES < len(CURRICULUM_ORDER) - 1)):
        while self.test_reader.val_epochs < 1:
            steps += 1
            # print(curr)
            rtimestart = time.time()
            b = self.test_reader.next_test_batch()
            (leftb, leftlens, rightb, rightlens,
             docb, typesb, wididxsb, widprobsb) = (
                b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7])
            (ind, vals, dvsize) = docb
            readtime += (time.time() - rtimestart)
            ctimestart = time.time()
            ind = torch.LongTensor(ind)
            vals = torch.FloatTensor(vals)
            docb = torch.sparse.FloatTensor(
                ind.t(), vals, torch.Size(dvsize))
            (leftb, leftlens, rightb, rightlens,
             typesb, wididxsb, widprobsb) = (
                torch.FloatTensor(leftb),
                torch.LongTensor(leftlens), torch.FloatTensor(rightb),
                torch.LongTensor(rightlens), torch.FloatTensor(typesb),
                torch.LongTensor(wididxsb), torch.FloatTensor(widprobsb))
            (leftb, leftlens, rightb, rightlens, docb,
             typesb, wididxsb, widprobsb) = utils.toCudaVariable(
                device_id,
                leftb, leftlens, rightb, rightlens, docb,
                typesb, wididxsb, widprobsb)

            truewidvec = utils.toCudaVariable(device_id,
                                              torch.LongTensor([0]*2))[0]
            convtime += (time.time() - ctimestart)
            ptimestart = time.time()

            rets = self.model.forward_context(
                leftb=leftb, leftlens=leftlens,
                rightb=rightb, rightlens=rightlens,
                docb=docb, wididxsb=wididxsb)
            (wididxscores, wididxprobs, mentype_probs) = (rets[0], rets[1],
                                                          rets[2])

            (loss, elloss, mentype_loss) = self.model.lossfunc(
                mentype=mentype,
                predwidscores=wididxscores, truewidvec=truewidvec,
                mentype_probs=mentype_probs, mentype_trueprobs=typesb)

            loss = loss.data.cpu().numpy()[0]
            elloss = elloss.data.cpu().numpy()[0]
            mentype_loss = mentype_loss.data.cpu().numpy()[0]

            processtime += (time.time() - ptimestart)

            totaltime = readtime + processtime + convtime
            print()
            loss = utils.round_all(loss/log_interval, 3)
            elloss = utils.round_all(elloss/log_interval, 3)
            mentype_loss = utils.round_all(mentype_loss/log_interval, 3)
            print("[{}, {}, rt:{:0.1f} secs ct:{:0.1f} pt:{:0.1f} "
                  "tt:{:0.1f} secs]: L:{} EL:{} MenTypL:{}".format(
                      steps, self.tr_reader.tr_epochs,
                      readtime, convtime, processtime, totaltime, loss,
                      elloss, mentype_loss))
            readtime, convtime, processtime = 0, 0, 0

    def optstep(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        # print("Rnn hh grad: {}".format(self.model.rnncell.w_hh.grad.data))
        # torch.nn.utils.clip_grad_norm(self.model.parameters(), clip)
        # print("Rnn hh grad (after clipping): {}".format(self.model.rnncell.w_hh.grad.data))
        self.optimizer.step()

def getCkptName(rootname):
    ckptname = rootname
    ckptname += '_s' + str(seed)    # Seed
    ckptname += '_wdim_' + str(wdim)    # gradient clip threshold
    ckptname += '_lr' + str(lr)    # learningRate
    ckptname += '_bs' + str(bs)    # BatchSize
    ckptname += '_clip_' + str(clip)    # gradient clip threshold
    ckptname += '_endim_' + str(endim)    # gradient clip threshold
    ckptname += '_mentype_' + str(mentype)    # gradient clip threshold
    ckptname += '_entype_' + str(entype)    # gradient clip threshold
    ckptname += '_endesc_' + str(endesc)    # gradient clip threshold
    ckptname += '.pkl'
    return ckptname

if __name__ == '__main__':
    torch.set_num_threads = 4

    parser = argparse.ArgumentParser()
    parser.add_argument('--configpath', default='configs/config.ini')
    parser.add_argument('--model', type=str, default='ELModel',
                        help='Type of model/baseline to run')

    parser.add_argument('--maxtrsteps', type=int, default=1000,
                        help='Number of training steps')
    parser.add_argument('--ckptname', type=str, default='model',
                        help='Name for the checkpoint file - ckptname.pkl')
    parser.add_argument('--ckptroot', type=str, default='/scratch/ngupta19/elpytorch/',
                        help='Name for the checkpoint dataset dir')

    parser.add_argument('--optim', type=str, default='sgd',
                        help='Optimizer')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Learning Rate')

    parser.add_argument('--bs', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--worddropout', type=float, default=0.4)
    parser.add_argument('--cohdropout', type=float, default=0.6)
    parser.add_argument('--init_range', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.0,
                        help='Momentum for SGD')
    parser.add_argument('--nestrov', action='store_true', default=False,
                        help='Use nestrov momentum')
    parser.add_argument('--wdim', type=int, default=300,
                        help='Word embedding size')
    parser.add_argument('--endim', type=int, default=100,
                        help='Entity embedding size')

    parser.add_argument('--log_interval', type=int, default=100,
                        help='Print progress every log_interval steps')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='use CUDA')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed')

    parser.add_argument('--mode', type=str, default='train',
                        help='Mode to run [train, val, analysis]')
    parser.add_argument('--clip', type=float, default=0.5,
                        help='Gradient Clip Threshold')
    parser.add_argument('--mentype', action='store_true', default=False,
                        help='use Mention Typing Loss')
    parser.add_argument('--entype', action='store_true', default=False,
                        help='use Entity Typing Loss')
    parser.add_argument('--endesc', action='store_true', default=False,
                        help='use Entity Description Loss')



    args = parser.parse_args()
    pp.pprint(args)

    configpath = args.configpath
    modeltype = args.model

    '''Command-line Arguments'''
    seed = args.seed
    lr = args.lr
    optim = args.optim
    bs = args.bs
    dropout = args.dropout
    worddropout = args.worddropout
    cohdropout = args.cohdropout
    init_range = args.init_range
    momentum = args.momentum
    nestrov = (args.nestrov and momentum > 0.0)
    log_interval = args.log_interval
    maxtrsteps = args.maxtrsteps
    wdim = args.wdim
    endim = args.endim
    use_cuda = args.cuda
    mode = args.mode
    ckptroot = args.ckptroot
    ckptname = args.ckptname
    modeltype = args.model
    clip = args.clip
    mentype = args.mentype
    entype = args.entype
    endesc = args.endesc



    assert modeltype in MODELTYPES, "Model type is incorrect"

    if torch.cuda.is_available():
        if not use_cuda:
            print("\n### WARNING: You have a CUDA device, "
                  "so you should probably run with --cuda\n")

    if use_cuda:
        print("Using GPU device_id:0")
        device_id = 0
        utils.use_cuda()
        torch.cuda.set_device(device_id)
    else:
        print("Cuda Not Available!!")
        device_id = None

    utils.set_seed(seed)

    ckptfilename = getCkptName(args.ckptname)
    ckptpath = os.path.join(ckptroot, modeltype, ckptfilename)
    print("CKPT PATH: {}".format(ckptpath))

    # Initialized reader, model and optimizer
    trainer = Trainer()
    print("Done modelinit")
    print("Mode : {}".format(mode))
    # print(trainer.tr_reader.ans2idx)

    if mode == 'train':
        (bestmodel, bestval, beststeps, steps) = trainer.train()
        print("Saving model: {}".format(ckptpath))
        utils.save_checkpoint(m=bestmodel, o=trainer.optimizer,
                              steps=steps, beststeps=beststeps,
                              path=ckptpath)
        pp.pprint(args)

    elif mode == 'val':
        utils.load_checkpoint(ckptpath, trainer.model, trainer.optimizer)
        trainer.validation()
        # (vt, vc, va) = trainer.validation_performance()
        # print("Total: {}. Validation Acc: {}".format(vt, va))

    sys.exit()
