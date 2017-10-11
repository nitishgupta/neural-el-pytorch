import os
import numpy as np


def eval(candidates, priorProbs, predProbs, idx2wid, wid2Wikititle):
    unkwid = "<unk_wid>"
    trueWidUndef = 0
    trueWidNotCand = 0
    correct = 0
    total = 0
    for wididxs, priors, en_probs in zip(candidates,
                                         priorProbs, predProbs):
        total += 1
        if idx2wid[wididxs[0]] == unkwid:
            trueWidUndef += 1
        else:
            if priors[0] == 0.0:
                trueWidNotCand += 1
            elif np.argmax(en_probs) == 0:
                correct += 1

    return (total, correct, trueWidUndef, trueWidNotCand)


def jointProb(priorProbs, contextProbs):
    jointProbs = []

    for priors, conprobs in zip(priorProbs, contextProbs):
        jprobs = [i+j for (i,j) in zip(priors, conprobs)]
        jointProbs.append(jprobs)

    return jointProbs


def elEval(candidates, priorProbs, contextProbs, idx2wid, wid2Wikititle):
    jointProbs = jointProb(priorProbs, contextProbs)

    print("Priors Evaluation")
    (total, correct,
     trueWidUndef, trueWidNotCand) = eval(candidates, priorProbs,
                                          priorProbs, idx2wid, wid2Wikititle)
    accuracy = float(correct)/float(total)
    print("Total: {} Correct: {} Acc: {}  TrueWidUnk: {} "
          "trueWidNotACand: {}".format(
              total, correct, accuracy, trueWidUndef, trueWidNotCand))

    print("Context Prob Eval")
    (total, correct,
     trueWidUndef, trueWidNotCand) = eval(candidates, priorProbs,
                                          contextProbs, idx2wid, wid2Wikititle)
    accuracy = float(correct)/float(total)
    print("Total: {} Correct: {} Acc: {}  TrueWidUnk: {} "
          "trueWidNotACand: {}".format(
              total, correct, accuracy, trueWidUndef, trueWidNotCand))

    print("Joint Prob Eval")
    (total, correct,
     trueWidUndef, trueWidNotCand) = eval(candidates, priorProbs,
                                          jointProbs, idx2wid, wid2Wikititle)
    accuracy = float(correct)/float(total)
    print("Total: {} Correct: {} Acc: {}  TrueWidUnk: {} "
          "trueWidNotACand: {}".format(
              total, correct, accuracy, trueWidUndef, trueWidNotCand))
