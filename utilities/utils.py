import os
import pdb
import json
import copy
import pickle
import random

import torch
import numpy as np
from torch.autograd import Variable


def backward_hook(module, inputGrad, outGrad):
    #print(grad)
    #pdb.set_trace()
    print(module)
    #return outGrad
    return None


def save(fname, obj):
    """Save an object as pickle dump"""
    with open(fname, 'wb') as f:
        pickle.dump(obj, f)


def load(fname):
    """Load object from pickle dump"""
    with open(fname, 'rb') as f:
        return pickle.load(f)


def load_json(json_fname):
    """Load all content of a json file"""
    assert os.path.exists(json_fname)
    with open(json_fname, 'r') as f:
        json_data = json.load(f)
    return json_data


def save_checkpoint(m, o, steps, beststeps, path):
    try:
        ckptdir = os.path.dirname(path)
        if not os.path.exists(ckptdir):
            os.makedirs(ckptdir)
        state = {
    				'm_state_dict': m.state_dict(),
    				'o_state_dict': o.state_dict(),
    				'steps': steps,
                    'beststeps': beststeps
    			}
        torch.save(state, path)
        print("Model saved to path: {}".format(path))
    except:
         print("Saving failed to path: {}".format(path))


def load_checkpoint(path, m, o):
    print("[#] Loading model checkpoint from : {}".format(path))
    try:
        state = torch.load(path)
        loaded_m_state = state['m_state_dict']
        loaded_o_state = state['o_state_dict']
        steps = state['steps']
        if 'beststeps' in state:
            beststeps = state['beststeps']
        else:
            beststeps = steps

        try:
        	m.load_state_dict(loaded_m_state)
        except:
            print("  [#] Partial model in ckpt. Loading ..")
            mstate = m.state_dict()
            #print(mstate.keys())
            #print("\n")
            #print(loaded_m_state.keys())
            #print(set(mstate.keys()).difference(set(loaded_m_state.keys())))
            #print(set(loaded_m_state.keys()).difference(set(mstate.keys())))
            keystobeupdated = set(mstate.keys()).intersection(set(loaded_m_state.keys()))
            for k in keystobeupdated:
            	mstate[k] = loaded_m_state[k]
            print("   [#] State dict keys updated ...")
            m.load_state_dict(mstate)
            print("   [#] State dict loaded ...")

        # o.load_state_dict(loaded_o_state)
        print("[#] Loading successful. Steps:{} BestSteps:{}".format(
            steps, beststeps))
        return steps
    except:
        print("[#] Loading Failed")
        return -1



def save_model(model, file_name):
    if file_name != '':
        with open(file_name, 'wb') as f:
            torch.save(model.state_dict(), f)

def save_optim(o, path):
    if file_name != '':
        with open(file_name, 'wb') as f:
            torch.save(o.state_dict(), path)


def load_model(file_name):
    with open(file_name, 'rb') as f:
        return torch.load(f)


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def tocpuNPList(var):
    return var.data.cpu().numpy().tolist()



def use_cuda():
    assert torch.cuda.is_available(), 'CUDA is not available'
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

def toCudaVariable(device_id, *tensors):
    cudaVariables = []
    for t in tensors:
        if device_id is not None:
            t = t.cuda(device_id)
        t = Variable(t)
        cudaVariables.append(t)
    return cudaVariables

def toVariable(tensors):
    variables = []
    for t in tensors:
        variables.append(Variable(t))
    return variables

def tocuda(tensors, device_id):
    cuda_tensors = []
    for t in tensors:
        t = t.cuda(device_id)
        cuda_tensors.append(t)
    return cuda_tensors

def toVariable(tensors):
    variables = []
    for t in tensors:
        variables.append(Variable(t))
    return variables

def sortDictOnKeys(d):
    sorted_tuples = sorted(d.items(), key=lambda x:x[0])
    return sorted_tuples

def sortDictOnValues(d):
    sorted_tuples = sorted(d.items(), key=lambda x:x[1])
    return sorted_tuples


def _parseBoolArg(arg):
    return True if arg.lower() == 'true' else False


def round_all(stuff, prec):
    if isinstance(stuff, list):
        return [round_all(x, prec) for x in stuff]
    if isinstance(stuff, tuple):
        return tuple(round_all(x, prec) for x in stuff)
    if isinstance(stuff, float):
        return round(float(stuff), prec)
    else:
        return stuff
