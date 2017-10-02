def init_rnn(rnn, init_range, weights=None, biases=None):
    weights = weights or ['weight_ih_l0', 'weight_hh_l0']
    biases = biases or ['bias_ih_l0', 'bias_hh_l0']
    # Init weights
    for w in weights:
        rnn._parameters[w].data.uniform_(-init_range, init_range)
    # Init biases
    for b in biases:
        rnn._parameters[b].data.fill_(0)


def init_rnn_cell(rnn, init_range):
    init_rnn(rnn, init_range,
             ['weight_ih', 'weight_hh'], ['bias_ih', 'bias_hh'])


def init_cont(cont, init_range):
    for m in cont:
        if hasattr(m, 'weight'):
            m.weight.data.uniform_(-init_range, init_range)
        if hasattr(m, 'bias'):
            m.bias.data.fill_(0)
