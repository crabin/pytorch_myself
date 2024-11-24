import torch.nn as nn
import torch

rnn = nn.RNN(input_size=100, hidden_size=20, num_layers=2)

print(rnn._parameters.keys())

print(rnn.weight_hh_l0.shape, rnn.weight_ih_l0.shape)

x = torch.randn(10, 3, 100)

out, h = rnn(x, torch.zeros(2, 3, 20))

print(out.shape, h.shape)


cell1 = nn.RNNCell(100, 30)
cell2 = nn.RNNCell(30, 20)
h1 = torch.zeros(3, 30)
h2 = torch.zeros(3, 20)
for xi in x:
    h1 = cell1(xi, h1)
    h2 = cell2(h1, h2)

print(h2.shape)