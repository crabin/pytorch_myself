import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch import optim

batch = 1
seq_len = 50
num_time_steps = seq_len
input_size = 1
output_size = input_size
hidden_size = 10
num_layers = 1
batch_first = True
lr = 0.01


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                          batch_first=batch_first)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden_prev):
        out, hidden_prev = self.rnn(x, hidden_prev)

        out = out.view(-1, hidden_size)  # [batch * seq_len, hidden_size]
        out = self.linear(out)  # [batch * seq_len, output_size]
        out = out.unsqueeze(dim=0)  # [1, batch*seq_len, output_size]
        return out, hidden_prev


def train_RNN():
    model = Net()
    print("model:\n", model)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    hidden_prev = torch.zeros(1, 1, hidden_size)
    losses = []
    for i in range(600):
        start = np.random.randint(3, size=1)[0]
        time_steps = np.linspace(start, start + 10, num_time_steps)
        data = np.sin(time_steps)
        data = data.reshape(num_time_steps, 1)
        x = torch.tensor(data[:-1]).float().view(1, num_time_steps - 1, 1)
        y = torch.tensor(data[1:]).float().view(1, num_time_steps - 1, 1)

        output, hidden_prev = model(x, hidden_prev)
        hidden_prev = hidden_prev.detach()  ## 最后一层隐藏层需要 datach，防止梯度爆炸

        loss = criterion(output, y)
        model.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(f"Iteration: {i} loss {loss.item()}")
            losses.append(loss.item())
    ## 绘制loss
    plt.plot(losses, 'r')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig('RNN_loss.png')
    return hidden_prev, model

hidden_prev, model = train_RNN()

for p in model.parameters():
    print(p.grad.norm())
    torch.nn.utils.clip_grad_norm_(p, 10)


start = np.random.randint(3, size=1)[0]
time_steps = np.linspace(start, start + 10, num_time_steps)
data = np.sin(time_steps)
data = data.reshape(num_time_steps, 1)
x = torch.tensor(data[:-1]).float().view(1, num_time_steps - 1, 1)
y = torch.tensor(data[1:]).float().view(1, num_time_steps - 1, 1)

prediction = []
input = x[:, 0, :]
for _ in range(x.shape[1]):
    input = input.view(1, 1, 1)
    (pred, hidden_prev) = model(input, hidden_prev)
    input = pred
    prediction.append(pred.detach().numpy()[0])

x = x.data.numpy()
y = y.data.numpy()
plt.scatter(time_steps[:-1], x.squeeze(), s=90)
plt.plot(time_steps[:-1], x.squeeze())
plt.scatter(time_steps[1:], prediction)
plt.savefig('RNN_prev.png')
plt.show()

