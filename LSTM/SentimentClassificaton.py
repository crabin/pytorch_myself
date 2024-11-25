# import torch
# from torch import nn, optim
# from torchtext.data.utils import get_tokenizer
# from torchtext.datasets import IMDB
# from torchtext.vocab import build_vocab_from_iterator, GloVe
# from torchtext.data import BucketIterator
#
# device = "cuda" if torch.cuda.is_available() else "cpu"
#
# torch.manual_seed(123)
#
# # 创建分词器
# tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
#
# # 加载IMDB数据集
# train_iter, test_iter = IMDB(split=('train', 'test'))
#
#
# # 构建词汇表
# def yield_tokens(data_iter):
#     for label, line in data_iter:
#         yield tokenizer(line)
#
#
# # 构建词汇表
# vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
# vocab.set_default_index(vocab["<unk>"])
#
# # 使用预训练的GloVe词向量
# vectors = GloVe(name='6B', dim=100)
# vocab.set_vectors(vectors.stoi, vectors.vectors, vectors.dim)
#
# # 处理文本和标签的函数
# text_pipeline = lambda x: vocab(tokenizer(x))
# label_pipeline = lambda x: 1.0 if x == "pos" else 0.0
#
# # 打印数据长度
# print("len of train data", len(train_iter))
# print("len of test data", len(test_iter))
#
# # 打印第15个样本的文本和标签
# for i, (label, line) in enumerate(train_iter):
#     if i == 15:
#         print("Text:", line)
#         print("Label:", label)
#         break
#
# # 创建BucketIterator
# batchsz = 30
# train_iterator, test_iterator = BucketIterator.splits(
#     (train_iter, test_iter),
#     batch_size=batchsz,
#     device=device,
#     sort_within_batch=True,
#     sort_key=lambda x: len(x.text),
# )

# K80 gpu for 12 hours
import torch
from torch import nn, optim
from torchtext.legacy import data, datasets
import numpy as np

print('GPU:', torch.cuda.is_available())

torch.manual_seed(123)

TEXT = data.Field(tokenize='spacy')
LABEL = data.LabelField(dtype=torch.float)
train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)

print('len of train data:', len(train_data))
print('len of test data:', len(test_data))

print(train_data.examples[15].text)
print(train_data.examples[15].label)

# word2vec, glove
TEXT.build_vocab(train_data, max_size=10000, vectors='glove.6B.100d')
LABEL.build_vocab(train_data)

batchsz = 30
device = torch.device('cuda')
train_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, test_data),
    batch_size=batchsz,
    device=device
)


class Net(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(net, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, bidirectional=True, dropout=0.5)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        embedding = self.dropout(self.embedding(x))

        output, (hidden, cell) = self.rnn(embedding)

        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)

        out = self.fc(hidden)
        return out


net = Net(len(TEXT.vocab), 100, 256)

pretrained_embedding = TEXT.vocab.vectors
print('pretrained_embedding:', pretrained_embedding.shape)
net.embedding.weight.data.copy_(pretrained_embedding)
print('embedding layer inited.')

optimizer = optim.Adam(net.parameters(), lr=1e-3)
criteon = nn.BCEWithLogitsLoss().to(device)
net.to(device)


def binary_acc(preds, y):
    preds = torch.round(torch.sigmoid(preds))
    correct = torch.eq(preds, y).float()
    acc = correct.sum() / len(correct)
    return acc


def train(net, iterator, optimizer, criteon):
    avg_acc = []
    net.train()

    for i, batch in enumerate(iterator):
        pred = net(batch.text).squeeze(1)
        loss = criteon(pred, batch.label)
        acc = binary_acc(pred, batch.label).item()
        avg_acc.append(acc)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(i, acc)

    avg_acc = np.array(avg_acc).mean()
    print("avg_acc:", avg_acc)


def eval(net, iterator, criteon):
    avg_acc = []

    net.eval()

    with torch.no_grad():
        for batch in iterator:
            # [b, 1] => [b]
            pred = net(batch.text).squeeze(1)

            #
            loss = criteon(pred, batch.label)

            acc = binary_acc(pred, batch.label).item()
            avg_acc.append(acc)

    avg_acc = np.array(avg_acc).mean()

    print('>>test:', avg_acc)


for epoch in range(10):
    eval(net, test_iterator, criteon)
    train(net, train_iterator, optimizer, criteon)
