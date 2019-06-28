import torch
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from neuron import NeuronNetwork

# ! Chuẩn bị dữ liệu
# * Đọc dataset
dataset = pd.read_csv("./dataset/train.csv", dtype=np.float32)

# * Tách features (784 chiều) và labels (0...9)
# ? Ở đây lấy một số dòng tối đa 42000
data_length = 10000
train_length = int(80 * data_length / 100)

labels = dataset.loc[:data_length].label.values
features = dataset.loc[:data_length, dataset.columns != "label"].values/255

# * Tách training set (:800) và test set (800:) và chuyển sang tensor
features_train = torch.from_numpy(features[:train_length])
features_test = torch.from_numpy(features[train_length:])
labels_train = torch.from_numpy(labels[:train_length]).type(torch.LongTensor)
labels_test = torch.from_numpy(labels[train_length:]).type(torch.LongTensor)

# * Batch, epoch và iteration
batch_size = 100
iters = 10000
epochs = int(iters / (len(features_train) / batch_size))

# * Tạo lại bộ dữ liệu với pytorch sẽ tối ưu quá trình tính toán song song
train = torch.utils.data.TensorDataset(features_train, labels_train)
test = torch.utils.data.TensorDataset(features_test, labels_test)

train_loader = torch.utils.data.DataLoader(
    train, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(
    test, batch_size=batch_size, shuffle=False)

# * Hiển thị dữ liệu từ các pixel
# plt.imshow(features[100].reshape(28,28))
# plt.axis("off")
# plt.savefig('graph.png')
# plt.show()

# ! Xây dựng mô hình
# * Bộ dữ liệu vào gồm 28 * 28 pixel là thuộct tính
input_dim = 28 * 28
# * Đầu ra là tỉ lệ của mỗi lớp (10 lớp)
output_dim = 10
# * Số node hidden layer
hidden_dim = 100
# * Khởi tạo model
model = NeuronNetwork(input_dim=input_dim,
                      output_dim=output_dim, 
                      hidden_dim=hidden_dim)
# * Cross entropy loss function
error = nn.CrossEntropyLoss()
# * SGD
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# * Training
i = 0
loss_list = []
for epoch in range(epochs):
  for it, (image, label) in enumerate(train_loader):
    train = Variable(image.view(-1, 28*28))
    label = Variable(label)
    optimizer.zero_grad()
    output = model(train)
    loss = error(output, label)
    # * Tính đạo hàm
    loss.backward()
    optimizer.step()
    # * Lưu số liệu vẽ đồ thị
    loss_list.append(loss.data)
    i += 1
# * Thống kê Loss
plt.plot(np.array(range(i)), loss_list)
plt.savefig('./img/fig.png')
plt.show()    
# ! Test mô hình
