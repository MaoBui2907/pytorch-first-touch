import torch
from logistic_regression import LogisticRegression
from torch.autograd import Variable
import torchvision.transforms as transforms
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# ! Chuẩn bị dữ liệu
# * Đọc dataset
dataset = pd.read_csv("./dataset/train.csv", dtype=np.float32)

# * Tách features (784 chiều) và labels (0...9)
# ? Ở đây chỉ lấy 1000 dòng
labels = dataset.label.values
features = dataset.loc[:1000, dataset.columns != "label"].values/255

# * Tách training set (:800) và test set (800:) và chuyển sang tensor
features_train = torch.from_numpy(features[:800])
features_test = torch.from_numpy(features[800:])
labels_train = torch.from_numpy(features[:800]).type(torch.LongTensor)
labels_test = torch.from_numpy(features[800:]).type(torch.LongTensor)

# * Batch, epoch và iteration
batch_size = 5
iters = 1000
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

# ! Xây dựng neuron network
# * Bộ dữ liệu vào gồm 28 * 28 pixel là thuộct tính
input_dim = 28 * 28
# * Đầu ra là tỉ lệ của mỗi lớp (10 lớp)
output_dim = 10
# * Khởi tạo model
model = LogisticRegression(input_dim, output_dim)
# * Cross entropy loss function
error = nn.CrossEntropyLoss()
# * SGD
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# * Chạy thuật toán với training set để tối ưu model
for epoch in range(epochs):
    for i, (feature, label) in enumerate(train_loader):
        train = Variable(feature.view(-1, 28*28))
        label = Variable(label)
        optimizer.zero_grad()
        predict = model(train)
        print("label", label)
        predict = torch.max(predict, 1)[0]
        label = torch.max(label, 1)[0]
        print("predict", predict)
        print("label", label)
        loss = error(predict, label)
        loss.backward()
        optimizer.step()
# * Đưa ra kết quả dự đoán
    for feature, label in test_loader:
        feature = Variable(feature.view(-1, 28*28))
        predicted = torch.max(model(feature), 1)[0]
        print(predicted)
