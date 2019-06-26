import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from linear_regression import LinearRegression

# ! chuẩn bị dữ liệu
# * x, y là tensor dữ liệu mảng một chiều
x = torch.unsqueeze(torch.tensor([3, 4, 5, 6, 7, 8, 9, 10], dtype=torch.float), 1)
y = torch.unsqueeze(torch.tensor([7.7, 7, 6.5, 6, 5.2, 5, 4.3, 3.8], dtype=torch.float ), 1)

# * Vẽ đồ thị điểm
# plt.scatter(x, y)
# plt.show()

# ! Linear regression
# * Khởi tạo model linear regression thông báo số chiều của đầu vào là 1 và đầu ra là 1 
model = LinearRegression(1, 1)

# * Khởi tạo MSE loss function
mse = nn.MSELoss()
# * Learning rate
alpha = 0.01
# * Tối ưu loss function với Stochastic gradient descent
optimizer = torch.optim.SGD(model.parameters(), lr=alpha)
# * Số lần lặp epoch
iter_number = 5000

# * Chạy thuật toán
for iter in range(iter_number):
    # * Từ giá trị theta của model hiện tại dự đoán giá trị ra
    y_pred = model(x)
    # * Từ giá trị dự đoán và giá trị thực tế tìm ra được loss function hiện tại
    loss = mse(y_pred, y)
    # print("Model:   ", [i for i in model.parameters()])
    print("Loss:    ", loss.data) 
    # print("Pred:    ", y_pred.data)
    # print("Real:    ", y)

    # * Tạo ra đạo hàm gốc là 0, pytorch sẽ cộng dồn đạo hàm trong quá trình back propagation.
    optimizer.zero_grad()
    # * Tính đạo hàm với back propagation
    loss.backward()
    optimizer.step()

