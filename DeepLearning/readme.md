# Neuron network với pytorch

linear regression + logistic function activation(softmax) = logistic regression

## Dữ liệu:
Dữ liệu số viết tay MNIST:
- 42k dòng dữ liệu
- 28 * 28 = 784 thuộc tính ở dạng số thể hiện giá trị ở pixel
- nhãn đầu ra là số từ 0 ... 9

## Hiện thực:

![alt "Mô hình"](./img/nn.png)

- Xây dựng mô hình Neuron Network 3 hidden layer với ReLU function làm activate
- cross entropy loss function
  
![alt "cross entropy"](./img/cross_entropy.png)

- Tối ưu với Stochastic Gradient Descent