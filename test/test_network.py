from src import mnist_loader
from src import network

# 加载数据
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# 创建神经网络，假设网络结构为 [784, 30, 10]，即输入层784个神经元，隐藏层30个神经元，输出层10个神经元
# net = network.Network([784, 30, 10])

# 使用加载的数据训练网络
# net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

net = network.Network([784, 100, 10])
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)