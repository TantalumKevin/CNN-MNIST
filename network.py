# 导入神经网络相关的模块
import torch.nn as nn
import torch.nn.functional as F

# 定义一个Net类，继承自nn.Module类
class Net(nn.Module):
    # 定义初始化方法
    def __init__(self):
        # 调用父类的初始化方法
        super(Net, self).__init__()
        # 定义第一个卷积层，输入通道数为1，输出通道数为6，卷积核大小为5*5，边缘填充为2
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2) 
        # 定义第二个卷积层，输入通道数为6，输出通道数为16，卷积核大小为5*5
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 定义第一个全连接层，输入节点数为16*5*5，输出节点数为120
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # 定义第二个全连接层，输入节点数为120，输出节点数为84
        self.fc2 = nn.Linear(120, 84)
        # 定义第三个全连接层，输入节点数为84，输出节点数为10
        self.fc3 = nn.Linear(84, 10)

    # 定义前向传播方法
    def forward(self, x):
        # 对输入x进行第一个卷积层的计算，并使用ReLU激活函数
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # 对x进行第二个卷积层的计算，并使用ReLU激活函数
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        # 将x变换为一维向量
        x = x.view(-1, 16 * 5 * 5)
        # 对x进行第一个全连接层的计算，并使用ReLU激活函数
        x = F.relu(self.fc1(x))
        # 对x进行第二个全连接层的计算，并使用ReLU激活函数
        x = F.relu(self.fc2(x))
        # 对x进行第三个全连接层的计算，并使用Softmax激活函数
        x = F.softmax(self.fc3(x), dim=1)
        # 返回网络的输出
        return x
