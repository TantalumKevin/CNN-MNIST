# 导入必要的库
import torch, network, predata, numpy as np
from matplotlib import pyplot as plt

# 定义10个类别的标签
classes = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")

# 导入网络结构并实例化一个对象
net = network.Net().float()

# 导入数据预处理函数并返回两个数据加载器对象
trainloader, testloader = predata.predata()

# 定义损失函数为交叉熵损失函数
criterion = torch.nn.CrossEntropyLoss()

# 定义优化器为随机梯度下降优化器，并设置学习率和动量
optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

# 定义训练轮数为30
epochs = 30

# 判断是否有可用的GPU设备
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

# 将网络转移到设备上
net = net.to(device)
loss_list = np.zeros([epochs,])
# 训练网络
for epoch in range(epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs = torch.unsqueeze(inputs, 1)
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 200 == 199:
            print(f"\rEpoch:{epoch+1}, iteration:{i+1}, loss: {running_loss/200:.3f}",end="")
            running_loss = 0.0
    loss_list[epoch] = loss.item()

plt.plot(loss_list)
print("Finished Training")

# 测试网络
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        inputs, labels = data
        inputs = torch.unsqueeze(inputs, 1)
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy of the network on the 10000 test images: {100*correct/total}")

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        inputs, labels = data
        inputs = torch.unsqueeze(inputs, 1)
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = net(inputs)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(labels.size()[0]):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(10):
    print(f"Accuracy of {classes[i]} : {100 * class_correct[i] / class_total[i]}" )
