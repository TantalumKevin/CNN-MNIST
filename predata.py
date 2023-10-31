# 导入必要的库
import torchvision
import torchvision.datasets.mnist as mnist
import numpy as np, os, torch
from torch.utils.data import TensorDataset

# 定义一个函数，用于下载MNIST数据集，并将其转换为张量格式
def install_data():
    train_data=torchvision.datasets.MNIST(
        root='', # 指定本地保存的文件夹路径，这里为空表示当前文件夹
        train=True, # 指定是否下载训练集
        transform=torchvision.transforms.ToTensor(), # 指定是否将图像转换为张量
        download=True # 指定是否从网上下载数据集
    )
    test_data=torchvision.datasets.MNIST(
        root='', # 同上
        train=False, # 指定是否下载测试集
        transform=torchvision.transforms.ToTensor(), # 同上
        download=True # 同上
    )

# 定义一个函数，用于对MNIST数据集进行归一化处理，并将其保存为npz格式的文件
def norm_data():
    root = "./MNIST" # 指定本地保存的文件夹路径
    raw = root + "/raw" # 指定原始数据集的子文件夹路径
    norm = root + "/norm.npz" # 指定归一化后的数据集的文件名

    # 从原始数据集中读取训练集和测试集的图像和标签
    train_set = (
        mnist.read_image_file(os.path.join(raw, 'train-images-idx3-ubyte')),
        mnist.read_label_file(os.path.join(raw, 'train-labels-idx1-ubyte'))
            )
    test_set = (
        mnist.read_image_file(os.path.join(raw, 't10k-images-idx3-ubyte')),
        mnist.read_label_file(os.path.join(raw, 't10k-labels-idx1-ubyte'))
            )
    
    # 将图像和标签转换为numpy数组格式
    train_data = train_set[0].numpy()
    train_label = train_set[1].numpy()
    test_data = test_set[0].numpy()
    test_label = test_set[1].numpy()

    # 计算训练集和测试集的图像的最大值和最小值，用于归一化
    train_data_max = np.expand_dims(np.expand_dims(np.max(np.max(train_data,axis = 1),axis = 1),axis=1),axis=1)
    test_data_max = np.expand_dims(np.expand_dims(np.max(np.max(test_data,axis = 1),axis = 1),axis=1),axis=1)
    train_data_min = np.expand_dims(np.expand_dims(np.min(np.min(train_data,axis = 1),axis = 1),axis=1),axis=1)
    test_data_min = np.expand_dims(np.expand_dims(np.min(np.min(test_data,axis = 1),axis = 1),axis=1),axis=1)

    # 将训练集和测试集的图像进行归一化，即将每个像素值缩放到[0, 1]的范围内
    train_data = (train_data - train_data_min) / (train_data_max - train_data_min)
    test_data = (test_data - test_data_min) / (test_data_max - test_data_min)

    # 将归一化后的数据集保存为npz格式的文件，方便后续加载
    np.savez(norm,train_data=train_data,train_label=train_label,test_data=test_data,test_label=test_label)

# 定义一个函数，用于从npz文件中加载归一化后的数据集，并将其转换为torch.Tensor类型的张量，并且创建数据加载器对象
def load_data(batch_size = 64):
    data = np.load("./MNIST/norm.npz") # 加载npz文件
    train_data = data["train_data"] # 获取训练集的图像
    train_label = data["train_label"] # 获取训练集的标签
    test_data = data["test_data"] # 获取测试集的图像
    test_label = data["test_label"] # 获取测试集的标签

    # 将numpy数组转换为torch.Tensor类型的张量，并且指定数据类型为float和long
    train_data = torch.tensor(train_data).float()
    train_label = torch.tensor(train_label).long()
    test_data = torch.tensor(test_data).float()
    test_label = torch.tensor(test_label).long()

    # 使用torch.utils.data.TensorDataset类来创建训练集和测试集的数据集对象，每个样本都是一个图像和标签的元组
    trainset = TensorDataset(train_data,train_label)
    testset = TensorDataset(test_data, test_label)
    
    # 判断是否有可用的GPU设备，如果有，就设置pin_memory参数为True，这样可以加快从CPU到GPU的数据传输速度
    if torch.cuda.is_available():
        pin_memory = True
    else:
        pin_memory = False
    
    # 使用torch.utils.data.DataLoader类来创建训练集和测试集的数据加载器对象，指定批次大小、是否打乱顺序、是否使用锁页内存等参数
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=False,pin_memory=pin_memory)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,shuffle=False,pin_memory=pin_memory)

    # 返回数据加载器对象，方便后续批量获取数据
    return trainloader, testloader

# 定义一个函数，用于调用前面定义的三个函数，完成数据集的下载、预处理和加载，并且判断是否需要重新下载或预处理数据
def predata():
    install_data() # 调用install_data()函数，下载MNIST数据集
    if not os.path.exists("./MNIST/norm.npz"): # 判断是否已经存在归一化后的数据集文件，如果不存在，就进行归一化处理
        norm_data() # 调用norm_data()函数，对MNIST数据集进行归一化处理
    else:
        print("Data file has existed.") # 如果已经存在归一化后的数据集文件，就打印提示信息
        
    return load_data() # 调用load_data()函数，加载归一化后的数据集，并返回数据加载器对象
    
# 使用if __name__ == "__main__"语句来判断是否以主模块运行程序，并且调用predata()函数来执行数据集的准备工作
if __name__ == "__main__":
    predata()