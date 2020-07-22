import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from util import *


def parse_cfg(cfgfile):
    """
    将网络结构配置文件(yolov3.cfg)解析，传入参数为配置文件名，返回一个字典的列表，每个字典存储一个层的信息
    特别地，其中第一个字典存储网络的基本信息
    """
    file = open(cfgfile, 'r')
    lines = file.read().split('\n')  # 把cfg配置文件中的所有行连成列表
    lines = [x for x in lines if len(x) > 0]  # 跳过空行(代码是python的列表推导式)
    lines = [x for x in lines if x[0] != '#']  # 跳过注释行
    lines = [x.rstrip().lstrip() for x in lines]  # 跳过有内容的行内容两端的空白
    block = {}  # 存储一层网络类型及各个参数的字典
    blocks = []  # 上述字典连成列表
    for line in lines:
        if line[0] == "[":  # ‘[’标志着一个新层的开始
            if len(block) != 0:  # 如果block不为空，说明应该存储这个block中的信息了
                blocks.append(block)  # 把block中信息存入blocks列表
                block = {}  # 重新初始化block为空
            block["type"] = line[1:-1].rstrip()  # 存储层的类型
        else:
            key, value = line.split("=")  # 存储当前行提供的层的属性
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)  # 存储最后一层的内容，因为不会再有下一个'['了
    return blocks  # 返回存储所有层类型和属性的列表，列表中元素是字典


class EmptyLayer(nn.Module):
    """
    程序中shortcut层和route层都不是在构建网络时搭建的，因为这样网络结构会很复杂
    所以用空层代替，而在前向传播中做连接和相加等操作
    """
    def __init__(self):
        super(EmptyLayer, self).__init__()


class DetectionLayer(nn.Module):
    """
    yolo层的网络，只有变量锚框（元素是二元组的长度为3的列表）
    表示三种yolo层输出尺寸之一所使用的三个大小的锚框
    """
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors  # 元素是二元组的长度为3的列表


def create_modules(blocks):
    """
    通过配置文件的信息来搭建神经网络
    传入参数是函数parse_cfg解析配置文件的结果，即parse_cfg函数返回的blocks列表
    返回一个二元组，包括网络基本信息字典和module列表(nn.ModuleList类型)
    """
    net_info = blocks[0]  # 保存网络的总体上的一些参数
    module_list = nn.ModuleList()  # 创建一个Module列表
    prev_filters = 3  # 输入数据的通道数，输入为RGB形式，所以为3
    output_filters = []  # 记录每一层输出的通道数
    for index, x in enumerate(blocks[1:]):  # 从第1层开始遍历
        module = nn.Sequential()  # 一个module是多个层的顺序序列

        # 判断块的类型并为之建立一个module，然后加入module列表中
        if x["type"] == "convolutional":  # 第一种情况，卷积层
            activation = x["activation"]
            try:
                batch_normalize = int(x["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True
            filters = int(x["filters"])
            padding = int(x["pad"])  # pad只表示有没有，不表示值，值是自己计算的
            kernel_size = int(x["size"])
            stride = int(x["stride"])
            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=bias)  # 创建卷积层
            module.add_module("conv_{0}".format(index), conv)  # 把卷积层添加到module中
            # 添加BatchNorm层
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)  # fileters是BatchNorm相邻的前一层卷积层的输出通道数
                module.add_module("batch_norm_{0}".format(index), bn)  # 把BatchNorm层添加到module中
            # 添加激活函数层（激活函数的属性值若为linear则代表什么也不做）
            if activation == "leaky":
                activn = nn.LeakyReLU(0.1, inplace=True)
                module.add_module("leaky_{0}".format(index), activn)

        elif x["type"] == "upsample":  # 第二种情况，上采样层
            stride = int(x["stride"])
            upsample = nn.Upsample(scale_factor=stride, mode="nearest")
            module.add_module("upsample_{}".format(index), upsample)

        elif x["type"] == "route":  # 第三种情况，route层（只有一个参数，可能有1或2个或4个值）
            x["layers"] = x["layers"].split(',')  # 注：连全局信息都改变了
            first = int(x["layers"][0])  # 第一个值
            try:
                second = int(x["layers"][1])  # 可能存在的第二个值
            except:
                second = 0
            try:
                third = int(x["layers"][2])  # 可能存在的第三个值
            except:
                third = 0
            try:
                fourth = int(x["layers"][3])  # 可能存在的第四个值
            except:
                fourth = 0 
            # 若参数用正数表示则会变为负数处理
            if first > 0:  # 若层标号用正数表示则也映射为负数
                first = first - index
            if second > 0:
                second = second - index
            if third > 0:
                third = third - index
            if fourth > 0:
                fourth = fourth - index

            route = EmptyLayer()  # 先建立一个新层并加入model序列（前向传播时连接）
            module.add_module("route_{0}".format(index), route)
            if fourth < 0: # 有4层的情况
                filters = output_filters[index + first] + output_filters[index + second] + \
                          output_filters[index + third] + output_filters[index + fourth]  # 通道维度连接
            else:  # 只可能fourth等于0，但first和second不确定
                 if second < 0:
                    filters = output_filters[index + first] + output_filters[index + second]  # 通道维度连接
                 else:  # 只可能second等于0
                    filters = output_filters[index + first]  # 仅输出start的索引的层的输出
            # 注：只计算并保存了输出的filter数，但并不真的做张量连接(为了更简洁，在前向传播中连接)

        elif x["type"] == "shortcut":  # 第四种情况，shortcut层（参数全部是-3和linear，说明不做激活处理）
            shortcut = EmptyLayer()  # 仅是新建一个空层，并不真的做张量加法(为了更简洁，在前向传播中连接)
            module.add_module("shortcut_{}".format(index), shortcut)

        elif x["type"] == "maxpool":  # 第五种情况，maxpool层，池化但并不丢弃特征，因为池化前的特征也将被保留
            stride = int(x["stride"])
            size = int(x["size"])
            maxpool = nn.MaxPool2d(size, stride=stride, padding=size // 2)
            module.add_module("maxpool_{}".format(index), maxpool)

        elif x["type"] == "yolo":  # 第五种情况，yolo层
            mask = x["mask"].split(",")  # mask表示当前属于第几个预选框(锚点)
            mask = [int(x) for x in mask]
            anchors = x["anchors"].split(",")  # 使用了根据掩码标签的属性建立索引的锚点，预选框， 将样本通过k-means算法计算出来的值
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]
            detection = DetectionLayer(anchors)  # 传入包含3个二元组的列表
            module.add_module("Detection_{}".format(index), detection)

        module_list.append(module)  # 把模块添加到nn.ModuleList()
        prev_filters = filters  # 只有卷积层和route层改变fileters的值，其他层用上上一次的值即可
        output_filters.append(filters)  # 记录每层的输出通道数

    return net_info, module_list  # 返回一个二元组，包括网络基本信息字典和模块列表(nn.ModuleList类型)


class Darknet(nn.Module):
    """
    Darknet53神经网络，初始化需要参数cfgfile，即配置文件名
    初始化时调用parse_cfg函数解析配置文件，然后调用create_modules函数对解析后的返回值搭建实际的模型
    net_info是保存网络基本信息的字典，module_list是模块列表(nn.ModuleList类型)
    """
    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)  # 模块信息列表
        self.net_info, self.module_list = create_modules(self.blocks)  # 网络基本信息字典和模块列表(nn.ModuleList类型)

    def forward(self, x, CUDA):
        """
        前向传播过程
        """
        needToSave = [1, 3, 5, 7, 8, 10, 12, 14, 15, 17, 18, 20, 21, 23, 24, 26, 27, 29, 30, 32, 33, 35, 36, 37, 39, 40,
                       42, 43, 45, 46, 48, 49, 51, 52, 54, 55, 57, 58, 60, 61, 62, 64, 65, 67, 68, 70, 71, 73, 79, 85, 91, 97]
        modules = self.blocks[1:]  # 模块信息列表
        outputs = {}  # 存储前面各个层的输出，因为route和shortcut层会用到前面某些层的输出
        write = 0  # 收集器未初始化（遇到第一个yolo层才会初始化）
        for i, module in enumerate(modules):
            module_type = (module["type"])

            if module_type == "convolutional" or module_type == "upsample":  # 卷积层和上采样层的处理
                x = self.module_list[i](x)
            elif module_type == "route":  # route层的处理
                # 注：建立模块时的计算仅是计算输出的通道数，而这里是真正计算输出的值
                layers = module["layers"]
                layers = [int(a) for a in layers]
                for j in range(len(layers)):
                    if (layers[j]) > 0:  # start为正时映射为负
                        layers[j] = layers[j] - i
                if len(layers) == 1:  # 只有一个参数时，输出参数所指层的输出
                    x = outputs[i + (layers[0])]
                elif len(layers) == 2:  # 有两个参数时，把两个参数分别所指的两个层在通道维度连接起来
                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]
                    x = torch.cat((map1, map2), 1)
                else:  # 有四个参数时
                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]
                    map3 = outputs[i + layers[2]]
                    map4 = outputs[i + layers[3]]
                    x = torch.cat((map1, map2, map3, map4), 1)
            elif module_type == "shortcut":  # shortcut层的处理
                from_ = int(module["from"])
                x = outputs[i - 1] + outputs[i + from_]  # 把from参数所指的层和前一层做element-wise相加
            elif module_type == "maxpool":
                x = self.module_list[i](x)
            elif module_type == 'yolo':  # yolo层的处理
                anchors = self.module_list[i][0].anchors  # 把DetectionLayer对象的参数,赋值给变量anchors
                # （[i]是序列，[i][0]才是etectionLayer对象）
                inp_dim = int(self.net_info["height"])  # 获取网络基本信息中的高度
                num_classes = int(module["classes"])  # 获取网络基本信息中的种类数
                # 变换
                x = x.data  # 只取张量数值部分信息
                x = predict_transform(x, inp_dim, anchors, num_classes, CUDA)
                # 返回值维度为[batch, box, attrs]而不是四维的形式，所以可以相加
                if not write:  # 如果收集器没有被初始化
                    detections = x
                    write = 1
                else:
                    detections = torch.cat((detections, x), 1)
                    # 三个相加结果即纵向每张图片的box变多了[batch, 13 * 13 * 3 + 23 * 26 * 3 + 52 * 52 * 3, 5 + classes]
                    # [batch, 10647, 5 + classes] (因为yolo3中3次预测的大小是确定的)
            if i in needToSave:
                outputs[i] = x
            else:
                outputs[i] = 0
        return detections

    def load_weights(self, weightfile):
        """
        加载训练好的参数，传入权重文件yolov3.weights
        """
        fp = open(weightfile, "rb")
        # The first 5 values are header information
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number
        # 4,5. Images seen by the network (during training)
        header = np.fromfile(fp, dtype=np.int32, count=5)

        self.header = torch.from_numpy(header)
        self.seen = self.header[3]

        weights = np.fromfile(fp, dtype=np.float32)

        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]["type"]

            # If module_type is convolutional load weights
            # Otherwise ignore.

            if module_type == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i + 1]["batch_normalize"])
                except:
                    batch_normalize = 0

                conv = model[0]

                if (batch_normalize):
                    bn = model[1]

                    # Get the number of weights of Batch Norm Layer
                    num_bn_biases = bn.bias.numel()

                    # Load the weights
                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    # Cast the loaded weights into dims of model weights.
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    # Copy the data to model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)

                else:
                    # Number of biases
                    num_biases = conv.bias.numel()

                    # Load the weights
                    conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases

                    # reshape the loaded weights according to the dims of the model weights
                    conv_biases = conv_biases.view_as(conv.bias.data)

                    # Finally copy the data
                    conv.bias.data.copy_(conv_biases)

                # Let us load the weights for the Convolutional layers
                num_weights = conv.weight.numel()

                # Do the same as above for weights
                conv_weights = torch.from_numpy(weights[ptr:ptr + num_weights])
                ptr = ptr + num_weights

                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)

