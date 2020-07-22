import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2


def predict_transform(prediction, input_dim, anchors, num_classes, CUDA=True):
    """
    把输出的通道数较多的特征图转换为二维张量，每一行按顺序表示一个预测框
    然后进行sigmoid激活、添加中心偏置、log-space变换处理
    在Darknet类前向传播中的yolo层传播过程被调用
    传入参数分别为网络输出、输入尺寸、锚列表、类别数量、GPU标志
    返回值维度为[batch, box, attrs]
    """
    batch_size = prediction.size(0)
    stride = input_dim // prediction.size(2)  # 相对于原图缩小倍数
    grid_size = input_dim // stride  # 网格长/宽(长和宽相等)
    bbox_attrs = 5 + num_classes  # 一个box的深度
    num_anchors = len(anchors)  # 在yolov3中值规定为3
    # 把输出的通道数较多的特征图转换为二维张量，每一行按顺序表示一个预测框，[batch, box, attrs]
    prediction = prediction.view(batch_size, bbox_attrs * num_anchors, grid_size * grid_size)
    prediction = prediction.transpose(1, 2).contiguous()
    prediction = prediction.view(batch_size, grid_size * grid_size * num_anchors, bbox_attrs)
    anchors = [(a[0] / stride, a[1] / stride) for a in anchors]  # 参数anchors是绝对的像素值长度，需要按特征图相对于原图的缩小比例缩小
    # Sigmoid激活函数用于centre_X, centre_Y和object_confidencce
    prediction[:, :, 0] = torch.sigmoid(prediction[:, :, 0])
    prediction[:, :, 1] = torch.sigmoid(prediction[:, :, 1])
    prediction[:, :, 4] = torch.sigmoid(prediction[:, :, 4])
    # 以下变换注意参考论文
    # 添加中心偏置(注：输出中心坐标已经在前面做了sigmoid，这步只需要再加预测网格的左上角坐标)
    grid = np.arange(grid_size)  # tensor转化为numpy类型
    a, b = np.meshgrid(grid, grid)  # 生成两个grid_size x grid_size的矩阵
    x_offset = torch.FloatTensor(a).view(-1, 1)  # [grid_size x grid_size, 1]
    y_offset = torch.FloatTensor(b).view(-1, 1)  # [grid_size x grid_size, 1]
    if CUDA:
        device = torch.device('cuda')
        x_offset = x_offset.to(device)
        y_offset = y_offset.to(device)
    x_y_offset = torch.cat((x_offset, y_offset), 1)  # 从上向下，从左向右所有网格左上角点的坐标
    x_y_offset = x_y_offset.repeat(1, 3)  # repeat函数沿着特定的维度重复这个张量，在这里是第一个维度重复1次，第二个维度重复3次，[grid_size x grid_size, 2 x 3]
    x_y_offset = x_y_offset.view(-1, 2)  # 改为每个坐标在列上重复3次，[grid_size x grid_size x 3, 2]
    # 这样做的原因是已经把四维张量转换为了三维[batch, 预测框, 预测框中的内容]
    # 而每三个连着的预测框都是一个网格的，所有把每个网格的坐标纵向复制为3个
    x_y_offset = x_y_offset.unsqueeze(0)  # 增加batch维度·[1, grid_size x grid_size x 3, 2]
    prediction[:, :, :2] += x_y_offset  # 给每个预测框的sigmoid后的中心坐标加上偏置，需要注意在batch维度相加是触发了broadcast机制
    # log—space变换预测框大小
    anchors = torch.FloatTensor(anchors)  # [3, 2]
    if CUDA:
        device = torch.device('cuda')
        anchors = anchors.to(device)
    anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0)  # [1, 3 x grid_size x grid_size, 2]
    prediction[:, :, 2:4] = torch.exp(prediction[:, :, 2:4]) * anchors
    # 每张图片的每个预测框都进行 e^预测框宽高 * 输出的宽高 操作，注意，一个网格的3个框大小是不同的
    prediction[:, :, 5: 5 + num_classes] = torch.sigmoid((prediction[:, :, 5: 5 + num_classes]))  # 对分类的概率做sigmoid
    prediction[:, :, :4] *= stride  # 预测框根据特征图相对于输入图的比例进行扩大
    return prediction  # [batch, box, attrs]


def unique(all_class):
    """
    由于同一个类别可以有多个正确的检测，所以我们使用该函数来获得给定图片中所有出现的类
    该函数会在write_results函数中被调用，传入参数维度为[非0行数]（非0指预测有效，没有因为置信度没有达到阈值而被忽略）
    返回值维度为[非0行数-重复的类别数]
    """
    all_class_np = all_class.cpu().numpy()  # tensor -> numpy.array
    unique_class_np = np.unique(all_class_np)
    # 对于一维数组或者列表，unique函数去除其中重复的元素，并按元素由大到小返回一个新的无元素重复的元组或者列表，[非0行数-重复的类别数]
    unique_class_tensor = torch.from_numpy(unique_class_np)  # numpy.array -> tensor
    tensor_res = all_class.new(unique_class_tensor.shape)  # 创建一个大小为unique_class_tensor.shape，数据类型和所在设备与all_class一致的张量
    tensor_res.copy_(unique_class_tensor)  # 存储unique_class_tensor然后返回，[非0行数-重复的类别数]
    return tensor_res


def bbox_iou(box1, box2):  # 第一次传入[1, 7]，[这一类的预测框数-i-1， 7] （这一类的预测框数-i-1即遍历时的剩余行数）
    """
    该函数在write_results中被调用，用来进行非极大值抑制
    第1个传入参数是循环体变量i索引处的边界框（维度[1, 7]），第2个传入参数是多行边界框组成的的一个tensor（维度[这一类的预测框数-i-1， 7]）
    返回值是一个包含了第一个输入的边界框与第二个输入的所有边界框的IOU维度为[剩余行数]的张量，每个值都是对应值与参数1的IOU
    """
    # 得到预测狂的坐标
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]  # 得到4个维度为[1]的张量
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]  # 得到4个维度为[剩余行数]的张量
    # 求出相交矩形的坐标值（若不相交则值会出错，但计算面积时会转换为0）
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # 相交区域的面积
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)
    # torch.clamp将输入input张量每个元素的夹紧到区间 [min,max]，并返回结果到一个新张量
    # 合并区域的面积
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    iou = inter_area / (b1_area + b2_area - inter_area)  # 得到维度为[剩余行数]的tensor，每个值都是对应值与参数1的iou
    return iou


def write_results(prediction, confidence, num_classes, nms_conf=0.4):
    """
    该函数把前向传播得到的tensor达不到阈值的预测和被非极大值抑制的预测剔除
    传入参数分别为prediction([batch, 10647, 5 + classes])、confidence(目标置信度阈)、num_classes（物体类别总数）、nms_conf(NMSIOU的阈值)
    返回值维度为[可以接受的预测数, 8]，这8个值分别为：
    该图片在本批中的序号、左上角x、左上角y、右下角x、右下角y、预测有物体存在的概率、预测为某个类别的概率、该类别序号
    """
    conf_mask = (prediction[:, :, 4] > confidence).float().unsqueeze(2)
    # 切片后[batch, 行数], 对比后还是[batch, 行数]，但行数维的值从实际值变为布尔值
    # [b, 行数， 5+classes] -> [b, 行数, 1]，其中行数这维的值为布尔值，值为该行属性中的目标置信度是否达到阈值
    prediction = prediction * conf_mask
    # [b, 行数， 5+classes] * [b, 行数， 1]
    # 不合格的行元素全变为0(用到了broadcast机制)，得到[b, 行数, 5+classes]，在达到阈值则值不变，没达到阈值则5+classes维元素全为0
    box_corner = prediction.new(prediction.shape)  # 随机初始化一个与prediction维度相同的变量
    # 下面5步为了把prediction中的中心坐标、高度、宽度转化为左上角和右下角的坐标，方便计算IOU
    box_corner[:, :, 0] = (prediction[:, :, 0] - prediction[:, :, 2] / 2)  # 左上角x####################################################################################################################################
    box_corner[:, :, 1] = (prediction[:, :, 1] - prediction[:, :, 3] / 2)  # 左上角y
    box_corner[:, :, 2] = (prediction[:, :, 0] + prediction[:, :, 2] / 2)  # 右下角x
    box_corner[:, :, 3] = (prediction[:, :, 1] + prediction[:, :, 3] / 2)  # 右下角y
    prediction[:, :, :4] = box_corner[:, :, :4]

    batch_size = prediction.size(0)
    write = False  # 还没有初始化输出
    for ind in range(batch_size):  # 对于每张图片
        image_pred = prediction[ind]  # 一张图片对应的张量[行数, 5+classes]
        # confidence threshholding
        max_conf, max_conf_score = torch.max(image_pred[:, 5:5 + num_classes], 1)
        # 返回[行数]（值是每行的最大概率值）, [行数]（值是每行的最大概率对应的下标）
        max_conf = max_conf.float().unsqueeze(1)  # [行数, 1]
        max_conf_score = max_conf_score.float().unsqueeze(1)  # [行数 ,1]
        seq = (image_pred[:, :5], max_conf, max_conf_score)  # 3个张量连接在一起（[行数, 5], [行数, 1], [行数, 1]）
        image_pred = torch.cat(seq, 1)
        # 在第一维度把3个张量拼接，得到[行数， 7]，其中每一行的0-3下标索引预测框两角坐标，4索引预测图中有目标的概率，5索引最大的概率，6索引最大概率对应哪一类
        non_zero_ind = (torch.nonzero(image_pred[:, 4]))  # torch.nonzero返回所有非0元素的索引（原本不合格的行是全0），参数维度为[行数]，返回值为[非0行数, 1]
        try:
            non_zero_ind = non_zero_ind.squeeze()  # [非0行数, 1] -> [非0行数]
            image_pred_ = image_pred[non_zero_ind, :].view(-1, 7)  # 返回值为[非0行数, 该行对应的7个属性]，找到那些预测概率非0的行(其实不需要再view)
        except:
            continue

        # NMS
        if image_pred_.shape[0] == 0:  # 没有行，即没有检测到目标
            continue  # 处理下一张图片
        # 有行的话获取图像中检测到的各个种类
        img_classes = unique(image_pred_[:, -1])  # -1下标索引的是预测的种类标号，传入的是维度为[非0行数]的张量，得到的是[图片内存在的类别数]

        for cls in img_classes:  # 对于一个图片中所有可能存在的物体类别
            # 执行NMS，获得一个特定类的所有预测框
            cls_mask = image_pred_ * (image_pred_[:, -1] == cls).float().unsqueeze(1)
            # 比较后得到维度为[非0行数]的张量，元素类型为布尔，转换为浮点类型后维度扩展为[非0行数, 1]，依靠广播机制相乘
            # 这样不属于该类的行的元素会全部变为0
            # 得到维度为[非0行数, 7]的张量，但这里的“非0行”中一部分又变成了0，因为不属于这一类，“非0”是指预测存在物体的概率没有低于阈值
            class_mask_ind = torch.nonzero(cls_mask[:, -2]).squeeze()  # 得到[这一类的预测框数]
            image_pred_class = image_pred_[class_mask_ind].view(-1, 7)  # 得到[这一类的预测框数, 7]

            # 对检测进行排序，概率最大者在前
            conf_sort_index = torch.sort(image_pred_class[:, 4], descending=True)[1]
            # sort返回二元组，0索引值的排序，1索引排序后的值对应的下标
            image_pred_class = image_pred_class[conf_sort_index]  # 得到[这一类的预测框数, 7]，即维度不变但已经排了序
            idx = image_pred_class.size(0)  # 得到预测的数量

            for i in range(idx):  # 对于特定类的每一个预测
                # 计算IOU
                try:
                    # 循环过程中可能会从image_pred_class中删除一些边界框
                    # 这样一来，迭代可能会出现索引越界触发IndexError
                    # 或者image_pred_class[i+1:]返回一个空张量触发ValueError
                    # 所以加入了异常处理
                    ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i + 1:])
                    # 传入[1, 7]，[这一类的预测框数-i-1，7]，得到维度为[剩余行数]的张量，每个值都是对应值与参数1的iou
                except ValueError:
                    break
                except IndexError:
                    break
                # 如果有任何索引大于i的边界框与第i个边界框的IoU大于阈值nms_conf，那这个边界框就会被删除
                iou_mask = (ious < nms_conf).float().unsqueeze(1)  # 得到维度为[剩余行数, 1]的张量，元素值为0或1
                image_pred_class[i + 1:] *= iou_mask  # 使image_pred_class的剩余行中的某些行全部变为0
                non_zero_ind = torch.nonzero(image_pred_class[:, 4]).squeeze()
                # 得到[剩余非0行数, 1]后squeeze，得到[[剩余非0行数]，值是非0元素下标
                image_pred_class = image_pred_class[non_zero_ind].view(-1, 7)  # 移除元素全为0的行
            # 特定类的每一个预测循环结束
            batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)
            # 最终输出是[可以接受的预测数, 8]，7变为8的原因是加了一列批序号，batch_ind维数是[可以接受的预测数, 1]，其中元素为图片在批中的序号
            seq = batch_ind, image_pred_class
            if not write:  # 如果没有初始化输出
                output = torch.cat(seq, 1)  # [可以接受的预测数, 1]与[可以接受的预测数, 7]连接 -> [可以接受的预测数, 8]
                write = True
            else:
                out = torch.cat(seq, 1)
                output = torch.cat((output, out))  # 默认在第0维连接，即增加行数
        # 每一个预测到的类循环结束
    # 每一个batch的循环结束
    try:  # 检查输出是否已经初始化，如果没有，就意味着这批图像中没有一个检测到，有则正常输出
        return output  # [可以接受的预测数, 8]
    except:
        return 0


def load_classes(namesfile):#---------------------------------------------------------------------------------------------------------------------------------------------------------------
    """
    返回物体类别名称文件中的物体类别名称列表，传入参数为物体类别名称文件名
    """
    fp = open(namesfile, "r")
    names = fp.read().split("\n")[:-1]  # 切片是因为最后一个名字后有新行，所以最后一个元素为''（空串），需要剔除
    return names


def letterbox_image(img, inp_dim):
    """
    使用填充调整图像的大小但保持不变的长宽比（填充用RGB(128,128,128)）
    传入参数为openCV读取的图片，模型要求的输入大小（即长/宽，要求中长宽是相同的）
    """
    img_h, img_w = img.shape[0], img.shape[1]
    new_w = int(img_w * min(inp_dim / img_w, inp_dim / img_h))
    new_h = int(img_h * min(inp_dim / img_w, inp_dim / img_h))
    resized_image = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    # 改变输入维度，interpolation是插值方法，这里采用基于4x4像素邻域的3次插值法
    canvas = np.full((inp_dim, inp_dim, 3), 128)  # [宽, 高, 3]维度的数组，用128填充（这就是要求的输入维度）
    canvas[(inp_dim - new_h) // 2:(inp_dim - new_h) // 2 + new_h,
                (inp_dim - new_w) // 2:(inp_dim - new_w) // 2 + new_w, :] = resized_image
    # 长宽不等的图片按比例缩放后实际并不等于要求的输入维度，要使其维度符合要求需要在边缘补充一些像素，在这里的处理方法是
    # 把长宽不等的图片（numpy数组形式）复制到正确大小的数组上，且位于中心位置，此时边缘就是初始化时用的值位128的像素
    return canvas / 1.0  # 返回值维度为[宽, 高, 3]的numpy数组


def prep_image(img, inp_dim):
    """
    获取用letterbox_image函数处理过大小的的OpenCV图像并将其转换为网络的输入（该函数用来将numpy数组转换为PyTorch的输入格式）
    注：OpenCV以numpy数组的形式加载图像(高x宽x通道)，以BGR作为颜色通道的顺序。PyTorch的图像输入格式为(批量x通道x高x宽)，通道顺序为RGB。
    传入letterbox_image的返回值和要求的输入大小
    返回维度为[1, 3, 高, 宽]的tensor
    """
    img = cv2.resize(img, (inp_dim, inp_dim))  # ??????????????？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？
    img = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)  # 转换为tensor并增加batch张量
    return img  # 返回维度为[1, 3, 高, 宽]的tensor

