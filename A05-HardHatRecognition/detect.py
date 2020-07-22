# coding=utf-8
from PyQt5.QtCore import QThread
from darknet import *
from util import *
from PyQt5 import uic
from PyQt5.QtGui import QImage, QPixmap, QMovie
from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox
import cv2
import time
import os


class Window:

    def __init__(self):
        # 通过UI文件中动态创建一个相应的窗口对象并做一些初始化操作
        self.ui = uic.loadUi("ui/hatDetector.ui")
        self.ui.setFixedSize(self.ui.width(), self.ui.height())  # 禁止拉伸窗口大小，因为暂时不能解决运行时改变窗口大小闪退的问题
        self.ui.startButton.clicked.connect(self.detect)  # 将按钮与事件联系起来
        self.ui.stopButton.clicked.connect(self.stop)  # 将按钮与事件联系起来
        self.ui.fileButton.clicked.connect(self.chooseVideoFile)  # 将按钮与事件联系起来
        self.ui.cameraButton.clicked.connect(self.useCamera)  # 将按钮与事件联系起来
        self.ui.imgButton.clicked.connect(self.chooseImg)  # 将按钮与事件联系起来
        self.ui.imgFolderButton.clicked.connect(self.chooseImgFolder)  # 将按钮与事件联系起来
        self.ui.stopButton.setEnabled(False)  # 开始检测前不能点击停止检测按钮
        self.readAndShow("file", "imgs/warn.jpg", self.ui.logoLabel)  # 显示右下角图片
        self.readAndShow("file", "imgs/main.png", self.ui.label)  # 显示主窗口图片
        # 设置一些参数的初始值
        self.batch_size = 1
        self.confidence = 0.5
        self.nms_thresh = 0.6
        self.zoom = 416
        self.videoFile = None  # 选择的视频文件
        self.imgFile = None  # 选择的图片文件
        self.imgFolder = None  # 选择的含有图片的文件夹
        self.imgFiles = []  # 选择文件夹后的图片列表
        self.mode = None
        self.CUDA = torch.cuda.is_available()
        self.num_classes = 2
        self.classes = load_classes("data/hatDetect.names")
        self.cfgFile = "cfg/yolov3.cfg"
        self.weightsFile = "weights/yolov3.weights"
        self.frames = 0  # 已检测的帧的数量
        # 建立神经网络并设置一些初始值
        print("Loading network.....")
        self.model = Darknet(self.cfgFile)
        self.model.load_weights(self.weightsFile)
        print("Network successfully loaded")
        self.model.net_info["height"] = self.zoom
        self.inp_dim = int(self.model.net_info["height"])
        if self.CUDA:
            self.model.cuda()
        self.mainThread = None  # 注意，本类里有一个Qthread线程对象，用于通过按钮停止或开始检测并播放视频的线程

    def readAndShow(self, mode, targetImg, position):
        """
        用于将传入的图片显示在某个label上
        mode分为“file”和”cvImg“，字面意思
        target是传入文件，可以是文件名或opencv读取的图片
        position是要显示的label
        """
        position.setScaledContents(True)  # 拉伸填充
        if mode == "file":
            img = cv2.imread(targetImg)
        elif mode == "cvImg":
            img = targetImg
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = QImage(img.data, img.shape[1], img.shape[0], img.shape[1] * 3, QImage.Format_RGB888)
        pix = QPixmap(img)
        position.setPixmap(pix)

    def argsInit(self):
        """
        初始化参数函数，点击开始检测后会根据界面上用户设置的参数值修改参数，尤其是判断用户选择检测的模式是否合适
        """
        self.confidence = self.ui.conSpinBox.value()
        self.nms_thresh = self.ui.nmsSpinBox.value()
        zoom = self.ui.zoomSpinBox.value()
        self.zoom = zoom
        self.model.net_info["height"] = self.zoom
        self.inp_dim = int(self.model.net_info["height"])
        if self.mode == "video":
            cap = cv2.VideoCapture(self.videoFile)  # VideoCapture()中参数是0则表示打开笔记本的内置摄像头，参数是视频文件路径则打开视频

        if self.videoFile is None and self.imgFile is None and self.imgFolder is None:  # 未选择任何文件
            QMessageBox.information(self.ui, "警告", "未选择文件或文件夹！")
            self.ui.startButton.setEnabled(True)
            return False
        elif self.videoFile is not None and not cap.isOpened():  # 选择了视频 
            QMessageBox.information(self.ui, "警告", "无法打开该视频！")
            self.ui.startButton.setEnabled(True)
            return False
        elif self.imgFile is not None:  # 选择了图片
            img = cv2.imread(self.imgFile)
            if img is None:
                QMessageBox.information(self.ui, "警告", "无法打开该图片！")
                self.ui.startButton.setEnabled(True)
                return False
        elif self.imgFolder is not None:  # 选择了文件夹
            files = os.listdir(self.imgFolder)
            for singleImg in files:
                if singleImg.endswith(('jpg', 'png')):
                    self.imgFiles.append(self.imgFolder + '/' + singleImg)
            if len(self.imgFiles) == 0:
                QMessageBox.information(self.ui, "警告", "文件夹下没有JPG或PNG图片！")
                self.ui.startButton.setEnabled(True)
                return False
        return True

    def chooseVideoFile(self):  # 视频文件按钮对应的函数
        temp = QFileDialog.getOpenFileName(None, "请选择视频文件", '/', "Video File(*.mp4)")
        if temp is not None:
            self.imgFile = None
            self.imgFolder = None
            self.imgFiles = []
            self.videoFile = temp[0]
            self.mode = "video"
        else:
            return
        self.ui.fileLabel.setText('视频文件路径：' + self.videoFile)

    def useCamera(self):  # 实时检测按钮对应的函数
        self.videoFile = 0
        self.imgFile = None
        self.imgFolder = None
        self.imgFiles = []
        self.mode = "video"
        self.ui.fileLabel.setText('使用本机的默认摄像头')

    def chooseImg(self):  # 选择图片文件按钮对应的函数
        temp = QFileDialog.getOpenFileName(None, "请选择图片文件", '/', "PNG File(*.png);;JPG File (*.jpg)")
        if temp is not None:
            self.videoFile = None
            self.imgFolder = None
            self.imgFiles = []
            self.imgFile = temp[0]
            self.mode = "img"
        else:
            return
        self.ui.fileLabel.setText('图片文件路径：' + self.imgFile)

    def chooseImgFolder(self):  # 批量检测（选择含有图片文件夹）按钮对应的函数
        temp = QFileDialog.getExistingDirectory(None, "选取包含多张图片的文件夹", "/")
        if temp is not None:
            self.videoFile = None
            self.imgFile = None
            self.imgFiles = []
            self.imgFolder = temp
            self.mode = "imgFolder"
        else:
            return
        self.ui.fileLabel.setText('图片文件夹路径：' + self.imgFolder)

    def stop(self):  # 停止检测按钮对应的函数
        self.ui.stopButton.setEnabled(False)
        self.ui.startButton.setEnabled(True)
        self.ui.fileButton.setEnabled(True)
        self.ui.cameraButton.setEnabled(True)
        self.ui.imgButton.setEnabled(True)
        self.ui.imgFolderButton.setEnabled(True)
        #self.ui.label.setScaledContents(True)
        self.mainThread.stop()  # 可见实际是停止了检测并播放视频的线程

    def write(self, x, results):
        """
        在图片中根据神经网络计算结果进行方框标注的函数
        x是针对一个检测框，信息是图像索引、4个角坐标、目标置信度得分、最大置信类得分、该类的索引
        results是OpenCV读取的原始图片列表
        """
        c1 = tuple(x[1:3].int())  # 左上角坐标
        c2 = tuple(x[3:5].int())  # 右下角坐标
        img = results  # 找到对应的图片
        cls = int(x[-1])  # 获取类别标号
        if cls == 0:
            color = [0, 255, 0]
        else:
            color = [0, 0, 255]
        cv2.rectangle(img, c1, c2, color, 2)  # 参数分别是图片、左上角坐标、右下角坐标、颜色、线条宽度
        return img

    def detect(self):  # 点击检测按钮后的事件
        self.ui.startButton.setEnabled(False)
        self.ui.fileButton.setEnabled(False)
        self.ui.cameraButton.setEnabled(False)
        self.ui.imgButton.setEnabled(False)
        self.ui.imgFolderButton.setEnabled(False)
        if not self.argsInit():
            self.ui.startButton.setEnabled(True)
            self.ui.fileButton.setEnabled(True)
            self.ui.cameraButton.setEnabled(True)
            self.ui.imgButton.setEnabled(True)
            self.ui.imgFolderButton.setEnabled(True)
            return
        self.frames = 0  # 已检测的帧的数量
        self.mainThread = MainThread(self)  # 实际是创建一个线程并运行
        self.mainThread.start()


class MainThread(QThread):  # 为检测专门开一个线程，这样才能及时停止或在检测时选择之后的检测模式

    def __init__(self, win):
        super(MainThread, self).__init__()
        self.working = False
        self.window = win

    def run(self):
        if self.window.mode == "img":  # 图片检测
            frame = cv2.imread(self.window.imgFile)
            frame = self.detectFrame(frame)
            self.window.readAndShow("cvImg", frame, self.window.ui.label)  # 显示标注后的图片
            self.window.ui.startButton.setEnabled(True)
            self.window.ui.stopButton.setEnabled(False)
            self.window.ui.fileButton.setEnabled(True)
            self.window.ui.cameraButton.setEnabled(True)
            self.window.ui.imgButton.setEnabled(True)
            self.window.ui.imgFolderButton.setEnabled(True)
            return
        elif self.window.mode == "imgFolder":  # 图片文件夹检测
            self.window.ui.startButton.setEnabled(False)
            dir = '/'.join(self.window.imgFolder.split('/')[:-1]) + '/DetectResult'
            if not os.path.exists(dir):  # 如果指定的输出目录不存在，则新建
                try:
                    os.mkdir(dir)
                except:
                    QMessageBox.information(self.window.ui, "错误", "没有获得创建文件夹的权限！")
                    self.window.ui.startButton.setEnabled(True)
                    return
            gif = QMovie('imgs/wait.gif')
            self.window.ui.label.setMovie(gif)
            gif.start()
            for imgName in self.window.imgFiles:
                QApplication.processEvents()
                img = cv2.imread(imgName)
                if img is not None:
                    img = self.detectFrame(img)
                    cv2.imwrite(dir + '/D_' + imgName.split('/')[-1], img)
            self.window.readAndShow("file", "imgs/main.png", self.window.ui.label)  # 显示主窗口图片
            QMessageBox.information(self.window.ui, "通知", "图片批量处理完毕！")
            self.window.ui.startButton.setEnabled(True)
            self.window.ui.fileButton.setEnabled(True)
            self.window.ui.cameraButton.setEnabled(True)
            self.window.ui.imgButton.setEnabled(True)
            self.window.ui.imgFolderButton.setEnabled(True)
            return
        elif self.window.mode == "video":
            self.window.ui.stopButton.setEnabled(True)
            self.working = True
            # 检测阶段
            self.window.model.eval()
            cap = cv2.VideoCapture(self.window.videoFile)  # VideoCapture()中参数是0则表示打开笔记本的内置摄像头，参数是视频文件路径则打开视频
            start = time.time()
            while cap.isOpened() and self.working:
                ret, frame = cap.read()
                # cap.read()按帧读取视频，ret,frame是获cap.read()方法的两个返回值。其中ret是布尔值，如果读取帧是正确的则返回True，
                # 如果文件读取到结尾，它的返回值就为False。frame就是每一帧的图像，是个三维numpy数组
                if ret:  # 读取帧成功
                    # 通过神经网络计算
                    frame = self.detectFrame(frame)
                    self.window.frames += 1
                    self.window.ui.fpsLabel.setText("FPS is {:5.2f}".format(self.window.frames / (time.time() - start)))
                    self.window.readAndShow("cvImg", frame, self.window.ui.label)  # 显示标注后的图片
                else:  # 若读取帧失败则本次循环不操作
                    self.window.ui.startButton.setEnabled(True)
                    self.window.ui.stopButton.setEnabled(False)
                    self.window.ui.fileButton.setEnabled(True)
                    self.window.ui.cameraButton.setEnabled(True)
                    self.window.ui.imgButton.setEnabled(True)
                    self.window.ui.imgFolderButton.setEnabled(True)
                    break
            # 退出后界面图片恢复
            self.window.readAndShow("file", "imgs/main.png", self.window.ui.label)  # 显示主窗口图片
            self.window.ui.fpsLabel.setText("FPS")
            return

    def stop(self):
        self.working = False

    def detectFrame(self, frame):
        """
        对一帧（一张图片）进行处理，返回加入了检测标注方框的图片
        """
        img = prep_image(letterbox_image(frame, self.window.inp_dim), self.window.inp_dim)
        im_dim = frame.shape[1], frame.shape[0]  # frame是[宽, 高, 3]的numpy数组
        im_dim = torch.FloatTensor(im_dim).repeat(1, 2)  # [1, 2]repeat后为[1, 4]，im_dim此时为(高, 宽, 高, 宽)
        if self.window.CUDA:
            im_dim = im_dim.cuda()
            img = img.cuda()
        with torch.no_grad():
            output = self.window.model(img, self.window.CUDA)
        output = write_results(output, self.window.confidence, self.window.num_classes, self.window.nms_thresh)

        if type(output) != int:  # 有目标
            im_dim = im_dim.repeat(output.size(0), 1)
            # im_dim为(高, 宽, 高, 宽)（维度[1, 4]），output维度为[可以接受的预测数, 8]
            # 此操作过后为[可以接受的预测数, 4]
            scaling_factor = torch.min(self.window.inp_dim / im_dim, 1)[0].view(-1, 1)  # 得到[可以接受的预测数, 1]，元素是每个锚框的缩放倍数
            output[:, [1, 3]] -= (self.window.inp_dim - scaling_factor * im_dim[:, 0].view(-1, 1)) / 2  # 左上角x和右下角x
            output[:, [2, 4]] -= (self.window.inp_dim - scaling_factor * im_dim[:, 1].view(-1, 1)) / 2  # 左上角x和右下角x
            # 因为要去掉填充边框，所以预测框顶点位置会改变（inp_dim - 原图宽/高*因子）/2  [n, 2] = [n, 2] - [n, 1]（broadcast）
            output[:, 1:5] /= scaling_factor  # 框的大小也要改变
            for i in range(output.shape[0]):  # 因为有些边界框的可能超出了图像边缘，我们要将其限制在图片范围内
                output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, im_dim[i, 0])
                output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, im_dim[i, 1])
            list(map(lambda x: self.window.write(x, frame), output))
        return frame


if __name__ == "__main__":
    app = QApplication([])
    window = Window()
    window.ui.show()
    app.exec_()
