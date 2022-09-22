from PyQt5 import QtCore, QtGui, QtWidgets
import sys, os
import pickle
import imageio
import pdb

import matplotlib.pyplot as plt
import numpy as np
import cv2
from demo.predictor import COCODemo
from demo.dqn_system import AODSystem
import time

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
plt.rcParams["font.sans-serif"]=["SimHei"]
plt.rcParams["axes.unicode_minus"]=False
plt.rcParams['figure.figsize'] = (8.0, 2.0)

class MsgBox(QtWidgets.QDialog):
    def __init__(self, UIName, wigth=640, hight=360):
        super().__init__()
        self.initUI(UIName, wigth, hight)
    
    def initUI(self, UIName, wigth, hight):
        self.resize(wigth, hight)
        self.setWindowTitle(UIName)
        self.label = QtWidgets.QLabel(self)

    def running(self, GIFImgDir, wigth=640, hight=360):
        movie = QtGui.QMovie(GIFImgDir)
        movie.setScaledSize(QtCore.QSize(wigth, hight))
        self.label.setMovie(movie)
        movie.start()

class Detector(QtCore.QThread):
    signal_msg = QtCore.pyqtSignal(str)
    signal_pkl = QtCore.pyqtSignal(str)
    ImFolder = None
    ImDetFolder = None

    def __init__(self):
        super(Detector, self).__init__()
        self.ConfigFile = 'configs/fcos/fcos_imprv_R_50_FPN_1x.yaml'
        self.DectorWeights = 'training_dir/fcos_imprv_R_50_FPN_1x_car/model_final.pth'
        self.drl_config_file = 'configs/fcos/fcos_imprv_R_50_FPN_1x_DRL_car.yaml'
        self.drl_weights =  'training_dir/ddqn_car/dqn_weights_final.h5f'
        self.pickle_dir = 'result.pickle'
        self.coco_demo = self.get_coco_demo()
        self.AODSystem = AODSystem(self.drl_config_file, self.DectorWeights, self.drl_weights, self.pickle_dir)
   
    def get_coco_demo(self):
        from fcos_core.config import cfg
        cfg.merge_from_file(self.ConfigFile)
        cfg.MODEL.WEIGHT = self.DectorWeights
        # cfg.freeze()
        
        thresholds_for_classes = [0.5, ] * cfg['MODEL']['ROI_BOX_HEAD']['NUM_CLASSES']
        coco_demo = COCODemo(
            cfg,
            confidence_thresholds_for_classes=thresholds_for_classes,
            min_image_size=800
        )
        return coco_demo

    def run(self):
        self.signal_msg.emit('开始执行第一阶段推理，请耐心等待')
        
        images_dir = self.ImFolder
        demo_im_names = os.listdir(images_dir)

        for im_name in demo_im_names:
            img = cv2.imread(os.path.join(images_dir, im_name))
            if img is None:
                continue
            start_time = time.time()
            composite = self.coco_demo.run_on_opencv_image(img)
            self.signal_msg.emit("正在检测   ---   序号/位置编码：{}   ---   推理时间: {:.2f}s".format(im_name.split('.')[0], time.time() - start_time))
            cv2.imwrite(os.path.join(self.ImDetFolder, im_name), composite)

        self.signal_msg.emit('开始执行第二阶段推理，请耐心等待')
        search_dict = {}
        for i in range(self.AODSystem.env.dataset_length):
            start_time = time.time()
            state = self.AODSystem.env.reset()
            done = False

            I0 = self.AODSystem.env.current_image_name
            B0 = self.AODSystem.env.benefit
            IN = [I0,]
            BN = [B0,]
            QN = []

            while(not done):
                with self.AODSystem.session.as_default():
                    with self.AODSystem.session.graph.as_default():
                        action, q_values = self.AODSystem.dqn.forward_with_q_values(state)
                        state, reward, done, info = self.AODSystem.env.step(action)
                        q_values[q_values<0] = 0
                
                QN.append(q_values)
                if not done:
                    IN.append(self.AODSystem.env.current_image_name)
                    BN.append(self.AODSystem.env.benefit)

            search_dict[I0] = [IN, BN, QN]
            self.signal_msg.emit("正在决策   ---   序号/位置编码：{}   ---   推理时间: {:.2f}s".format(I0, time.time() - start_time))

        with open(self.pickle_dir, "wb") as fp:
            pickle.dump(search_dict, fp, protocol=3)
            self.signal_pkl.emit(self.pickle_dir)

        self.signal_msg.emit('推理完成')

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(496, 284)
        MainWindow.setMinimumSize(QtCore.QSize(496, 284))
        MainWindow.setMaximumSize(QtCore.QSize(496, 284))
        self.textBrowser = QtWidgets.QTextBrowser(MainWindow)
        self.textBrowser.setGeometry(QtCore.QRect(10, 60, 476, 211))
        self.textBrowser.setMinimumSize(QtCore.QSize(476, 211))
        self.textBrowser.setMaximumSize(QtCore.QSize(476, 211))
        self.textBrowser.setObjectName("textBrowser")
        self.widget = QtWidgets.QWidget(MainWindow)
        self.widget.setGeometry(QtCore.QRect(7, 10, 481, 41))
        self.widget.setObjectName("widget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.widget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.DatasetDir = QtWidgets.QPushButton(self.widget)
        self.DatasetDir.setObjectName("DatasetDir")
        self.horizontalLayout.addWidget(self.DatasetDir)
        self.PickleDir = QtWidgets.QPushButton(self.widget)
        self.PickleDir.setObjectName("PickleDir")
        self.horizontalLayout.addWidget(self.PickleDir)
        self.ImageDir = QtWidgets.QPushButton(self.widget)
        self.ImageDir.setObjectName("ImageDir")
        self.horizontalLayout.addWidget(self.ImageDir)
        self.StartRunning = QtWidgets.QPushButton(self.widget)
        self.StartRunning.setObjectName("StartRunning")
        self.horizontalLayout.addWidget(self.StartRunning)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        ################定义相关变量并初始化################
        self.ImFolder = ''
        self.PkDict = {}
        self.ImId = ''
        self.FList = []
        self.ImList = []
        self.QList = []
        self.MainWindow = MainWindow

        self.ImDetFolder = 'demo/result'
        self.pickle_dir = 'result.pickle'
        self.thread = Detector()
        self.thread.ImDetFolder = self.ImDetFolder
        self.nameList = ['停止', '向左', '向右', '向上', '向下', '向后', '向前', '调暗', '调亮']

        if not os.path.exists(self.ImDetFolder):
            os.makedirs(self.ImDetFolder)

        ################button按钮点击事件回调函数################
        self.DatasetDir.clicked.connect(self.DatasetDirBntClicked)
        self.PickleDir.clicked.connect(self.PickleDirBntClicked)
        self.ImageDir.clicked.connect(self.ImageDirBntClicked)
        self.StartRunning.clicked.connect(self.StartRunningBntClicked)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "主动目标检测可视化工具"))
        self.DatasetDir.setText(_translate("MainWindow", "设置图像集地址"))
        self.PickleDir.setText(_translate("MainWindow", "开始执行模型推理"))
        self.ImageDir.setText(_translate("MainWindow", "设置初始位置编码"))
        self.StartRunning.setText(_translate("MainWindow", "开始执行可视化"))

    def DatasetDirBntClicked(self):
        ImFolder = QtWidgets.QFileDialog.getExistingDirectory(None, "选取文件夹", os.getcwd())
        if ImFolder != '':
            self.ImFolder = ImFolder
            self.textBrowser.append('设置成功，图像集地址为%s' % self.ImFolder)
        else:
            self.textBrowser.append('设置失败，请重新选择文件夹')

    def PickleDirBntClicked(self):
        if self.ImFolder == '':
            self.textBrowser.append('请先选择图像集地址')
            return
        if not os.path.exists(self.ImDetFolder):
            os.makedirs(self.ImDetFolder)
        if os.listdir(self.ImDetFolder):
            self.textBrowser.append('已存在检测结果%s，请及时清理再重新运行' % self.ImDetFolder)
            return
        if os.path.exists(self.pickle_dir):
            self.textBrowser.append('已存在推理结果%s，请及时清理再重新运行' % self.pickle_dir)
            return
        self.thread.ImFolder = self.ImFolder
        self.thread.ImDetFolder = self.ImDetFolder
        self.thread.signal_msg.connect(self.DetCallbackMSG)
        self.thread.signal_pkl.connect(self.DetCallbackPKL)
        self.thread.start()

    def DetCallbackMSG(self, msg):
        self.textBrowser.append(str(msg))
        self.textBrowser.repaint()

    def DetCallbackPKL(self, pkl):
        self.textBrowser.append('推理文件地址为%s' % pkl)
        self.textBrowser.repaint()

    def ImageDirBntClicked(self):
        if os.path.exists(self.pickle_dir):
            with open(self.pickle_dir, "rb") as fp:
                self.PkDict = pickle.load(fp)
            self.textBrowser.append('读取推理文件数据成功')

            ImId, ok = QtWidgets.QInputDialog.getText(None, '输入初始位置编码', '初始位置编码：')
            if ok and ImId:
                self.ImId = str(ImId)
                if self.ImId in self.PkDict:
                    self.ImList, self.FList, self.QList = self.PkDict[self.ImId]
                    self.textBrowser.append('设置成功，初始位置编码为%s' % self.ImId)
                else:
                    self.textBrowser.append('设置失败，当前初始位置无效，请重新执行模型推理或重新输入初始位置编码')
            else:
                self.textBrowser.append('设置失败，请重新输入初始位置编码')
        else:
            self.textBrowser.append('设置失败，请检查推理文件%s是否存在' % self.pickle_dir)

    def StartRunningBntClicked(self):
        if not os.path.exists(self.ImDetFolder):
            os.makedirs(self.ImDetFolder)
        if not os.listdir(self.ImDetFolder):
            self.textBrowser.append('未找到检测结果%s，请先执行模型推理' % self.ImDetFolder)
            return
        if not os.path.exists(self.pickle_dir):
            self.textBrowser.append('未找到推理结果%s，请先执行模型推理' % self.pickle_dir)
            return
        if self.ImId == '':
            self.textBrowser.append('请先输入初始位置编码')
            return

        GIFImgDir =  QtWidgets.QFileDialog.getSaveFileName(None, "保存结果", os.getcwd(), "gif files (*.gif)") 
        if GIFImgDir != '':
            gif_images = []
            for path in self.ImList:
                gif_images.append(imageio.imread(self.ImDetFolder + '/%s.jpg' % path))
            imageio.mimsave(GIFImgDir[0], gif_images, fps=1)

            gif_bars = []
            for QL in self.QList:
                fig = plt.figure()
                plt.yticks([])
                ax = plt.gca()
                ax.spines['top'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.spines['right'].set_visible(False)
                plt.bar(range(len(QL)), QL, tick_label=self.nameList)
                fig.canvas.draw()
                data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                gif_bars.append(data)
            GIFBarDir = GIFImgDir[0].split('.gif')[0] + '_bar.gif'
            imageio.mimsave(GIFBarDir, gif_bars, fps=1)
    
            self.MBox1 = MsgBox('模型推理结果', 640, 360)
            self.MBox2 = MsgBox('动作选择柱状图', 640, 160)
            self.MBox1.running(GIFImgDir[0])
            self.MBox2.running(GIFBarDir, hight=160)

            ImList = '->'.join(self.ImList)
            FList = [str(round(f, 3)) for f in self.FList]
            FList = '->'.join(FList)
            self.textBrowser.append('主动跟目标检测结果如下：')
            self.textBrowser.append('位置编码决策顺序为 %s' % ImList)
            self.textBrowser.append('相应F值变化为 %s' % FList)

            self.MBox1.show()
            self.MBox2.show()
            self.MBox1.exec()
            # self.MBox2.exec()
        else:
            self.textBrowser.append('设置失败，请重新选择保存地址')


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(mainWindow)
    mainWindow.show()
    sys.exit(app.exec())