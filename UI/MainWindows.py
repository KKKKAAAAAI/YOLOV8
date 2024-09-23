import os
from PyQt5.QtWidgets import  QMainWindow
import cv2
from .form.mainwindow import Ui_MainWindow
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import qdarkstyle
import json
from video_process1 import videoprocess
import time


class VideoReader:
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise Exception("Error: Could not open video.")
    
    def get_frame(self):
        ret, frame = self.cap.read()

        # 如果视频读取完毕，重新回到第一帧
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()

        return frame

    def release(self):
        self.cap.release()

class MainWinddow(QMainWindow):
    print_debug_signal = pyqtSignal(str) # 用于触发打印调试信息函数
    show_image_signal = pyqtSignal() # 用于触发图像处理结果函数
    # 初始化函数
    def __init__(self,parent=None):
        super(MainWinddow,self).__init__(parent)
        self.main_ui = Ui_MainWindow()
        self.main_ui.setupUi(self)
        # 设置软件名称
        self.setWindowTitle("AI posture assistant")

        self.Init_menuBar()
        # 设置背景颜色
        palette = qdarkstyle.palette.Palette()
        palette.ID = "dark"
        self.setStyleSheet(qdarkstyle.load_stylesheet(qt_api="pyqt5", palette=palette))
        # 绑定信号和槽
        self.main_ui.pushButton_start.clicked.connect(self.predicte_function) # 开始测试信号
        self.main_ui.pushButton_restart.clicked.connect(self.restart_function) # 重新开始信号
        self.main_ui.pushButton_end.clicked.connect(self.stop_function) # 停止测试信号
        self.main_ui.pushButton_scale.clicked.connect(self.scale_function) # 缩放信号
        self.main_ui.comboBox.currentIndexChanged.connect(self.change_function) # 换动作信号
        self.show_image_signal.connect(self.showimgae) # 显示处理结果信号

        # 教学视频
        self.VideoReader = None 
        
        self.teaching_video_path = {
            "Deadlift": "video/Deadlift.mp4",
            "Lat Pulldown": "video/Lat Pulldown.mp4",
            "Squats": "video/Squats.mp4",
            "Bench Press": "video/Bench Press.mp4",
            "Dumbbel Shoulder Press": "video/Dumbbel Shoulder Press.mp4",
            "Dumbbell Lateral Raises": "video/Dumbbell Lateral Raises.mp4",
            "Shoulder rehabilitation": "video/Shoulder rehabilitation.mp4",
            "Knee rehabilitation": "video/Knee rehabilitation.mp4",
            "Lumbar spine rehabilitation": "video/Lumbar spine rehabilitation.mp4"
        }
        self.action_tips = {
            "Deadlift": "The user faces the camera from the side, with their back straight and feet slightly outward throughout the entire process",
            "Lat Pulldown": "The user faces the camera sideways, with their body upright and back straight, their elbows in front of their shoulders, and the equipment pulled up to their chin position",
            "Squats": "The user faces the camera sideways and is required to keep their back straight throughout the entire process. They should squat with their knees facing outwards until their thighs are parallel to the ground",
            "Bench Press": "The user faces the camera from the side, with the barbell positioned directly above the shoulder and the elbow not overlapping with the shoulder in the lower back position. The elbow should be located below the shoulder",
            "Dumbbel Shoulder Press": "The user should face the camera sideways, with the elbow not overlapping with the shoulder, and the elbow should be located in front of the shoulder",
            "Dumbbell Lateral Raises": "The user faces the camera sideways, with their upper body always leaning forward, arms always bent, and dumbbells always positioned in front of their body",
            "Shoulder rehabilitation": "The user faces the camera directly, with the upper and lower arms bent about 90 degrees throughout the entire process, while the upper arm remains stationary throughout the entire process",
            "Knee rehabilitation": "The user faces the camera sideways, with the training leg placed in front of them. The initial training leg is straight, and then slightly bent",
            "Lumbar spine rehabilitation": "The user faces the camera sideways, with their back straight and not bent"
        }
        self.show_camera = True
        self.camera_id = 0
        # 深度学习模型推理类
        self.videoprocess = videoprocess(self.print_debug_signal,self.show_image_signal)

    def Init_menuBar(self):
        # 创建菜单栏
        menubar = self.menuBar()
        # 创建 FITNESS 菜单
        fitness_menu = menubar.addMenu("FITNESS")
        fitness_actions = [
            "Deadlift", "Lat Pulldown", "Squats", "Bench Press",
            "Dumbbel Shoulder Press", "Dumbbell Lateral Raises"
        ]
        for action_name in fitness_actions:
            action = QAction(action_name, self)
            action.triggered.connect(self.handle_action_triggered)
            fitness_menu.addAction(action)

        # 创建 REHABILITATION 菜单
        rehab_menu = menubar.addMenu("REHABILITATION")
        rehab_actions = [
            "Shoulder rehabilitation", "Knee rehabilitation", "Lumbar spine rehabilitation"
        ]
        for action_name in rehab_actions:
            action = QAction(action_name, self)
            action.triggered.connect(self.handle_action_triggered)
            rehab_menu.addAction(action)

        # 创建 Auto 菜单
        rehab_menu = menubar.addMenu("AUTO")
        rehab_actions = [
            "STGCN"
        ]
        for action_name in rehab_actions:
            action = QAction(action_name, self)
            action.triggered.connect(self.handle_action_triggered)
            rehab_menu.addAction(action)

    def handle_action_triggered(self):
        # 获取发送信号的 action
        action = self.sender()
        action_text = action.text()
        self.main_ui.label_select.setText(action_text)
        if action_text == "STGCN":
            # self.videoprocess.filename = "video/Deadlift1.mp4"
            self.videoprocess.stgcn_enable = True
            self.videoprocess.current_action_name = ""
            if self.VideoReader is not None:
                self.VideoReader.release()
            self.VideoReader = None
            return
        self.main_ui.textBrowser.setText(self.action_tips[action_text])
        self.videoprocess.current_action_name = action_text
        self.videoprocess.stgcn_enable = False
        # self.videoprocess.filename = self.teaching_video_path[action_text]
        
        # 加载教学视频
        if self.VideoReader is not None:
            self.VideoReader.release()
        self.VideoReader = VideoReader(self.teaching_video_path[action_text])
    # 重新开始
    def restart_function(self):
        self.videoprocess.restart()
    # 开始测试
    def predicte_function(self):
        # 启动深度学习推理线程
        self.videoprocess.filename = self.camera_id
        if self.videoprocess.filename == "":
            return
        if self.videoprocess.stopped:
            self.videoprocess.start()

    # 停止测试
    def stop_function(self):
        # 停止深度学习推理线程
        if self.videoprocess.stopped:
            return
        self.videoprocess.stopped = True
        if self.VideoReader is not None:
            self.VideoReader.release()
        self.VideoReader = None 

    def scale_function(self):
        self.show_camera = not self.show_camera
    def change_function(self):
        self.camera_id = self.main_ui.comboBox.currentIndex()

    # 显示界面函数
    def showimgae(self):
        if not self.videoprocess.stopped:
            if self.VideoReader is not None:
                image1 = self.VideoReader.get_frame()
                image2 = self.videoprocess.frame
                if self.show_camera:
                    image = self.overlay_images(image1, image2)
                else:
                    image = self.overlay_images(image2, image1)
            else:
                image = self.videoprocess.frame
                self.main_ui.label_select.setText(self.videoprocess.current_action_name)
            # 转换格式
            frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
            self.main_ui.label_image.setPixmap(QPixmap(QImage(frame.data,frame.shape[1],frame.shape[0],frame.shape[1]*3,QImage.Format_RGB888)))
            self.main_ui.label_image.setScaledContents(True)

            print("Stop counting flag：{} Current movement：{} Movement count：{}".format(self.videoprocess.stop_action_flag, self.videoprocess.current_action_name, self.videoprocess.current_action_num))
  

    def overlay_images(self, image1, image2):
        # 将 image2 重置为 1280x960
        image2_resized = cv2.resize(image2, (1280, 960))

        # 将 image1 重置为 320x240
        image1_resized = cv2.resize(image1, (320, 240))

        # 将 image1 叠加到 image2 的左上角
        # 替换 image2_resized 的左上角区域，位置为(0, 0)到(320, 240)
        image2_resized[0:240, 0:320] = image1_resized
        # 返回合成后的图像
        return image2_resized

    # 注册关闭窗口的回调函数
    def hook_close_win(self, close_fun):
        # 停止深度学习推理线程
        self.videoprocess.stopped = True
        self.close_fun = close_fun