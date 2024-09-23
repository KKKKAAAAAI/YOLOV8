from datetime import datetime
import os
from queue import Queue
import cv2
import time
import torch
import argparse
import numpy as np
from torchvision import transforms

from Track.Tracker import Detection, Tracker
from ActionsEstLoader import TSSTG
from PyQt5.QtCore import pyqtSignal, QThread
from fn import draw_single
from ultralytics import YOLO
# from utils.datasets import letterbox
# from utils.general import non_max_suppression_kpt
# from utils.plots import output_to_keypoint, plot_skeleton_kpts
from pose_utils import normalize_points_with_size, scale_pose
import pygame
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark=False
class AudioPlayerThread(QThread):
    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path
        self.stop_flag = True

    def run(self):
        self.stop_flag = False
        # 初始化 pygame 音乐模块
        pygame.mixer.init()

        # 加载音频文件
        pygame.mixer.music.load(self.file_path)

        # 播放音频文件
        pygame.mixer.music.play()

        # 等待播放完成
        while pygame.mixer.music.get_busy():
            self.msleep(100)  # 休眠避免占用过多资源

        self.stop_flag = True

def actions_predict(frame_action,tracker,action_model,image_size):
    action_predict_list = []
    # 提取track中有用信息
    for track in tracker.tracks:
        if not track.is_confirmed():
            continue
        action_dict = {"track_id":track.track_id, 
                        "bbox":track.to_tlbr().astype(int),
                        "center":track.get_center().astype(int),
                        "time_since_update":track.time_since_update,
                        "keypoints_list_show": track.keypoints_list[-1],
                        "action":'pending'}
        if len(track.keypoints_list) == 30:
            pts = np.array(track.keypoints_list, dtype=np.float32)
            action_dict["keypoints_list"] = pts
            action_predict_list.insert(0,action_dict)
        else:
            action_dict["keypoints_list"] = None
            action_predict_list.append(action_dict)
    # 构建动作识别模型输入量
    action_pts = None
    action_mot = None
    for i,action_dict in enumerate(action_predict_list):
        if action_dict["keypoints_list"] is None:
            continue
        pts = action_dict["keypoints_list"]
        pts[:, :, :2] = normalize_points_with_size(pts[:, :, :2], image_size[0], image_size[1])
        pts[:, :, :2] = scale_pose(pts[:, :, :2])
        pts = np.concatenate((pts, np.expand_dims((pts[:, 1, :] + pts[:, 2, :]) / 2, 1)), axis=1)

        pts = torch.tensor(pts, dtype=torch.float32)
        pts = pts.permute(2, 0, 1)[None, :]
        mot = pts[:, :2, 1:, :] - pts[:, :2, :-1, :]
        if i == 0:
            action_pts = pts
            action_mot = mot
        else:
            action_pts = torch.cat((pts,action_pts),dim=0)
            action_mot = torch.cat((action_mot,mot),dim=0)
    # 动作识别
    if action_pts is not None and action_mot is not None:
        action_result = action_model.predict_new(action_pts,action_mot)
    for i,action_dict in enumerate(action_predict_list):
        clr = (0, 255, 0)
        if action_dict["keypoints_list"] is not None:
            action_name = action_model.class_names[action_result[i].argmax()]
            action_dict["action"] = '{}: {:.2f}%'.format(action_name, action_result[i].max() * 100)
            if action_name == 'Fall Down':
                clr = (255, 0, 0)
            elif action_name == 'Lying Down':
                clr = (255, 200, 0)
       
        # 绘制行为
        if action_dict["time_since_update"] == 0:
            frame_action = cv2.rectangle(frame_action, (action_dict["bbox"][0], action_dict["bbox"][1]), (action_dict["bbox"][2], action_dict["bbox"][3]), (0, 255, 0), 2)
            frame_action = cv2.putText(frame_action, action_dict["action"], (action_dict["bbox"][0], action_dict["bbox"][1]-10), cv2.FONT_HERSHEY_COMPLEX,
                                0.4, clr, 2)
    return frame_action

# 视频处理线程类
class videoprocess(QThread):
    def __init__(self,print_debug_signal,show_image_signal,parent=None):
        super(videoprocess,self).__init__(parent)
        self.print_debug_signal = print_debug_signal # 文本框打印信号
        self.show_image_signal = show_image_signal # 显示处理结果信号
        self.filename = None # 测试视频路径
        self.save_out = False # 是否保存预测视频，影响帧率
        self.save_dir = "save"
        self.stopped = False # 是否停止测试
        self.device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")#判断能否调用GPU# 采用cpu或gpu推理，默认gpu
    
        # 构建姿态估计模型
        weigths = "weights/yolov8x-pose-p6.pt"
        self.pose_model = YOLO(weigths)
        self.pose_model.to(device=self.device)
      
        # 构建行为识别模型，
        self.action_model = TSSTG(device=self.device)
        # 当前判断动作名称
        self.stgcn_enable = False
        self.current_action_name = ""
        self.current_action_num = 0 # 当前动作个数
        self.last_action_name = ""
        self.stop_action_flag = False 
        self.speech = None
        self.action_finish_flag = -1 # 初始化动作完成标志
        self.last_distance = 0
        self.current_distance = 0
    # 线程执行函数
    def run(self):
        # 设置姿态跟踪器，无需修改
        max_age = 30
        self.tracker = Tracker(max_age=max_age, n_init=3)
        # 1.加载视频流文件
        # if self.filename < 0:
        #     return
        self.stream = cv2.VideoCapture(self.filename)
        self.fps = self.stream.get(cv2.CAP_PROP_FPS) # 获取视频帧率，基本无用
        self.frame_size = (int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH)), # 获取图片尺寸
                           int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.stopped = False # 默认情况为False
        self.count_enable_num = 0 # 使能计数器
        self.change_enable_num = 0 # 切换计数器
        self.current_action_flag = True
        self.incorrect_action_flag = True
        self.action_finish_flag = -1 # 初始化动作完成标志
        # 2.开始进行前向预测，self.stopped为真时退出
        with torch.no_grad():
            while not self.stopped:
                # 3.获取图像帧
                ret, frame = self.stream.read()
                if not ret: # 视频处理完毕，自动退出
                    break
                result = self.pose_model.predict(frame)[0]
                self.tracker.predict()
                # 5.开始检测 
                keypoints = result.keypoints
                conf = result.keypoints.conf
                bbox = result.boxes.xyxy.cpu().numpy()

                # 6.分离关键点
                detections = []
                if keypoints is not None and conf is not None and bbox is not None:
                    keypoints = keypoints.xy.cpu().numpy()
                    conf = conf.cpu().numpy()
                    keypoints = np.concatenate([keypoints, conf.reshape(-1,17,1)], axis=2)
                    for index, ps in enumerate(keypoints):
                        box = bbox[index]
                        ps = ps.reshape(17,3)
                        # Cut eyes and ears.
                        ps = np.concatenate([ps[:1, ...], ps[5:, ...]], axis=0)
                        detections.append(Detection(box,
                                                ps,
                                                ps[:,-1].mean()))
                # 9.更新姿态检测结果
                self.tracker.update(detections)
                self.frame = result.plot(draw_kpt=True)
                
                for i, track in enumerate(self.tracker.tracks):
                    if not track.is_confirmed():
                        continue
                    track_id = track.track_id
                    bbox = track.to_tlbr().astype(int)
                    center = track.get_center().astype(int)
                    # 未识别出的动作默认为pending
                    # Use 30 frames time-steps to prediction.
                    if len(track.keypoints_list) == 30 and self.stgcn_enable:
                        pts = np.array(track.keypoints_list, dtype=np.float32)
                        out = self.action_model.predict(pts, frame.shape[:2])
                        self.current_action_name = self.action_model.class_names[out[0].argmax()]
                    if track.time_since_update == 0:
                        print(self.current_action_name, self.last_action_name)
                        if self.current_action_name != self.last_action_name: # 动作发生变化
                            if self.change_enable_num < 3:
                                self.change_enable_num += 1
                            else: # 动作发生变化超过3帧，则认为动作发生变化
                                self.last_action_name = self.current_action_name
                                self.restart()
                        elif self.change_enable_num > 0: # 动作未发生变化
                            self.change_enable_num -= 1
                        # 不需要在进行计数了
                        if self.stop_action_flag or self.current_action_name == "":
                            continue
                        self.execute_exercise(self.current_action_name, self.frame, track.keypoints_list[-1])
                    break

                self.frame = cv2.putText(self.frame, f"{self.current_action_num}", (self.frame.shape[1] - 50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
                # 13.发送本帧处理结果
                self.show_image_signal.emit()
                
            if self.stream.isOpened():
                self.stream.release()
            # 清空显示
            self.stopped = True

    def restart(self):
        self.change_enable_num = 0
        self.current_action_num = 0
        self.current_action_flag = True
        self.stop_action_flag = False
        self.incorrect_action_flag = True
        self.action_finish_flag = -1
        self.last_distance = 0
        self.current_distance = 0

    # 1.提取关键点坐标
    def extrct_keypoints(self, pose, index:list):
        # 前三个为左边关键点坐标、后三个为右边关键点坐标
        keypoint_list = []
        left_scores = 0
        right_scores = 0
        for i, key in enumerate(index):
            score = pose[key,2]
            if i < 3:
                left_scores += score
            else:
                right_scores += score
            # 防止误检测导致关键点定位错误而引起计数错误
            if score < 0.2:
                point = np.array([0, 0])
            else:
                point = np.array([int(pose[key,0]), int(pose[key,1])])
            keypoint_list.append(point)
        return keypoint_list, left_scores, right_scores
    
    # 2.计算中心节点夹角
    def calculate_angle(self, keypoint_list):
        left_angle_deg, right_angle_deg, left_point, right_point = -1, -1, [0, 0], [0, 0]
        keypoint_list = np.array(keypoint_list)
        # 1.计算向量
        if not (np.all(keypoint_list[0] == [0, 0]) or 
            np.all(keypoint_list[1] == [0, 0]) or 
            np.all(keypoint_list[2] == [0, 0])):
            left_upper_vector = keypoint_list[0] - keypoint_list[1]
            left_lower_vector = keypoint_list[2] - keypoint_list[1]
            left_angle = np.arccos(np.around(np.dot(left_upper_vector, left_lower_vector) / (np.linalg.norm(left_upper_vector) * np.linalg.norm(left_lower_vector)),5))
            left_angle_deg = np.degrees(left_angle)
            left_point = keypoint_list[1]
        
        if not (np.all(keypoint_list[3] == [0, 0]) or 
            np.all(keypoint_list[4] == [0, 0]) or 
            np.all(keypoint_list[5] == [0, 0])):
            right_upper_vector = keypoint_list[3] - keypoint_list[4]
            right_lower_vector = keypoint_list[5] - keypoint_list[4]
    
            right_angle = np.arccos(np.around(np.dot(right_upper_vector, right_lower_vector) / (np.linalg.norm(right_upper_vector) * np.linalg.norm(right_lower_vector)),5))
            right_angle_deg = np.degrees(right_angle)
            right_point = keypoint_list[4]
        # 5.返回左、右节点夹角和坐标
        return left_angle_deg, right_angle_deg, left_point, right_point
    
    # 3.可视化角度
    def draw_angle(self, frame, point, point_angle, flag="left", length = 50):
        # 1.引出一条45度的直线
        angle = 45
        length = 50
        start_point = point
        end_point = (
            int(start_point[0] + length * np.cos(np.radians(angle))),
            int(start_point[1] - length * np.sin(np.radians(angle)))
        )
        # 2.采用左标记或者右标记
        if flag == "left":
            end_point = (
            int(start_point[0] + length * np.cos(np.radians(angle))),
            int(start_point[1] - length * np.sin(np.radians(angle)))
        )
        elif flag == "right":
            end_point = (
            int(start_point[0] - length * np.cos(np.radians(angle))),
            int(start_point[1] - length * np.sin(np.radians(angle)))
        )
        # 3.绘制直线
        cv2.line(frame, start_point, end_point, (0, 255, 0), 2)
        # 4.在绘制的直线末端引出一条水平线
        if flag == "left":
            horizontal_line_end_point = (end_point[0] + length, end_point[1])
        elif flag == "right":
            horizontal_line_end_point = (end_point[0] - length, end_point[1])
        cv2.line(frame, end_point, horizontal_line_end_point, (0, 255, 0), 2)
        # 5.在水平线上添加角度标签
        if flag == "left":
            text_position = (end_point[0] , end_point[1] - 10)
        elif flag == "right":
            text_position = (horizontal_line_end_point[0] , horizontal_line_end_point[1] - 10)
        # 6.绘制角度
        # 放置角度
        cv2.putText(frame, '{0:.2f}'.format(point_angle), text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        # 绘制实心圆
        cv2.circle(frame, point, 5, color=(0, 255, 0), thickness=-1)  # -1表示填充
        # 绘制空心圆                                                    
        cv2.circle(frame, point, 10, color=(0, 255, 0), thickness=2)

    def Bench_Press(self, frame, pose):
        keypoint_list, left_scores, right_scores = self.extrct_keypoints(pose, [1, 3, 5, 2, 4, 6])
        left_angle_deg, right_angle_deg, left_point, right_point= self.calculate_angle(keypoint_list)
        print(left_scores, right_scores, left_angle_deg, right_angle_deg, left_point, right_point)
        if left_scores > right_scores and left_angle_deg >= 0:
            angle_deg = left_angle_deg
            point = left_point
            elbow_point = keypoint_list[1]
            shoulder_point = keypoint_list[0]
        elif right_scores > left_scores and right_angle_deg >= 0:
            angle_deg = right_angle_deg
            point = right_point
            elbow_point = keypoint_list[4]
            shoulder_point = keypoint_list[3]
        else:
            return
        # 开始判断是否满足动作要求
        if angle_deg < 120: # 1.判断角度是否小于120度
            if self.count_enable_num < 3:
                    self.count_enable_num += 1
            elif self.current_action_flag:
                self.current_action_flag = False
                self.current_action_num += 1
                self.action_finish_flag += 1 # 开始动作为0
        else:
            if self.count_enable_num > 0:
                self.count_enable_num -= 1
            else:
                # 使能下一次计数
                if self.current_action_flag is False:
                    self.action_finish_flag += 1 # 结束动作为1
                self.current_action_flag = True
        if self.action_finish_flag == 0: # 正在执行动作
            if elbow_point[1] > shoulder_point[1]:
                self.incorrect_action_flag = False
        elif self.action_finish_flag == 1: # 动作结束
            if self.incorrect_action_flag is True: # 动作不正确
                audio_path = "UI/mp3/incorrect.mp3"
            else:
                audio_path = "UI/mp3/correct.mp3"

            self.speech = AudioPlayerThread(audio_path)
            self.speech.start()
            self.incorrect_action_flag = True
            self.action_finish_flag = -1 # 动作完成

        # 绘制角度
        self.draw_angle(frame, point, angle_deg)
        
        if self.current_action_num >= 5:
            self.stop_action_flag = True

    def Dumbbell_Shoulder_Press(self, frame, pose):
        keypoint_list, left_scores, right_scores = self.extrct_keypoints(pose, [1, 3, 5, 2, 4, 6])
        left_angle_deg, right_angle_deg, left_point, right_point= self.calculate_angle(keypoint_list)
        print(left_scores, right_scores, left_angle_deg, right_angle_deg, left_point, right_point)
        if left_scores > right_scores and left_angle_deg >= 0:
            angle_deg = left_angle_deg
            point = left_point
            elbow_point = keypoint_list[1]
            shoulder_point = keypoint_list[0]
        elif right_scores > left_scores and right_angle_deg >= 0:
            angle_deg = right_angle_deg
            point = right_point
            elbow_point = keypoint_list[4]
            shoulder_point = keypoint_list[3]
        else:
            return
        # 开始判断是否满足动作要求
        if angle_deg < 120: # 1.判断角度是否小于120度
            if self.count_enable_num < 3:
                    self.count_enable_num += 1
            elif self.current_action_flag:
                self.current_action_flag = False
                self.current_action_num += 1
                self.action_finish_flag += 1 # 开始动作为0
        else:
            if self.count_enable_num > 0:
                self.count_enable_num -= 1
            else:
                # 使能下一次计数
                if self.current_action_flag is False:
                    self.action_finish_flag += 1 # 结束动作为1
                self.current_action_flag = True
        if self.action_finish_flag == 0: # 正在执行动作
            if abs(elbow_point[0] - shoulder_point[1]) > 15: # 2.判断手腕和肩膀的水平距离是否小于10 ，肩膀与肘不能重复
                self.incorrect_action_flag = False
        elif self.action_finish_flag == 1: # 动作结束
            if self.incorrect_action_flag is True: # 动作不正确
                audio_path = "UI/mp3/incorrect.mp3"
            else:
                audio_path = "UI/mp3/correct.mp3"

            self.speech = AudioPlayerThread(audio_path)
            self.speech.start()
            self.incorrect_action_flag = True
            self.action_finish_flag = -1 # 动作完成

        # 绘制角度
        self.draw_angle(frame, point, angle_deg)
        if self.current_action_num >= 5:
            self.stop_action_flag = True

    def Dumbbell_Lateral_Raises(self, frame, pose):
        keypoint_list, left_scores, right_scores = self.extrct_keypoints(pose, [7, 1, 3, 8, 2, 4])
        left_angle_deg, right_angle_deg, left_point, right_point= self.calculate_angle(keypoint_list)
        print(left_scores, right_scores,left_angle_deg, right_angle_deg, left_point, right_point)
        if left_scores > right_scores and left_angle_deg >= 0:
            angle_deg = left_angle_deg
            point = left_point
            hip_point = keypoint_list[0]
            shoulder_point = keypoint_list[1]
        elif right_scores > left_scores and right_angle_deg >= 0:
            angle_deg = right_angle_deg
            point = right_point
            hip_point = keypoint_list[3]
            shoulder_point = keypoint_list[4]
        else:
            return
        # 开始判断是否满足动作要求
        if angle_deg > 45: # 
            if self.count_enable_num < 5:
                self.count_enable_num += 1
            elif self.current_action_flag:
                self.current_action_flag = False
                self.current_action_num += 1
                self.action_finish_flag += 1
        else:
            if self.count_enable_num > 0:
                self.count_enable_num -= 1
            else:
                # 使能下一次计数
                if self.current_action_flag is False:
                    self.action_finish_flag += 1 # 结束动作为1
                self.current_action_flag = True
        left_upper_vector = shoulder_point - hip_point
        left_lower_vector = (hip_point[0], hip_point[1] - 10) - hip_point
        left_angle = np.arccos(np.around(np.dot(left_upper_vector, left_lower_vector) / (np.linalg.norm(left_upper_vector) * np.linalg.norm(left_lower_vector)),5))
        left_angle_deg = np.degrees(left_angle)
        if abs(left_angle_deg - 0) > 10:  # 身体前倾
            self.incorrect_action_flag = False
        if self.action_finish_flag == 1:
            if self.incorrect_action_flag is True: # 动作不正确
                audio_path = "UI/mp3/incorrect.mp3"
            else:
                audio_path = "UI/mp3/correct.mp3"

            self.speech = AudioPlayerThread(audio_path)
            self.speech.start()
            self.incorrect_action_flag = True
            self.action_finish_flag = -1 # 动作完成
        # 绘制角度
        self.draw_angle(frame, point, angle_deg)
        if self.current_action_num >= 5:
            self.stop_action_flag = True

    def Squats(self, frame, pose):
        keypoint_list, left_scores, right_scores = self.extrct_keypoints(pose, [7, 9, 11, 8, 10, 12])
        left_angle_deg, right_angle_deg, left_point, right_point= self.calculate_angle(keypoint_list)
        print(left_scores, right_scores, left_angle_deg, right_angle_deg, left_point, right_point)
        if left_scores > right_scores and left_angle_deg >= 0:
            angle_deg = left_angle_deg
            point = left_point
            hip_point = keypoint_list[0]
            knee_point = keypoint_list[1]
        elif right_scores > left_scores and right_angle_deg >= 0:
            angle_deg = right_angle_deg
            point = right_point
            hip_point = keypoint_list[3]
            knee_point = keypoint_list[4]
        else:
            return
        # 开始判断是否满足动作要求
        if angle_deg < 90: # 1.判断角度是否小于120度
            if self.count_enable_num < 3:
                    self.count_enable_num += 1
            elif self.current_action_flag:
                self.current_action_flag = False
                self.current_action_num += 1
                self.action_finish_flag += 1 # 开始动作为0
        else:
            if self.count_enable_num > 0:
                self.count_enable_num -= 1
            else:
                # 使能下一次计数
                if self.current_action_flag is False:
                    self.action_finish_flag += 1 # 结束动作为1
                self.current_action_flag = True
        if self.action_finish_flag == 0: # 正在执行动作
            left_upper_vector = hip_point - knee_point
            left_lower_vector = (knee_point[0], knee_point[1] + 10) - knee_point
            left_angle = np.arccos(np.around(np.dot(left_upper_vector, left_lower_vector) / (np.linalg.norm(left_upper_vector) * np.linalg.norm(left_lower_vector)),5))
            left_angle_deg = np.degrees(left_angle)
            if abs(left_angle_deg - 90) < 10: 
                self.incorrect_action_flag = False
        elif self.action_finish_flag == 1: # 动作结束
            if self.incorrect_action_flag is True: # 动作不正确
                audio_path = "UI/mp3/incorrect.mp3"
            else:
                audio_path = "UI/mp3/correct.mp3"
      
            self.speech = AudioPlayerThread(audio_path)
            self.speech.start()
            self.incorrect_action_flag = True
            self.action_finish_flag = -1 # 动作完成

        # 绘制角度
        self.draw_angle(frame, point, angle_deg)
        if self.current_action_num >= 5:
            self.stop_action_flag = True

    def Deadlift(self, frame, pose):
        keypoint_list, left_scores, right_scores = self.extrct_keypoints(pose, [1, 7, 9, 2, 8, 10])
        left_angle_deg, right_angle_deg, left_point, right_point= self.calculate_angle(keypoint_list)
        print(left_angle_deg, right_angle_deg, left_point, right_point)
        if left_scores > right_scores and left_angle_deg >= 0:
            angle_deg = left_angle_deg
            point = left_point
            shoulder_point = keypoint_list[0]
            hip_point = keypoint_list[1]
        elif right_scores > left_scores and right_angle_deg >= 0:
            angle_deg = right_angle_deg
            point = right_point
            shoulder_point = keypoint_list[3]
            hip_point = keypoint_list[4]
        else:
            return
        # 开始判断是否满足动作要求
        if angle_deg > 90: # 1.判断角度是否小于120度
            if self.count_enable_num < 3:
                    self.count_enable_num += 1
            elif self.current_action_flag:
                self.current_action_flag = False
                self.current_action_num += 1
                self.action_finish_flag += 1
        else:
            if self.count_enable_num > 0:
                self.count_enable_num -= 1
            else:
                # 使能下一次计数
                if self.current_action_flag is False:
                    self.action_finish_flag += 1 # 结束动作为1
                self.current_action_flag = True
        self.current_distance = np.linalg.norm(shoulder_point - hip_point)
        if self.last_distance == 0:
            self.last_distance = self.current_distance
        if abs(self.current_distance - self.last_distance) < 30:  #  全程距离几乎保持不变，则认为动作不正确
            self.incorrect_action_flag = False
    
        if self.action_finish_flag == 1:
            if self.incorrect_action_flag is True: # 动作不正确
                audio_path = "UI/mp3/incorrect.mp3"
            else:
                audio_path = "UI/mp3/correct.mp3"

            self.speech = AudioPlayerThread(audio_path)
            self.speech.start()
            self.incorrect_action_flag = True
            self.action_finish_flag = -1 # 动作完成
    
        # 绘制角度
        self.draw_angle(frame, point, angle_deg)
        if self.current_action_num >= 5:
            self.stop_action_flag = True

    def Shoulder_rehabilitation(self, frame, pose):
        keypoint_list, left_scores, right_scores = self.extrct_keypoints(pose, [1, 3, 5, 2, 4, 6])
        left_angle_deg, right_angle_deg, left_point, right_point= self.calculate_angle(keypoint_list)
        left_hight = keypoint_list[0][1] + keypoint_list[1][1] + keypoint_list[2][1]
        right_hight = keypoint_list[3][1] + keypoint_list[4][1] + keypoint_list[5][1]
        if left_hight < right_hight: # 左侧手臂抬起
            angle_deg = left_angle_deg
            point = left_point
            wrist_point = keypoint_list[2]
            shoulder_point = keypoint_list[0]
        elif right_hight < left_hight: # 右侧手臂抬起
            angle_deg = right_angle_deg
            point = right_point
            wrist_point = keypoint_list[5]
            shoulder_point = keypoint_list[3]
        else:
            return
        # 开始判断是否满足动作要求
        if wrist_point[1] < shoulder_point[1]: # 1.手腕高过肩部
            if self.count_enable_num < 3:
                    self.count_enable_num += 1
            elif self.current_action_flag:
                self.current_action_flag = False
                self.current_action_num += 1
                self.action_finish_flag += 1
        else:
            if self.count_enable_num > 0:
                self.count_enable_num -= 1
            else:
                # 使能下一次计数
                if self.current_action_flag is False:
                    self.action_finish_flag += 1 # 结束动作为1
                self.current_action_flag = True
        if abs(angle_deg - 90) < 10: # : # 动作正确
            self.incorrect_action_flag = False
        if self.action_finish_flag == 1:
            if self.incorrect_action_flag is True: # 动作不正确
                audio_path = "UI/mp3/incorrect.mp3"
            else:
                audio_path = "UI/mp3/correct.mp3"

            self.speech = AudioPlayerThread(audio_path)
            self.speech.start()
            self.incorrect_action_flag = True
            self.action_finish_flag = -1 # 动作完成

        # 绘制角度
        self.draw_angle(frame, point, angle_deg)
        if self.current_action_num >= 15:
            self.stop_action_flag = True

    def Lumbar_spine_rehabilitation(self, frame, pose):
        keypoint_list, left_scores, right_scores = self.extrct_keypoints(pose, [7, 9, 11, 8, 10, 12])
        left_angle_deg, right_angle_deg, left_point, right_point= self.calculate_angle(keypoint_list)
        print(left_scores, right_scores, left_angle_deg, right_angle_deg, left_point, right_point)
        # 根据角度来判断那侧腿部抬起
        if left_angle_deg > right_angle_deg: # 左腿抬起
            angle_deg = left_angle_deg
            point = left_point
            ankle_point = keypoint_list[2]
            hip_point = keypoint_list[0]
            shoulder_point = np.array([int(pose[1,0]), int(pose[1,1])])
        elif right_angle_deg > left_angle_deg: # 右腿抬起
            angle_deg = right_angle_deg
            point = right_point
            ankle_point = keypoint_list[5]
            hip_point = keypoint_list[3]
            shoulder_point = np.array([int(pose[2,0]), int(pose[2,1])])
        else: # 两侧腿同时抬起
            return
        # 开始判断是否满足动作要求
        if abs(ankle_point[1] - hip_point[1]) < 80: 
            if self.count_enable_num < 3:
                    self.count_enable_num += 1
            elif self.current_action_flag:
                self.current_action_flag = False
                self.current_action_num += 1
                self.action_finish_flag += 1
        else:
            if self.count_enable_num > 0:
                self.count_enable_num -= 1
            else:
               # 使能下一次计数
                if self.current_action_flag is False:
                    self.action_finish_flag += 1 # 结束动作为1
                self.current_action_flag = True
        self.current_distance = np.linalg.norm(shoulder_point - hip_point)
        if self.last_distance == 0:
            self.last_distance = self.current_distance
        if abs(self.current_distance - self.last_distance) < 30:  #  全程距离几乎保持不变，则认为动作不正确
            self.incorrect_action_flag = False
            print(self.current_distance,self.last_distance,abs(self.current_distance - self.last_distance))
        if self.action_finish_flag == 1:
            if self.incorrect_action_flag is True: # 动作不正确
                audio_path = "UI/mp3/incorrect.mp3"
            else:
                audio_path = "UI/mp3/correct.mp3"

            self.speech = AudioPlayerThread(audio_path)
            self.speech.start()
            self.incorrect_action_flag = True
            self.action_finish_flag = -1 # 动作完成
        # 绘制角度
        self.draw_angle(frame, point, angle_deg)
        if self.current_action_num >= 6:
            self.stop_action_flag = True
    
    def Knee_rehabilitation(self, frame, pose):
        keypoint_list, left_scores, right_scores = self.extrct_keypoints(pose, [7, 9, 11, 8, 10, 12])
        left_angle_deg, right_angle_deg, left_point, right_point= self.calculate_angle(keypoint_list)
        print(left_scores, right_scores, left_angle_deg, right_angle_deg, left_point, right_point)
        if left_scores > right_scores and left_angle_deg >= 0: # 左腿在前
            angle_deg = left_angle_deg
            point = left_point
            athor_angle = right_angle_deg
        elif right_scores > left_scores and right_angle_deg >= 0: # 右腿在前
            angle_deg = right_angle_deg
            point = right_point
            athor_angle = left_angle_deg
        else:
            return
        # 开始判断是否满足动作要求
        if angle_deg < 160: # 1.判断角度是否小于160度
            if self.count_enable_num < 3:
                    self.count_enable_num += 1
            elif self.current_action_flag:
                self.current_action_flag = False
                self.current_action_num += 1
                self.action_finish_flag += 1
        else:
            if self.count_enable_num > 0:
                self.count_enable_num -= 1
            else:
                # 使能下一次计数
                if self.current_action_flag is False:
                    self.action_finish_flag += 1 # 结束动作为1
                self.current_action_flag = True
        if abs(athor_angle-180) < 20:  #  
            self.incorrect_action_flag = False
        if self.action_finish_flag == 1:
            if self.incorrect_action_flag is True: # 动作不正确
                audio_path = "UI/mp3/incorrect.mp3"
            else:
                audio_path = "UI/mp3/correct.mp3"

            self.speech = AudioPlayerThread(audio_path)
            self.speech.start()
            self.incorrect_action_flag = True
            self.action_finish_flag = -1 # 动作完成
        # 绘制角度
        self.draw_angle(frame, point, angle_deg)
        if self.current_action_num >= 6:
            self.stop_action_flag = True
    
        # 绘制角度
        self.draw_angle(frame, point, angle_deg)
        if self.current_action_num >= 6:
            self.stop_action_flag = True

    def Lat_Pulldown(self, frame, pose):
        keypoint_list, left_scores, right_scores = self.extrct_keypoints(pose, [1, 3, 5, 2, 4, 6])
        left_angle_deg, right_angle_deg, left_point, right_point= self.calculate_angle(keypoint_list)
        print(left_angle_deg, right_angle_deg, left_point, right_point)
        if left_scores > right_scores and left_angle_deg >= 0:
            angle_deg = left_angle_deg
            point = left_point
            wrist_point = keypoint_list[2]
            shoulder_point = keypoint_list[0]
        elif right_scores > left_scores and right_angle_deg >= 0:
            angle_deg = right_angle_deg
            point = right_point
            wrist_point = keypoint_list[5]
            shoulder_point = keypoint_list[3]
        else:
            return
        # 开始判断是否满足动作要求
        if angle_deg < 120: # 1.判断角度是否小于120度
            if self.count_enable_num < 3:
                    self.count_enable_num += 1
            elif self.current_action_flag:
                self.current_action_flag = False
                self.current_action_num += 1
                self.action_finish_flag += 1 # 开始动作为0
        else:
            if self.count_enable_num > 0:
                self.count_enable_num -= 1
            else:
                # 使能下一次计数
                if self.current_action_flag is False:
                    self.action_finish_flag += 1 # 结束动作为1
                self.current_action_flag = True
        if self.action_finish_flag == 0: # 正在执行动作
            if abs(wrist_point[1] - shoulder_point[1]) < 30: # 2.判断手腕和肩膀的垂直距离是否小于30
                self.incorrect_action_flag = False
        elif self.action_finish_flag == 1: # 动作结束
            if self.incorrect_action_flag is True: # 动作不正确
                audio_path = "UI/mp3/incorrect.mp3"
            else:
                audio_path = "UI/mp3/correct.mp3"
 
            self.speech = AudioPlayerThread(audio_path)
            self.speech.start()
            self.incorrect_action_flag = True
            self.action_finish_flag = -1 # 动作完成
        # 绘制角度
        self.draw_angle(frame, point, angle_deg)
        if self.current_action_num >= 5:
            self.stop_action_flag = True
    
     # 根据关键字调用对应的函数
    def execute_exercise(self, exercise_name, frame, pose):
        # 动作映射字典
        exercise_map = {
            "Bench Press": self.Bench_Press,
            "Dumbbel Shoulder Press": self.Dumbbell_Shoulder_Press,
            "Dumbbell Lateral Raises": self.Dumbbell_Lateral_Raises,
            "Squats": self.Squats,
            "Deadlift": self.Deadlift,
            "Shoulder rehabilitation": self.Shoulder_rehabilitation,
            "Lumbar spine rehabilitation": self.Lumbar_spine_rehabilitation,
            "Knee rehabilitation": self.Knee_rehabilitation,
            "Lat Pulldown": self.Lat_Pulldown
        }

        # 获取并调用对应的函数
        if exercise_name in exercise_map:
            exercise_map[exercise_name](frame, pose)
        else:
            print(f"Exercise '{exercise_name}' not found.")
    