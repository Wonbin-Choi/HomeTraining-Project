# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 09:51:50 2024

@author: USER
"""
from gtts import gTTS
from playsound import playsound
import threading
import cv2 as cv
import numpy as np
from PyQt5.QtWidgets import *
import sys
import cv2
import mediapipe as mp

file_name1 = 'goDown.mp3'
tts_ko = gTTS(text='더 내려가세요',lang='ko')
tts_ko.save(file_name1)
 
file_name2 = 'goUp.mp3'
tts_ko = gTTS(text='이제 올라가세요',lang='ko')
tts_ko.save(file_name2)

def calculate_angle(a,b,c):
    
    # 각 값을 받아 넘파이 배열로 변형
    a = np.array(a) # 첫번째
    b = np.array(b) # 두번째
    c = np.array(c) # 세번째

    # 라디안을 계산하고 실제 각도로 변경한다.
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    # 180도가 넘으면 360에서 뺀 값을 계산한다.
    if angle >180.0:
        angle = 360-angle

    # 각도를 리턴한다.
    return angle

class ExerciseVideo(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('운동 자세 인식기')
        self.setGeometry(200,200,400,100)
        
        startButton=QPushButton('운동 시작',self)
        self.pickCombo=QComboBox(self)     
        self.pickCombo.addItems(['푸쉬업','스쿼트'])
        quitButton=QPushButton('운동 끝',self)        
        
        startButton.setGeometry(10,10,140,30)
        self.pickCombo.setGeometry(150,10,110,30)                  
        quitButton.setGeometry(280,10,100,30)
        
        startButton.clicked.connect(self.startFunction) 
        quitButton.clicked.connect(self.quitFunction)
        
    def play_sound(self,file_name):
        playsound(file_name)

    def startFunction(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose

        self.counter,self.angle1,self.angle2,self.exe_set=0,0,0,0
        self.stage_pushup = 'up'
        self.stage_squart = 'up'
        self.sound_played_pushup = False
        self.sound_played_squart = False             
        self.cap=cv.VideoCapture(0,cv.CAP_DSHOW) 
        # 미디어 파이프 인스턴스 설정(신뢰도는 0.5, 연속 프레임 신뢰도 0.5)
        with self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                
                # 이미지 다시 칠하기: 미디어 파이프에 전달하기 위해 BGR -> RGB로 변경
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                if image is None:
                    print("Error: Unable to load image.")
                # 감지하기
                results = pose.process(image)
                
                # 이미지 도트 쓰기 기능 True로 하고 RGB -> BGR로 색 변경
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                #  운동 종류 콤보 박스
                exercise_type=self.pickCombo.currentIndex()
                
                # 푸쉬업
                if exercise_type==0:
                    count_types = "PushUp"
                    # 랜드마크 추출
                    try:
                        landmarks = results.pose_landmarks.landmark
                        
                        # 어깨, 팔꿈치, 팔목 값 저장
                        left_shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                        left_elbow = [landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                        left_wrist = [landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                        
                        self.angle1 = calculate_angle(left_shoulder, left_elbow, left_wrist)
                        
                        # 계산되 각도를 팔꿈치 위치에 표시
                        cv2.putText(image, str(self.angle1), 
                                       tuple(np.multiply(left_elbow, [640, 480]).astype(int)), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                    except:
                        pass
                    
                    
                    # 랜더링한 이미지를 감지
                    self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                                            self.mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                            self.mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                             )               

                    # 운동횟수
                    if self.stage_pushup == "up":
                        if self.angle1 < 70:
                            cv2.putText(image, 'Go up Now',(300,200), cv2.FONT_HERSHEY_SIMPLEX,
                                               1, (0, 0, 255), 3, cv2.LINE_AA)
                            threading.Thread(target=self.play_sound, args=(file_name2,)).start()
                            self.stage_pushup = "down"
                        if 80 <= self.angle1 < 140:
                            cv2.putText(image, 'Lower your posture',(300,200), cv2.FONT_HERSHEY_SIMPLEX,
                                        1, (0, 0, 255), 3, cv2.LINE_AA)
                            if not self.sound_played_pushup:
                                threading.Thread(target=self.play_sound, args=(file_name1,)).start()
                                self.sound_played_pushup = True
                    elif self.stage_pushup == "down":
                         if self.angle1 > 160:
                             self.stage_pushup = "up"
                             self.sound_played_pushup = False
                             self.counter += 1
                    # 카운트 20회 => 1세트 끝, 1분 동안 쉬는 시간
                    if self.counter == 5:
                        self.counter = 0
                        sec = 60
                        while sec:
                            img1 = cv2.imread('lightWeight.png')
                            cv2.putText(img1, str(sec), (290,180), 
                                               cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 1, cv2.LINE_AA)
                            cv2.imshow('Exercise',img1)
                            if key==ord('q'): break
                            sec -= 1
                        self.exe_set += 1
                    if self.exe_set == 5:
                        break
                    
                # 스쿼트
                elif exercise_type==1:
                    count_types="Squart"
                     
                    try:
                        landmarks=results.pose_landmarks.landmark
                        
                        right_hip   = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y]
                        right_knee  = [landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                        right_ankle = [landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y]  

                        self.angle2 = calculate_angle(right_hip, right_knee, right_ankle)
                        
                        cv2.putText(image, str(self.angle2), 
                                        tuple(np.multiply(right_knee, [640, 480]).astype(int)), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                    except:
                        pass    
                    
                    self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                                            self.mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                            self.mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                             )               

                    # 운동횟수
                    if self.stage_squart == "up":
                        if self.angle2 < 90:
                           cv2.putText(image, 'Go up Now',(300,200), cv2.FONT_HERSHEY_SIMPLEX,
                                              1, (0, 0, 255), 3, cv2.LINE_AA)
                           threading.Thread(target=self.play_sound, args=(file_name2,)).start()
                           self.stage_squart = "down"
                        if 95 <= self.angle2 < 150:
                            cv2.putText(image, 'Lower your posture',(300,200), cv2.FONT_HERSHEY_SIMPLEX,
                                               1, (0, 0, 255), 3, cv2.LINE_AA)
                            if not self.sound_played_squart:
                                threading.Thread(target=self.play_sound, args=(file_name1,)).start()
                                self.sound_played_squart = True
                    elif self.stage_squart == "down":
                        if self.angle2 >160:
                            self.stage_squart = "up"
                            self.sound_played_squart = False
                            self.counter += 1
                    # 카운트 20회 => 1세트 끝, 1분 동안 쉬는 시간
                    if self.counter == 5:
                        self.counter = 0
                        sec = 60
                        while sec:
                            img1 = cv2.imread('lightWeight.png')
                            cv2.putText(img1, str(sec), (290,180), 
                                               cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 1, cv2.LINE_AA)
                            cv2.imshow('Exercise',img1)
                            key = cv2.waitKey(1000)
                            if key==ord('q'): break
                            sec -= 1
                        self.exe_set += 1
                    if self.exe_set == 5:
                        break
                      
                # 운동횟수 카운터 표시
                cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
                cv2.putText(image, count_types, (30,37), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(image, str(self.exe_set+1)+' set', (120,37), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(image, str(self.counter), (180,37), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.imshow('Exercise', image)
                cv2.waitKey(1)
    
                    
    def quitFunction(self):
        self.cap.release()
        cv.destroyAllWindows()        
        self.close()
                
app=QApplication(sys.argv) 
win=ExerciseVideo() 
win.show()
app.exec_()