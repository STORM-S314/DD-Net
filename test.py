from models.DDNet_Original import DDNet_Original as DDNet
import torch
from pathlib import Path
import numpy as np
from utils import *
def predict(positions,frame_l,joints_n,joints_d,device,model):
    '''
    判断手势
    '''
    p = zoom(positions, target_l=frame_l,
                     joints_num=joints_n, joints_dim=joints_d)
    M = get_CG(positions,joints_n,frame_l)
    X_0 = np.stack([M])
    X_1 = np.stack([p])
    # pdb.set_trace()
    X_0 = torch.from_numpy(X_0).type('torch.FloatTensor')
    X_1 = torch.from_numpy(X_1).type('torch.FloatTensor')
    pred = -1
    with torch.no_grad():
        pred = model(X_0.to(device),X_1.to(device)).cpu().numpy()
    return np.argmax(pred, axis=1)
def class2str(pred):
    '''
    将类别转变为文字
    '''
    classes = np.array([
        'Grab',
        'Tap',
        'Expand',
        'Pinch',
        'Rotation Clockwise',
        'Rotation Counter Clockwise',
        'Swipe Right',
        'Swipe Left',
        'Swipe Up',
        'Swipe Down',
        'Swipe X',
        'Swipe +',
        'Swipe V',
        'Shake'
    ])
    return classes[pred]
savedir = Path('experiments') / Path('1687414290')
frame_l = 32 #帧长
joint_n = 22 #关节数
joint_d = 3 #关节维度
class_num = 14 #类别数
feat_d = 231 #特征
filters = 64 #滤波器数量
device = torch.device("cpu")
Net = DDNet(frame_l,joint_n,joint_d,feat_d,filters,class_num);
model = Net.to(device)
model.load_state_dict(torch.load(str(savedir/"model.pt")))
model.eval()
#尝试维护100帧的列表进行手势识别
#shape[100,22,3]
positions =np.zeros((100,22,3),dtype='float64')

import cv2
import mediapipe as mp
import time
import pdb
cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture('rtsp://admin:123456@192.168.43.1:8554/live')#使用rtsp推流
mpHands = mp.solutions.hands
hands = mpHands.Hands(model_complexity=0)
mpDraw = mp.solutions.drawing_utils
handLmsStyle = mpDraw.DrawingSpec(color=(0,0,255),thickness=3)
handConStyle = mpDraw.DrawingSpec(color=(0,255,0),thickness=2)
pTime = 0
cTime = 0
while True:
    ret, img = cap.read()
    if ret:
        img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)
        # print(result.multi_hand_landmarks)
        imgHeight = img.shape[0]
        imgWidth = img.shape[1]
        if result.multi_hand_landmarks:
            for handLms in result.multi_hand_landmarks:
                mpDraw.draw_landmarks(img,handLms,mpHands.HAND_CONNECTIONS,handLmsStyle,handConStyle)
                lamp_x = 0
                lamp_y = 0
                lamp_z = 0
                position = []
                for i , lm in enumerate(handLms.landmark):
                    #填充position
                    position.append([
                        lm.x,lm.y,lm.z
                    ])
                    if(i==0 or i==5 or i==9 or i==13 or i==17):
                        lamp_x+=lm.x
                        lamp_y+=lm.y
                        lamp_z+=lm.z
                    # pdb.set_trace()
                    xPos = int(lm.x*imgWidth)
                    yPos = int(lm.y*imgHeight)
                    # print(i,lm.x,lm.y)
                    cv2.putText(img,str(i),(xPos-50,yPos+25),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),2)
                    if i%4==0:
                        cv2.circle(img,(xPos,yPos),5,(255,0,0),cv2.FILLED)
                lamp_x/=5
                lamp_y/=5
                lamp_z/=5
                
                lamp_xPos = int(lamp_x*imgWidth)
                lamp_yPos = int(lamp_y*imgHeight)
                cv2.putText(img,str(21),(lamp_xPos-50,lamp_yPos+25),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),2)
                cv2.circle(img,(lamp_xPos,lamp_yPos),5,(255,0,0),cv2.FILLED)

                position.append([
                    lamp_x,
                    lamp_y,
                    lamp_z
                ])
                position = np.array([position])
                #FIFO
                #插入22个关节点
                positions=np.concatenate([positions,position],axis=0)
                #删除旧的关节点
                positions=np.delete(positions,0,axis=0)
                
                #判断一下手势
                result = predict(positions,frame_l,joint_n,joint_d,device,model)
                cv2.putText(img,f'class:{class2str(result)}',(430,50),cv2.FONT_HERSHEY_COMPLEX,2,(255,200,55),5)
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(img,f'fps:{int(fps)}',(30,50),cv2.FONT_HERSHEY_COMPLEX,2,(255,200,55),5)
        cv2.imshow('img',img)
        if cv2.waitKey(1) == ord('q'):
            break

