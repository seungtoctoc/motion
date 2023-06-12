import cv2
import mediapipe as mp
import numpy as np
import time, os
import sys


# parameter (docName, action)
docName = sys.argv[1]
action = sys.argv[2]


# test
# docName = "testDoc2"
# action = "third"
# print("docName and Action : ", docName, action)


# sequence length
seq_length = 90
# recoding time
motion_time = 20


# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)


# timer
created_time = int(time.time())


# make folder 'dataset'
os.makedirs('dataset', exist_ok=True)


camera = cv2.VideoCapture(0)

while camera.isOpened():
    # data array
    data = []
    
    
    # init time
    start_time = time.time()

    # imshow for test
    # ret, img = camera.read()
    # img = cv2.flip(img, 1)
    # cv2.imshow('motion recognition', img)
    # if cv2.waitKey(1) == ord('q'):
    #     break


    while time.time() - start_time < motion_time:
        # read camera
        ret, img = camera.read()
        
        
        # image processing
        img = cv2.flip(img, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        
        # hand motion extraction
        if result.multi_hand_landmarks is not None:
            for res in result.multi_hand_landmarks:
                joint = np.zeros((21, 4))
                for j, lm in enumerate(res.landmark):
                    # x, y, z, visbility data of hand
                    joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

               # compute angles between joints
                v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3] # parent joint
                v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3] # child joint
                v = v2 - v1 # [20, 3]
                
                # normalize v
                v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]
                
                # get angles
                angle = np.arccos(np.einsum('nt,nt->n',
                    v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                    v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]
                angle = np.degrees(angle) # convert radian to degree

                # put label (action name)
                angle_label = np.array([angle], dtype=np.float32)
                angle_label = np.append(angle_label, action)

                # concatenate joint(x, y, z, visibility) -> matrix
                d = np.concatenate([joint.flatten(), angle_label])

                data.append(d)

                # show landmarks
                # mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)
                
        # cv2.imshow('motion recognition', img)
        
    # data -> numpy array
    data = np.array(data)
    
    
    # create sequence data
    full_seq_data = []
    for seq in range(len(data) - seq_length):
        full_seq_data.append(data[seq:seq + seq_length])

    full_seq_data = np.array(full_seq_data)
    # print(full_seq_data.shape)
    
    
    # shape example -> (270, 90, 100) -> (270개의 시퀀스, 각 시퀀스는 90프레임으로 구성, 각 프레임은 100개의 특성 값)
    # 시퀀스 300개 이하일 경우 - 데이터 부족하다고 판단, false 출력
    if full_seq_data.shape[0] <= 300:
        print("false")
        # print("false", docName, action, full_seq_data.shape)
    else:
        print("true")
        # print("true", docName, action, full_seq_data.shape)
    
    
    # save sequence data in dataset folder
    np.save(os.path.join('dataset', f'seq_{docName}_{action}'), full_seq_data)
    
    break



