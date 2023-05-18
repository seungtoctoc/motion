import cv2
import mediapipe as mp
import numpy as np
import time, os
import sys

camera = cv2.VideoCapture(0)

# parameter (docName, action)
docName = sys.argv[1]
action = sys.argv[2]

# docName = "testDoc2"
# action = "third"


# check parameters
# print("docName and Action : ", docName, action)
# seq_length
seq_length = 150
# recoding time
motion_time = 12


# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# timer
created_time = int(time.time())

# making folder 'dataset'
# os.makedirs('dataset', exist_ok=True)

while camera.isOpened():
    # data array
    data = []
    # waiting 2s
    cv2.waitKey(2000)
    # init time
    start_time = time.time()

    # imshow for test
    # ret, img = camera.read()
    # img = cv2.flip(img, 1)
    # cv2.imshow('img', img)

    while time.time() - start_time < motion_time:
        ret, img = camera.read()
        img = cv2.flip(img, 1)
        # change color
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # put motion to mediapipe
        result = hands.process(img)
        # change color
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # if hand recognition
        if result.multi_hand_landmarks is not None:
            for res in result.multi_hand_landmarks:
                joint = np.zeros((21, 4))
                for j, lm in enumerate(res.landmark):
                    # check for x, y, z, visbility
                    joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

                # angles between joints
                v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3] # Parent joint
                v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3] # Child joint
                v = v2 - v1 # [20, 3]
                
                # Normalize v
                v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]
                
                # Get angle using arcos of dot product
                angle = np.arccos(np.einsum('nt,nt->n',
                    v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                    v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]
                angle = np.degrees(angle) # Convert radian to degree

                # put label, idx = index of actions
                angle_label = np.array([angle], dtype=np.float32)
                angle_label = np.append(angle_label, action)

                # concatenate joint(x, y, z, visibility) -> matrix (100 size)
                d = np.concatenate([joint.flatten(), angle_label])

                data.append(d)

                # show landmarks
                mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)
                
        # cv2.imshow('img', img)
        
    # data -> numpy array
    data = np.array(data)
    
    # print
    # print(docName, action, data.shape)
    
    # save data in dataset folder (numpy)
    # np.save(os.path.join('dataset', f'raw_{docName}_{action}'), data)

    # create sequence data
    full_seq_data = []
    for seq in range(len(data) - seq_length):
        full_seq_data.append(data[seq:seq + seq_length])

    full_seq_data = np.array(full_seq_data)
    # print(full_seq_data.shape)
    
    if full_seq_data.shape[0] <= 200:
        print("false", docName, action, full_seq_data.shape)
    else:
        print("true", docName, action, full_seq_data.shape)
    
    # save data in dataset folder (full_seq_data array)
    np.save(os.path.join('dataset', f'seq_{docName}_{action}'), full_seq_data)
    
    break



