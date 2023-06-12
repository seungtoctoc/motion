import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import sys

# parameter (docName, action)
docName = sys.argv[1]
target_action = sys.argv[2]


# docName = "final2"
# target_action = "test"


# setting value
workTime = 30
seq_length = 90
judgment_confidence = 0.97


# more than threshold in windowSize -> did that action
action_window_size = 30 
action_threshold = 25  


actions = ['first', 'second', 'third']


model = load_model(f'models/model_{docName}.h5')


# MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)


cap = cv2.VideoCapture(0)


seq = []
predict_action = []
start_time = cv2.getTickCount()


while cap.isOpened():
    # read camera
    ret, img = cap.read()
    
    
    # image processing
    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    
    # init this_action
    this_action = "none"
    
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


            # concatenate joint(x, y, z, visibility) -> matrix
            d = np.concatenate([joint.flatten(), angle])

            seq.append(d)


            # show landmarks
            # mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)


            if len(seq) < seq_length:
                continue

            input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)

            # 결과 예측 y_pred에 저장
            y_pred = model.predict(input_data).squeeze()

            # 인덱스 뽑아냄
            i_pred = int(np.argmax(y_pred))
            
            # confidence 뽑아냄
            conf = y_pred[i_pred]


            # 확실하지 않다고 판단 -> continue
            if conf < judgment_confidence:
                continue


            # judgment_confidence 이상 -> predict_action 에 저장
            action = actions[i_pred]
            predict_action.append(action)


            if len(predict_action) < action_window_size:
                continue


            # judgement
            window_actions = predict_action[-action_window_size:]
            action_counts = {a: window_actions.count(a) for a in actions}
            max_count = max(action_counts.values())
            dominant_action = max(action_counts, key=action_counts.get)

            if max_count >= action_threshold:
                this_action = dominant_action
            else:
                this_action = '?'
            
            
            if this_action == target_action:
                #print("did action", this_action)
                break
            
                
            # cv2.putText(img, f'{this_action.upper()}', org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)


    # cv2.imshow('motion recognition', img)
    # if cv2.waitKey(1) == ord('q'):
    #     break
    
    
    # did action
    if this_action == target_action:
        print("true", this_action)
        break
    
    
    # time out
    elapsed_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
    if elapsed_time >= workTime:
        print("false")
        break
    
    
    