import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import sys

# parameter (docName, action)
docName = sys.argv[1]
target_action = sys.argv[2]


# for test
workTime = 30

# docName = "testDoc3"
# target_action = "nonee"


#setMotion.py와 유사
actions = ['first', 'second', 'third']
seq_length = 90


action_window_size = 30  # 액션 판단을 위한 윈도우 크기
action_threshold = 25  # 윈도우 내에서 동일한 액션의 비중이 일정 이상이면 액션 확정

# value setting
judgment_confidence = 0.95


model = load_model(f'models/model_{docName}.h5')

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

# w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
# out = cv2.VideoWriter('input.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (w, h))
# out2 = cv2.VideoWriter('output.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (w, h))

seq = []
predict_action = []
start_time = cv2.getTickCount()

while cap.isOpened():
    ret, img = cap.read()
    img0 = img.copy()

    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    this_action = "none"
    
    if result.multi_hand_landmarks is not None:
        for res in result.multi_hand_landmarks:
            joint = np.zeros((21, 4))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

            # Compute angles between joints
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

            # 데이터를 joint, angle 이어줌
            d = np.concatenate([joint.flatten(), angle])

            seq.append(d)

            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

            if len(seq) < seq_length:
                continue

            input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)


            # 결과 예측 y_pred에 저장
            y_pred = model.predict(input_data).squeeze()

            # 인덱스 뽑아냄
            i_pred = int(np.argmax(y_pred))
            
            # confidence 뽑아냄
            conf = y_pred[i_pred]







            # 확실하지 않다고 판단, continue
            if conf < judgment_confidence:
                continue


            # judgment_confidence 이상 -> predict_action 에 저장
            action = actions[i_pred]
            predict_action.append(action)



            if len(predict_action) < action_window_size:
                continue

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



    cv2.imshow('motion recognition', img)
    
    if this_action == target_action:
        print("true", this_action)
        break
    
    elapsed_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
    if elapsed_time >= workTime:
        print("false")
        break

    if cv2.waitKey(1) == ord('q'):
        break