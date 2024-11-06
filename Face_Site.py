import cv2
import numpy as np

def Predict(img, face_cascade, eye_cascade, emotion_model, emotion_labels):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    boxes = []  # Danh sách lưu trữ các bounding box
    emotions_detected = []  # Danh sách lưu trữ các cảm xúc được phát hiện

    for (x, y, w, h) in faces:
        boxes.append((x, y, w, h))  # Thêm bounding box vào danh sách
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        
        # Resize the face region to 48x48 for the emotion model
        roi_gray_resized = cv2.resize(roi_gray, (48, 48))
        roi_gray_resized = roi_gray_resized.astype('float32') / 255.0
        roi_gray_resized = np.expand_dims(roi_gray_resized, axis=0)
        roi_gray_resized = np.expand_dims(roi_gray_resized, axis=-1)
        
        # Predict emotion
        emotion_prediction = emotion_model.predict(roi_gray_resized)
        max_index = int(np.argmax(emotion_prediction))
        emotion_label = emotion_labels[max_index]
        emotions_detected.append(emotion_label)  # Thêm cảm xúc vào danh sách
        
        # Draw emotion label on the image
        cv2.putText(img, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)

    # Vẽ bounding box cho khuôn mặt
    for (x, y, w, h) in boxes:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Vẽ bounding box cho khuôn mặt
    
    # Trả về hình ảnh đã xử lý và danh sách cảm xúc
    return img, emotions_detected
