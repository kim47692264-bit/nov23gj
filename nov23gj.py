import streamlit as st
import cv2
import numpy as np
import mediapipe as mp

st.title("얼굴 기울기(roll) 측정")

mp_face = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils

uploaded = st.camera_input("얼굴을 카메라로 찍어주세요")

if uploaded is not None:
    # 업로드된 이미지를 OpenCV 형식으로 변환
    file_bytes = np.frombuffer(uploaded.getvalue(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    h, w, _ = img.shape

    with mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.6) as face_detector:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_detector.process(img_rgb)

        if results.detections:
            detection = results.detections[0]

            # 눈 좌표로 roll angle 계산 (네가 쓰던 함수 로직 그대로 넣으면 됨)
            kps = detection.location_data.relative_keypoints
            right_eye = kps[0]
            left_eye = kps[1]

            x1, y1 = right_eye.x * w, right_eye.y * h
            x2, y2 = left_eye.x * w, left_eye.y * h

            angle_rad = np.arctan2(y2 - y1, x2 - x1)
            angle_deg = angle_rad * 180.0 / np.pi

            mp_draw.draw_detection(img, detection)
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=f"얼굴 기울기: {angle_deg:.1f}°")

            st.write(f"**얼굴 기울기(roll angle): {angle_deg:.1f}°**")
        else:
            st.warning("얼굴을 찾지 못했어요. 조금 더 가까이 와서 다시 찍어줘!")