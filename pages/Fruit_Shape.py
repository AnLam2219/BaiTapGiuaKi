import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

st.title("Fruit Detection")

try:
    if st.session_state["LoadModel"] == True:
        print("Loaded model")
except:
    st.session_state["LoadModel"] = True
    st.session_state["Net"] = YOLO("F:/HK2_N3/ThiGiacMay/BaiTapGiuaKi/pages/models/yolov8n_traicay.pt", task="detect")
    print("First Load model")


img_file_buffer = st.file_uploader("Upload image for process", type=["bmp", "png", "jpg", "jpeg","tif"])

if img_file_buffer is not None:
    # Mở ảnh bằng PIL
    image = Image.open(img_file_buffer).convert("RGB")

    # Hiển thị ảnh gốc
    st.image(image, caption="Original image", use_container_width=True)

    # Chuyển sang NumPy để xử lý
    img_np = np.array(image)
    if st.button("Predict"):
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        img_out = img_bgr.copy()

        annotator = Annotator(img_out)

        results = st.session_state["Net"].predict(img_out, conf=0.6, verbose=False)

        boxes = results[0].boxes.xyxy.cpu()
        clss = results[0].boxes.cls.cpu().tolist()
        confs = results[0].boxes.conf.tolist()
        names = st.session_state["Net"].names
        for box, cls, conf in zip(boxes, clss, confs):
            label = f"{names[int(cls)]} {conf:.2f}"
            annotator.box_label(box, label=label, txt_color=(255,0,0), color=(255,255,255))

        img_result = cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)
        st.image(img_result, caption="Predicted image", use_container_width=True)
