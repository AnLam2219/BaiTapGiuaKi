import streamlit as st
import cv2
import pytesseract
from PIL import Image, ImageDraw, ImageFont
import numpy as np
st.title("Nhận dạng văn bản")
col1,col2 = st.columns(2)
imgin_frame = col1.empty()
imgout_frame = col2.empty()

pytesseract.pytesseract.tesseract_cmd = r"c:\Program Files\Tesseract-OCR\tesseract.exe"
font_path = "C:\\Windows\\Fonts\\arial.ttf"  # hoặc roboto.ttf, times.ttf...
custom_config = r'--oem 1 --psm 6'
font = ImageFont.truetype(font_path, size=18)
img_file_buffer = st.sidebar.file_uploader("Upload image for process", type=["bmp", "png", "jpg", "jpeg","tif"])
text = []
predict = st.sidebar.button("Predict")
if predict:
    if img_file_buffer is not None:
        # Mở ảnh bằng PIL
        image = Image.open(img_file_buffer).convert("RGB")
        imgin_frame.image(image, caption="Original image", use_container_width=True)
        img_np = np.array(image)
        img_pil = Image.fromarray(img_np)
        draw = ImageDraw.Draw(img_pil)
        text = pytesseract.image_to_string(img_pil, lang='eng+vie', config=custom_config)
        boxes = pytesseract.image_to_data(img_pil,lang='eng+vie', config=custom_config)
        for i, box in enumerate(boxes.splitlines()):
            if i != 0:
                box = box.split()
                if len(box) == 12:
                    x, y, w, h = int(box[6]), int(box[7]), int(box[8]), int(box[9])
                    # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
                    draw.rectangle([x, y, x + w, y + h], outline=(128, 0, 128), width=2)
                    # Vẽ chữ có dấu lên ảnh
                    draw.text((x, y-20), box[11], font=font, fill=(128, 0, 128))
        imgout_frame.image(img_pil, caption="Predict image", use_container_width=True)
        st.header("Văn bản nhận dạng được:")
        st.text(text)
