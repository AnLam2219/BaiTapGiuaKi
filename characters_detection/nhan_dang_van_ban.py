import cv2
import pytesseract
from PIL import Image, ImageDraw, ImageFont
import numpy as np
# Cài đặt pytesseract
pytesseract.pytesseract.tesseract_cmd = r"c:\Program Files\Tesseract-OCR\tesseract.exe"
# Đọc ảnh và chuyển đổi màu sắc từ BGR sang RGB
img = cv2.imread('F:\\HK2_N3\\ThiGiacMay\\BaiTapGiuaKi\\characters_detection\\image.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# Chuyển sang dạng PIL để vẽ chữ có dấu
img_pil = Image.fromarray(img)
draw = ImageDraw.Draw(img_pil)
font_path = "C:\\Windows\\Fonts\\arial.ttf"  # hoặc roboto.ttf, times.ttf...
font = ImageFont.truetype(font_path, size=18)
custom_config = r'--oem 1 --psm 6'
text = pytesseract.image_to_string(img_pil, lang='eng+vie', config=custom_config)
print(text)
# viets chữ vào file text.txt
with open('text.txt', 'w', encoding='utf-8') as f:
    f.write(text)
boxes = pytesseract.image_to_data(img_pil,lang='eng+vie', config=custom_config)
print(boxes)
for i, box in enumerate(boxes.splitlines()):
    if i != 0:
        box = box.split()
        if len(box) == 12:
            x, y, w, h = int(box[6]), int(box[7]), int(box[8]), int(box[9])
            # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
            draw.rectangle([x, y, x + w, y + h], outline=(128, 0, 128), width=2)
            # Vẽ chữ có dấu lên ảnh
            draw.text((x, y-20), box[11], font=font, fill=(128, 0, 128))
            # cv2.putText(img, box[11], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
img_result = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
cv2.imshow('img', img_result)
cv2.waitKey(0)
cv2.destroyAllWindows()