import streamlit as st
import cv2 
import numpy as np
from PIL import Image
st.title("Shape Detection")

#Hàm phân ngưỡng
def phan_nguong(imgin):
    #M: độ cao, N: độ rộng
    #Ảnh mà ma trận MxN
    M,N = imgin.shape
    #tạo mảng từ ảnh
    imgout = np.zeros((M,N), np.uint8)
    #quét qua từng phần tử
    for x in range(0,M):
        for y in range(0,N):
            r = imgin[x,y]
            #Số threshold từ ảnh của sensor 
            if (r==63):
                s = 255
            else:
                s = 0

            imgout[x,y]= np.uint8(s)
    #Hàm xóa nhiễu từ Opencv2
    imgout = cv2.medianBlur(imgout,7)
    #trả về giá trị cho hàm
    return imgout
img_file_buffer = st.sidebar.file_uploader("Upload image for process", type=["bmp", "png", "jpg", "jpeg","tif"])
Predict = st.sidebar.button("Predict")
# img_file_buffer = cv2.imread("F:/HK2_N3/ThiGiacMay/BaiTapGiuaKi/test/010.bmp",cv2.IMREAD_COLOR)
if img_file_buffer is not None:
    # Mở ảnh bằng PIL
    image = Image.open(img_file_buffer).convert("RGB")
    # Hiển thị ảnh gốc
    st.image(image, caption="Original image", use_container_width=True)
    # Chuyển sang NumPy để xử lý
    img_np = np.array(image,dtype=np.uint8)
    if Predict:
        img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        imgout = phan_nguong(img_gray)
        contours, hierarchy = cv2.findContours(imgout,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        # Lọc lại các phần tử đủ kích thước, loại bỏ các nhiễu hạt nhỏ
        filtered_contours = []
        for c in contours:
            # c là một ma trận lấy từ contours có dạng (n,1,2) n là số điểm, 1 là một mảng nhỏ array, 2 là điểm x,y
            if c.shape[0] > 25:
                filtered_contours.append(c)
        s = []
        for c in filtered_contours:
            # boundingRect trả về tọa độ góc bên trái (x,y), độ rộng và độ cao
            x, y, w, h = cv2.boundingRect(c)
            roi = imgout[y:y+h, x:x+w]  # Cắt vùng ROI (Region of Interest)
            m = cv2.moments(roi)
            hu = cv2.HuMoments(m)
            # print(hu[0,0])
            if 0.0006241509 <= hu[0,0] <= 0.0006304083:
                s.append('HinhTron')
            elif 0.0006479692 <= hu[0,0] <=0.0006677594:
                s.append('HinhVuong')
            elif 0.0007086009 <= hu[0,0] <=0.0007771916:
                s.append('HinhTamGiac')
            else:
                s.append('Unknown')
            # cv2.imshow('ROI', roi)
            # cv2.waitKey(0)
        imgout = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        # vòng lặp tự có thêm tạo biến đếm: for biến đếm, biến lặp in enumerate(mảng các phần tử được quét qua bởi c, số bắt đầu đếm của n)
        for n, c in enumerate(filtered_contours, 1):
            # Tính hình chữ nhật bao quanh nhỏ nhất
            rect = cv2.minAreaRect(c)
            # gán tên tương ứng tại mỗi Contours
            text = s[n-1]
            
            # Lấy 4 điểm của hình chữ nhật
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            
            # Vẽ đường viền hình chữ nhật
            cv2.drawContours(imgout, [box], 0, (0, 255, 0), 2)
            
            # Lấy tọa độ tâm của hình chữ nhật
            center = (int(rect[0][0]), int(rect[0][1])+100)
            
            # Gắn nhãn tại vị trí tâm
            cv2.putText(imgout, text, center, cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 1)
        st.image(imgout, caption="Predicted image", use_container_width=True)
        



