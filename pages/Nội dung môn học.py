import streamlit as st
from PIL import Image
import numpy as np
import cv2
import io

st.title("Intensity Transformations")

L = 256
col1,col2 = st.columns(2)
imgin_frame = col1.empty()
imgout_frame = col2.empty()
chapter = st.sidebar.selectbox("Chapter",["Chapter 3","Chapter 4","Chapter 9"])
option = ""
if chapter == "Chapter 3": 
    option = st.sidebar.radio("----------------",("Image Negatives","Log Transformations","Power-Law Transformations","Piecewise Linear Transformation","Histogram","Histogram Equalization","Local Histogram Processing"))
elif chapter == "Chapter 4":
    option = st.sidebar.radio("----------------",("Spectrum","Remove Moiré","Remove Interference","Create Motion","Demotion","Demotion Noise"))
else:
     option = st.sidebar.radio("----------------",("Erosion","Dilation","Boundary","Contour","Convex Hull","Defect Detect","Connected Components","Remove Small Rice"))
img_file1 = img_file = st.sidebar.file_uploader("Upload image for process", type=["bmp", "png", "jpg", "jpeg","tif"])
process = st.sidebar.button("Process")

# Hàm riêng trong chương 4
def CreateNotchFilter(M,N):
    # tạo mảng PxQ 2 lớp và toàn bộ giá trị bằng 1
    H = np.ones((M,N),np.complex64)
    # sửa lớp số 2 thành toàn số 0
    H.imag = 0.0
    # các điểm gây nhiễu đã tìm thấy trên ảnh
    u1,v1 = 44,55
    u2,v2 = 85,55
    u3,v3 = 40,111
    u4,v4 = 84,111
    
    u5,v5 = M-44,N-55
    u6,v6 = M-85,N-55
    u7,v7 = M-40,N-111
    u8,v8 = M-84,N-111
    # cho ngẫu nhiên giá trị D0
    D0 = 15
    for u in range(0,M):
        for v in range(0,N):
            # u1, v1
            D = np.sqrt((1.0*u-u1)**2+(1.0*v-v1)**2)
            if D <= D0:
                H.real[u,v]=0.0
            # u2, v2
            D = np.sqrt((1.0*u-u2)**2+(1.0*v-v2)**2)
            if D <= D0:
                H.real[u,v]=0.0
            # u3, v3
            D = np.sqrt((1.0*u-u3)**2+(1.0*v-v3)**2)
            if D <= D0:
                H.real[u,v]=0.0
            # u4, v4
            D = np.sqrt((1.0*u-u4)**2+(1.0*v-v4)**2)
            if D <= D0:
                H.real[u,v]=0.0
            # u5, v5
            D = np.sqrt((1.0*u-u5)**2+(1.0*v-v5)**2)
            if D <= D0:
                H.real[u,v]=0.0
            # u6, v6
            D = np.sqrt((1.0*u-u6)**2+(1.0*v-v6)**2)
            if D <= D0:
                H.real[u,v]=0.0
            # u7, v7
            D = np.sqrt((1.0*u-u7)**2+(1.0*v-v7)**2)
            if D <= D0:
                H.real[u,v]=0.0
            # u8, v8
            D = np.sqrt((1.0*u-u8)**2+(1.0*v-v8)**2)
            if D <= D0:
                H.real[u,v]=0.0
    return H
def CreateNotchInferenceFilter(M,N):
    H = np.ones((M,N),np.complex64)
    H.imag = 0.0
    D0 =7
    D1 =7
    for u in range(0,M):
        for v in range(0,N):
            if u not in range(M//2-D1,M//2+D1+1):
                if v in range(N//2-D0,N//2+D0+1):
                    H.real[u,v] = 0.0
    return H
def CreateMotionFilter(M,N):
    H = np.ones((M,N),np.complex64)
    H.imag = 0.0
    a = 0.1
    b = 0.1
    T = 1.0
    for u in range(0,M):
        for v in range(0,N):
            phi = np.pi*((u-M//2)*a+(v-N//2)*b)
            if abs(phi)<1.0e-6:
                RE = T
                IM = 0.0
            else:
                RE = T*np.sin(phi)/phi*np.cos(phi)
                IM = -T*np.sin(phi)/phi*np.sin(phi)
            H.real[u,v] = RE
            H.imag[u,v] = IM
    return H
# hàm bỏ qua bước mở rộng ảnh
def FrequencyFiltering(imgin, H):
    M,N = imgin.shape
    f = imgin.astype(np.float64)
    # Bước 1: DFT
    
    F = np.fft.fft2(f)
    # bước 2: shift in the center of the image
    F = np.fft.fftshift(F)
    # Bước 3: Nhân F với H
    G= F*H
    
    # Bước 4: Shift ra trở lại
    G = np.fft.ifftshift(G)

    # Bước 5: IDFT
    g = np.fft.ifft2(G)
    gR = g.real.copy()
    gR = np.clip(gR,0,L-1)
    imgout = gR.astype(np.uint8)
    return imgout
def DeMotionFilter(M,N):
    H = np.ones((M,N),np.complex64)
    H.imag = 0.0
    a = 0.1
    b = 0.1
    T = 1.0
    phi_prev = 0.0
    for u in range(0,M):
        for v in range(0,N):
            phi = np.pi*((u-M//2)*a+(v-N//2)*b)
            temp = np.sin(phi) 
            if abs(temp)<1.0e-6:
                if abs(phi)<1.0e-6:
                    RE = 1/T
                    IM = 0.0
                else:
                    phi = phi_prev
                    RE = phi/(T*np.sin(phi))*np.cos(phi)
                    IM = phi/(T*np.sin(phi))*np.sin(phi)
            else:
                RE = phi/(T*np.sin(phi))*np.cos(phi)
                IM = phi/(T*np.sin(phi))*np.sin(phi)
            H.real[u,v] = RE
            H.imag[u,v] = IM
            phi_prev = phi
    return H
def DeMotion(imgin):
    M,N = imgin.shape
    H = DeMotionFilter(M,N)
    imgout = FrequencyFiltering(imgin,H)
    return imgout
if img_file is not None:
    image = Image.open(img_file) #ảnh này là RGB theo PIL 
    imgin_frame.image(image)
    # nội dung chương 3
    if option == "Image Negatives":
        if process:
            imgin = np.array(image,dtype=np.uint8)
            print(imgin)
            # Xử lý ám ảnh
            M,N = imgin.shape
            imgout = np.zeros((M,N),np.uint8) + np.uint8(255)
            #phân ngưỡng
            for x in range (0,M):
                for y in range (0,N):
                    r  = imgin[x,y]
                    s = L - 1 - r
                    imgout[x,y] = np.uint8(s)
            # hiển thị kết quả bên cột 2
            imgout_frame.image(imgout)
    elif option == "Log Transformations":
        # st.header("Log Transformations")
        if process:
            imgin = np.array(image,dtype=np.uint8)
            M,N = imgin.shape
            imgout = np.zeros((M,N),np.uint8)
            # thêm 1.0 để chắc chắn thành số thực
            c = (L-1.0)/np.log(1.0*L)
            # quét ảnh
            for x in range(0,M):
                for y in range(0,N):
                    r = imgin[x,y]
                    if ( r == 0):
                        r = 1
                    s =c*np.log(1.0+r)
                    imgout[x,y] = np.uint8(s)
            imgout_frame.image(imgout)
    elif option == "Power-Law Transformations":
        st.header("Power-Law Transformations")
        if process:
            imgin = np.array(image,dtype=np.uint8)
            M,N = imgin.shape
            imgout = np.zeros((M,N),np.uint8)
            gamma = 5.0
            c = np.power(L-1.0,1-gamma)
            for x in range(0,M):
                for y in range(0,N):
                    r = imgin[x,y]
                    if ( r == 0):
                        r = 1
                    s = c*np.power(1.0*r,gamma)
                    imgout[x,y] = np.uint8(s)
            imgout_frame.image(imgout)
    elif option == "Piecewise Linear Transformation":
        st.header("Piecewise Linear Transformation")
        if process:
            imgin = np.array(image,dtype=np.uint8)
            M,N = imgin.shape
            imgout = np.zeros((M,N),np.uint8)
            rmin = imgin[0,0]
            rmax = imgin[0,0]
            # tìm giá trị rmin và rmax bằng hàm minMaxLox
            rmin, rmax, _, _ = cv2.minMaxLoc(imgin)
            r1 = rmin
            s1 = 0 
            r2 =rmax
            s2 = L-1
            for x in range(0,M):
                for y in range(0,N):
                    r = imgin[x,y]
                    # if r < rmin:
                    #     rmin = r
                    # if r > rmax:
                    #     rmax = r
                    # đoạn I
                    if r < rmin:
                        s = s1/r1*r
                    elif r<r2:
                    # Đoạn II
                        s = (s2-s1)/(r2-r1)*(r-r1)+s1
                    # Đoạn III
                    else:
                        s = (L-1-s2)/(L-1-r2)*(r-r2)+s2
                    # thay đổi điểm ảnh của imgout bằng điểm ảnh s đã chỉnh sửa 
                    imgout[x,y] = np.uint8(s) 
            imgout_frame.image(imgout)
    elif option == "Histogram":
        st.header("Histogram")
        if process:
            imgin = np.array(image,dtype=np.uint8)
            M,N = imgin.shape
            imgout = np.zeros((M,L,3),np.uint8)+np.uint8(255)
            h = np.zeros(L,np.int32)
            for x in range(0,M):
                for y in range(0,N):
                    r = imgin[x,y]
                    h[r]=h[r]+1
            p = 1.0*h/(M*N)
            scale = 3000
            for r in range(0,L):
                cv2.line(imgout,(r,M-1),(r,M-1-np.int32(scale*p[r])),(255,0,0))
            imgout_frame.image(imgout)
    elif option == "Histogram Equalization":
        st.header("Histogram Equalization")
        if process:
            imgin = np.array(image,dtype=np.uint8)
            M,N  = imgin.shape
            imgout = np.zeros((M,N),np.uint8)
            h = np.zeros(L,np.int32)
            
            for x in range(0,M):
                for y in range(0,N):
                    r = imgin[x,y]
                    h[r]=h[r]+1
            p = 1.0*h/(M*N)
            s = np.zeros(L,np.float64)
            for k in range (0,L):
                for j in range (0,k+1):
                    s[k] = s[k]+p[j]
            for x in range (0,M):
                for y in range (0,N):
                    r = imgin[x,y]
                    imgout[x,y] = np.uint8((L-1)*s[r])
            imgout_frame.image(imgout)
    elif option == "Local Histogram Processing":
        st.header("Local Histogram Processing")
        if process:
            imgin = np.array(image,dtype=np.uint8)
            M,N = imgin.shape
            imgout = np.zeros((M,N),np.uint8)
            m =3
            n =3
            a = m//2
            b = n//2 #Chia lấy phần nguyên
            for x in range(a,M-a):
                for y in range(b,N-b):
                    w = imgin[x-a:x+a+1, y-b:y+b+1]
                    w = cv2.equalizeHist(w)
                    imgout[x,y] = w[a,b]
            imgout_frame.image(imgout)

    # Nội dung chương 4
    elif option == "Spectrum":
        if process:
            imgin = np.array(image,dtype=np.uint8)
            M,N = imgin.shape
            # Bước 1: DFT
            f = imgin.astype(np.float32)//(L-1)
            # bước 2: shift in the center of the image
            F = np.fft.fft2(f)
            F = np.fft.fftshift(F)
            # Bước 3: Nhân F với H
            S = np.sqrt(F.real**2+F.imag**2)
            S = np.clip(S,0,L-1)
            imgout = S.astype(np.uint8)
            imgout_frame.image(imgout)
    elif option == "Remove Moiré":
        if process:
            imgin = np.array(image,dtype=np.uint8)
            M,N = imgin.shape
            H = CreateNotchFilter(M,N)
            imgout = FrequencyFiltering(imgin,H)
            imgout_frame.image(imgout)
    elif option == "Remove Interference":
        if process:
            imgin = np.array(image,dtype=np.uint8)
            M,N = imgin.shape
            H = CreateNotchInferenceFilter(M,N)
            imgout = FrequencyFiltering(imgin,H)
            imgout_frame.image(imgout)
    elif option =="Create Motion":
        if process:
            imgin = np.array(image,dtype=np.uint8)
            M,N = imgin.shape
            H = CreateMotionFilter(M,N)
            imgout = FrequencyFiltering(imgin,H)
            imgout_frame.image(imgout)
    elif option == "Demotion":
        if process:
            imgin = np.array(image,dtype=np.uint8)
            M,N = imgin.shape
            H = DeMotionFilter(M,N)
            imgout = FrequencyFiltering(imgin,H)
            imgout_frame.image(imgout)
    elif option == "Demotion Noise":
        if process:
            imgin = np.array(image,dtype=np.uint8)
            temp = cv2.medianBlur(imgin,7)
            imgout = DeMotion(temp)
            imgout_frame.image(imgout)
    # Nội dung chương 9
    elif option == "Erosion":
        if process:
            imgin = np.array(image,dtype=np.uint8)
            for x in range(0,imgin.shape[0]):
                for y in range(0,imgin.shape[1]):
                    if imgin[x][y] == 1:
                        imgin[x][y] = 255
            w = cv2.getStructuringElement(cv2.MORPH_RECT,(45,45))
            imgout = cv2.erode(imgin,w)
            imgout_frame.image(imgout)
    elif option == "Dilation":
        if process:
            imgin = np.array(image,dtype=np.uint8)
            for x in range(0,imgin.shape[0]):
                for y in range(0,imgin.shape[1]):
                    if imgin[x][y] == 1:
                        imgin[x][y] = 255
            w = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
            imgout = cv2.dilate(imgin,w)
            imgout_frame.image(imgout)
    elif option == "Boundary":
        if process:
            imgin = np.array(image,dtype=np.uint8)
            for x in range(0,imgin.shape[0]):
                for y in range(0,imgin.shape[1]):
                    if imgin[x][y] == 1:
                        imgin[x][y] = 255
            w = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
            erosate_img = cv2.erode(imgin,w)
            imgout = imgin - erosate_img
            imgout_frame.image(imgout)
    elif option == "Contour":
        if process:
            imgin = np.array(image,dtype=np.uint8)
            for x in range(0,imgin.shape[0]):
                for y in range(0,imgin.shape[1]):
                    if imgin[x][y] == 1:
                        imgin[x][y] = 255
            imgout = cv2.cvtColor(imgin,cv2.COLOR_GRAY2BGR)
            contours,_ = cv2.findContours(imgin, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            contour = contours[0]
            n = len(contour)
            for i in range(0,n-1):
                # biết được điểm đầu điểm cuối của biên dạng
                x1 = contour[i,0,0]
                y1 = contour[i,0,1]
                x2 = contour[i+1,0,0]
                y2 = contour[i+1,0,1]
                cv2.line(imgout,(x1,y1),(x2,y2),(0,255,0),2)
            # nối 2 điểm đầu cuối lại
            x1 = contour[n-1,0,0]
            y1 = contour[n-1,0,1]
            x2 = contour[0,0,0]
            y2 = contour[0,0,1]
            cv2.line(imgout,(x1,y1),(x2,y2),(0,255,0),2)
            imgout_frame.image(imgout)
    elif option == "Convex Hull":
        if process:
            imgin = np.array(image,dtype=np.uint8)
            for x in range(0,imgin.shape[0]):
                for y in range(0,imgin.shape[1]):
                    if imgin[x][y] == 1:
                        imgin[x][y] = 255
            imgout = cv2.cvtColor(imgin,cv2.COLOR_GRAY2BGR)
            # Buoc 1: Tinh contour
            # Lưu ý: contour là biên có thứ tự của ảnh nhị phân
            contours,_ = cv2.findContours(imgin, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            # ảnh của tan chỉ có một contour
            contour = contours[0]
            # bước 2 tính bao lồi từ contour
            hull = cv2.convexHull(contour,returnPoints=False)
            n = len(hull)
            # nối các điểm từ hull trả về để có đường bao lồi
            for i in range(0,n-1):
                vi_tri_1 = hull[i,0]
                vi_tri_2 = hull[i+1,0]
                x1 = contour[vi_tri_1,0,0]
                y1 = contour[vi_tri_1,0,1]
                x2 = contour[vi_tri_2+1,0,0]
                y2 = contour[vi_tri_2+1,0,1]
                cv2.line(imgout,(x1,y1),(x2,y2),(0,255,0),2)
            vi_tri_1 = hull[n-1,0]
            vi_tri_2 = hull[0,0]
            x1 = contour[vi_tri_1,0,0]
            y1 = contour[vi_tri_1,0,1]
            x2 = contour[vi_tri_2,0,0]
            y2 = contour[vi_tri_2,0,1]
            cv2.line(imgout,(x1,y1),(x2,y2),(0,255,0),2)
            imgout_frame.image(imgout)
    elif option == "Defect Detect":
        if process:
            imgin = np.array(image,dtype=np.uint8)
            for x in range(0,imgin.shape[0]):
                for y in range(0,imgin.shape[1]):
                    if imgin[x][y] == 1:
                        imgin[x][y] = 255
            imgout = cv2.cvtColor(imgin,cv2.COLOR_GRAY2BGR)
            # Buoc 1: Tinh contour
            # Lưu ý: contour là biên có thứ tự của ảnh nhị phân
            contours,_ = cv2.findContours(imgin, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            # ảnh của tan chỉ có một contour
            contour = contours[0]
            # bước 2 tính bao lồi từ contour
            hull = cv2.convexHull(contour,returnPoints=False)
            n = len(hull)
            # nối các điểm từ hull trả về để có đường bao lồi
            for i in range(0,n-1):
                vi_tri_1 = hull[i,0]
                vi_tri_2 = hull[i+1,0]
                x1 = contour[vi_tri_1,0,0]
                y1 = contour[vi_tri_1,0,1]
                x2 = contour[vi_tri_2+1,0,0]
                y2 = contour[vi_tri_2+1,0,1]
                cv2.line(imgout,(x1,y1),(x2,y2),(0,255,0),2)
            vi_tri_1 = hull[n-1,0]
            vi_tri_2 = hull[0,0]
            x1 = contour[vi_tri_1,0,0]
            y1 = contour[vi_tri_1,0,1]
            x2 = contour[vi_tri_2,0,0]
            y2 = contour[vi_tri_2,0,1]
            cv2.line(imgout,(x1,y1),(x2,y2),(0,255,0),2)
            defects = cv2.convexityDefects(contour,hull)
            max_depth = np.max(defects[:,:,3])
            n = len(defects)
            for i in range(0,n):
                depth = defects[i,0,3]
                if depth > max_depth/2:
                    vi_tri = defects[i,0,2]
                    x = contour[vi_tri,0,0]
                    y = contour[vi_tri,0,1]
                    cv2.circle(imgout,(x,y),5,(255,0,0),-1)
            imgout_frame.image(imgout)
    elif option == "Connected Components":
        if process:
            imgin = np.array(image,dtype=np.uint8)
            for x in range(0,imgin.shape[0]):
                for y in range(0,imgin.shape[1]):
                    if imgin[x][y] == 1:
                        imgin[x][y] = 255
            threshhold = 200
            # phân ngưỡng chuyển ảnh thành ảnh trắng đen (binary image)
            _, temp = cv2.threshold(imgin,threshhold,L-1,cv2.THRESH_BINARY)
            #  Xóa nhiễu median
            imgout = cv2.medianBlur(temp,7)
            # đếm số lượng 
            n,label = cv2.connectedComponents(imgout ,None)
            a = np.zeros(n,np.int32)
            M,N = label.shape
            for x in range(0,M):
                for y in range(0,N):
                    r = label[x,y]
                    if r>1:
                        a[r] = a[r] + 1
            s = 'Co %d thanh phan lien thong' % (n-1)
            cv2.putText(imgout,s,(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255))
            for r in range(1,n):
                s = '%3d %4d' % (r,a[r])
                cv2.putText(imgout,s,(10,(r+1)*20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255))
            imgout_frame.image(imgout)    
    elif option == "Remove Small Rice":
        if process:
            imgin = np.array(image,dtype=np.uint8)
            for x in range(0,imgin.shape[0]):
                for y in range(0,imgin.shape[1]):
                    if imgin[x][y] == 1:
                        imgin[x][y] = 255
            w = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(81,81)) #số 81 là kích thước lớn nhất của hạt gạo
            temp = cv2.morphologyEx(imgin,cv2.MORPH_TOPHAT,w) #cho ra được ảnh làm rõ bóng của nó
            # threshhold làm ảnh trắng đen
            threshhold = 65
            _,temp = cv2.threshold(temp,threshhold,L-1,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
            n,label = cv2.connectedComponents(temp ,None)
            a = np.zeros(n,np.int32)
            M,N = label.shape
            for x in range(0,M):
                for y in range(0,N):
                    r = label[x,y]
                    if r>0:
                        a[r] = a[r] + 1
            max_value = np.max(a)
            imgout = np.zeros((M,N),np.uint8) #tạo ra ảnh trông màu đen có cùng kích thước
            for x in range(0,M):
                for y in range(0,N):
                    r = label[x,y]
                    if r > 0:
                        if a[r] > max_value*0.5: #giữ lại các hạt gạo có kích thước lớn hơn 70%
                            imgout[x,y] = L-1 #thêm những hạt gạo đạt tiêu chuẩn 
            imgout_frame.image(imgout,use_container_width=True)    