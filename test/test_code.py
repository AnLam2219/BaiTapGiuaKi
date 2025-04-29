import numpy as np
import cv2
L= 256
a = 2222
image = cv2.imread("F://HK2_N3//ThiGiacMay//TaiLieu_XuLyAnh_ThiGiacMay//SachXuLyAnh//DIP3E_Original_Images_CH03//Fig0305(a)(DFT_no_log).tif",cv2.IMREAD_COLOR)
imgin = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
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
cv2.imshow("imgout",imgout)
cv2.waitKey(0)