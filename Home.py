import streamlit as st
from PIL import Image
import base64
st.set_page_config(
    page_title="Thị Giác Máy", layout="wide",
    page_icon="🎥",
)


# Chuyển ảnh sang base64
def image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

image1 = image("Logoute.png")  
image2 = image("LogoFME.jpg")  
image3 = image("TGM.jpg")
image4 = image("TMG.png")
# Logo Trường
st.markdown(f"""
    <div style="text-align: center; margin-top: 20px;">   
        <img src="data:image/png;base64,{image1}" style="width: 70px; margin-top: -100px;margin-bottom: -20px"/>
        <div style="display: inline-block;margin-left: 20px">
            <img src="data:image/png;base64,{image2}" style="width: 70px;margin-top: -65px; "/>
            <div style="font-size: 18px; color: #66b3ff; font-weight: bold; font-family: 'Arial', sans-serif; margin-top: -10px;text-shadow: 1px 1px 1px rgba(0, 0, 0, 0.1)">
                    FME
                </div>
        </div>
        <h2 style="margin-top: -10px; color: #2e3090; font-size: 25px; font-weight: bold; text-transform: uppercase;padding-left: 60px;text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.3)">Trường Đại học Sư phạm Kỹ thuật TP.HCM
        </h2>
        <h3 style="margin-top: 20px; color: #66b3ff; font-size: 23px; font-weight: bold; text-transform: uppercase;padding-left: 60px;margin-top: -20px;text-shadow: 1px 1px 1px rgba(0, 0, 0, 0.3)">Khoa Cơ khí Chế tạo máy
        </h3>
        <h4 style="color: ##2e3090; font-size: 48px; font-weight: bold; text-transform: uppercase; padding-left: 60px; text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.3); margin-top: 20px;">
            Chào mừng các bạn đến với project Thị Giác Máy của nhóm mình
        </h4>
        <h5 style="font-size: 22px; color: #66b3ff; font-style: italic;margin-top: 40px; font-weight: lighter; font-family: 'Arial', sans-serif;">
            Họ và tên: Lâm Phước An  <span style="margin-left: 20px;">  MSSV: 22146257
        </h5>
        <h6 style="font-size: 22px; color: #66b3ff; font-style: italic;margin-top: -10px; font-weight: lighter; font-family: 'Arial', sans-serif;">
            Họ và tên: Trần Quốc Tuấn <span style="margin-left: 20px;">  MSSV: 22146445
        </h6>
        <div style="text-align: right; margin-top: 20px">   
        <img src="data:image/png;base64,{image3}" style="width: 400px;"/>
        </div>
        <div style="text-align: left; margin-top: -220px">   
        <img src="data:image/png;base64,{image4}" style="width: 300px;"/>
        </div>
       
""", unsafe_allow_html=True)
st.sidebar.success("Bạn hãy chọn các mục trên")
