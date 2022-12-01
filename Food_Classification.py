import streamlit as st

from predict import *
import os

name2idx = {v:k for k,v in menu_code.items()}
display_image_path = 'display_image'

def process(img_path):
    img = io.imread(img_path)
    st.image(img, use_column_width=True, channels="RGB")
    label = predict_image_onnx(img)

    text = f"""Top 5 Predicted Menu:"""
    answer = f"""{' | '.join(label)}"""
    # st.success(text)
    st.markdown(f"<h3 style='text-align: center; color: green;'>{text}</h1>", unsafe_allow_html=True)
    st.markdown(f"<h4 style='text-align: center; color: green;'>{answer}</h1>", unsafe_allow_html=True)

    fig = plt.figure(figsize=(30, 8))
    i = 1
    for label_name in label:
        plt.subplot(1,5,i)
        idx_name = name2idx[label_name]
        src_img = display_image_path + '/' + idx_name + '.jpg'
        img = io.imread(src_img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized_img = cv2.resize(img, (224, 224))
        resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        plt.imshow(resized_img)
        plt.title(f"""Top {i}: {label_name}""", fontsize = 14, fontweight = 'bold')
        plt.axis('off')
        i += 1
    st.image(get_img_from_fig(fig))

st.set_page_config(
    page_title="i-LiveWell:AI-Assisted Thai-Food Menu Image Reading",
    page_icon="🍜",
    # layout="wide",
    # initial_sidebar_state="expanded",
    )

st.title("ผู้ช่วยทำนายเมนูอาหารไทย")
st.image('CU_UTC_Rad_logo.png',use_column_width=True)

# https://github.com/Amiiney/cld-app-streamlit/blob/main/app.py


# Set the selectbox for demo images
# st.write("**Select an image for a DEMO**")
# menu_list = sorted(os.listdir('./sample_image'))
# menu = ["Select an Image"] + menu_list
# choice = st.selectbox("Select an image", menu)


# Set the box for the user to upload an image
st.write("**กรุณาอัพโหลดรูปภาพ**")
uploaded_image = st.file_uploader(
    # "Upload your image in JPG or PNG format", type=["jpg", "png", "jpeg"]
    "ถ่ายรูปมื้ออาหารของคุณเลย !!!", type=["jpg", "png", "jpeg"]
)

if uploaded_image:
    process(uploaded_image)

# elif choice != 'Select an Image':
#     img_path = f"""sample_image/{choice}"""
#     process(img_path)
