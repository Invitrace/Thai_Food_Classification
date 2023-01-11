from skimage import io
import cv2
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
font_path = 'Tahoma.ttf'  # Your font path goes here
fm.fontManager.addfont(font_path)
plt.rcParams['font.family'] = 'Tahoma'

import onnxruntime as rt
import numpy as np
from PIL import Image, ImageDraw

menu_code = { '00':'แกงเขียวหวานไก่',
            '01':'แกงเทโพ',
            '02':'แกงเลียง',
            '03':'แกงจืดเต้าหู้หมูสับ',
            '04':'แกงจืดมะระยัดไส้',
            '05':'แกงมัสมั่นไก่',
            '06':'แกงส้มกุ้ง',
            '07':'แกงผัดเผ็ดมะม่วงหิมพานต์',
            '08':'ไข่เจียว',
            '09':'ไข่ดาว',
            '10':'ไข่พะโล้',
            '11':'ไข่ลูกเขย',
            '12':'กล้วยบวชชี',
            '13':'ก๋วยเตี๋ยวคั่วไก่',
            '14':'กะหล่ำปลีผัดน้ำปลา',
            '15':'กุ้งแม่น้ำเผา',
            '16':'กุ้งอบวุ้นเส้น',
            '17':'ขนมครก',
            '18':'ข้าวเหนียวมะม่วง',
            '19':'ข้าวขาหมู',
            '20':'ข้าวคลุกกะปิ',
            '21':'ข้าวซอย',
            '22':'ข้าวผัด',
            '23':'ข้าวผัดกุ้ง',
            '24':'ข้าวมันไก่',
            '25':'ข้าวหมกไก่',
            '26':'ต้มข่าไก่',
            '27':'ต้มยำกุ้ง',
            '28':'ทอดมัน',
            '29':'ปอเปี๊ยะทอด',
            '30':'ผักบุ้งไฟแดง',
            '31':'ผัดไท',
            '32':'ผัดกะเพรา',
            '33':'ผัดซีอิ๋วเส้นใหญ่',
            '34':'ผัดฟักทองใส่ไข่',
            '35':'ผัดมะเขือยาวหมูสับ',
            '36':'ผัดหอยลาย',
            '37':'ฝอยทอง',
            '38':'พะแนงไก่',
            '39':'ยำถั่วพู',
            '40':'ยำวุ้นเส้น',
            '41':'ลาบหมู',
            '42':'สังขยาฟักทอง',
            '43':'สาคูไส้หมู',
            '44':'ส้มตำ',
            '45':'หมูปิ้ง',
            '46':'หมูสะเต๊ะ',
            '47':'ห่อหมก',   
}

key_list = list(menu_code.keys())
val_list = list(menu_code.values())

session = rt.InferenceSession('foodydudy_model.onnx')

# To crop an image with a certain percentage
def crop_to_square_with_certain_percentage(im, percentage = 0.1):
    # Get the width and height of the image
    width, height = im.size

    # Calculate the length of the shorter side
    if width > height:
        shorter_length = height
    else:
        shorter_length = width

    # Crop the image to a square using the shorter length
    left = (width - shorter_length) / 2
    top = (height - shorter_length) / 2
    right = (width + shorter_length) / 2
    bottom = (height + shorter_length) / 2

    # Get the cropped image and store it
    cropped_im = im.crop((left, top, right, bottom))

    # Crop the image by a certain percentage
    percentage = percentage #0.1 # 10%
    new_width = int(shorter_length - (shorter_length * percentage))
    new_height = new_width
    x = (shorter_length - new_width) / 2
    y = (shorter_length - new_height) / 2

    # Crop the image 
    cropped_im = cropped_im.crop((x, y, x + new_width, y + new_height))

    # Draw a rectangle on the original image
    draw = ImageDraw.Draw(im)
    draw.rectangle((left+x, top+y, right-x, bottom-y), outline=(255, 0, 0), width=5)

    # Save the image
#     im.save("image_with_rectangle.jpg")
    
    return im, cropped_im

def resizing_image(image, target_size):
    # Get the width and height of the image
    width, height = image.size

    # Calculate the aspect ratio
    aspect_ratio = width / height

    # determine the longer side of the image
    longer_side = max(width, height)

    # define the new longer side length
    new_longer_side = target_size

    # calculate the new size
    if width > height:
        new_size = (new_longer_side, int(new_longer_side / aspect_ratio))
    else:
        new_size = (int(new_longer_side * aspect_ratio), new_longer_side)
    
    # Resize the image
    image = image.resize(new_size, resample=Image.Resampling.LANCZOS)

    # Save the resized image
#     image.save('resized_image.jpg')
    return image

def predict_image_onnx(img, session = session, topn = 3):
    # img=io.imread(img_url)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # resized_img = cv2.resize(img, (224, 224))

    img = Image.fromarray(img)
    img_with_draw, cropped_im = crop_to_square_with_certain_percentage(img, percentage= 0.15)
    resized_img = resizing_image(cropped_im, 224)
    resized_img = resized_img.convert("RGB")
    x = np.expand_dims(resized_img, axis=0).astype(np.float32)
    inputDetails = session.get_inputs()
    prediction_proba = session.run(None, {inputDetails[0].name: x})[0]
    prediction = prediction_proba.argsort().ravel()[::-1] # Sort index from largest prob to low prob
    label_prob = [f'{val_list[i]} ({prediction_proba[0][i]:.1%})' for i in prediction[:topn]]
    label = [val_list[i] for i in prediction[:topn]]
    # plt.figure(figsize=(14,10)) 
    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # plt.title(f"""{' | '.join(label)}""", size = 10, color = "black", fontweight = 'bold')
    # plt.xticks([])
    # plt.yticks([])
    # plt.show

    return label, label_prob, img_with_draw


# define a function which returns an image as numpy array from figure
def get_img_from_fig(fig, dpi=256):
    import io
    buf = io.BytesIO()
    fig.savefig(buf, bbox_inches='tight', format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img