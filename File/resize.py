from PIL import Image
import glob,os

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

root_path = "./Mydata/"
width = 300
height = 300 

files = glob.glob(root_path+"my_img/*")
for file in files:
    file_path = os.path.basename(file)
    img = Image.open(file)
    img_resized = img.resize((width, height))
    print(root_path + "resize_img/" + file_path)
    img_resized.save(root_path + "resize_img/" + file_path,quality=100)
   