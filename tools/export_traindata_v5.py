import xml.etree.cElementTree as ET
import argparse
import os 
import tqdm
import shutil

# 归一化
def convert(size , box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x, y, w, h)

# labelme xml解析函数
def analysis_xml_from_app_labelme(xml_path, label_path, labels):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    w = int(size.find("width").text)
    h = int(size.find("height").text)
    
    # 标签文件
    name_image_path = root.find("filename").text
    name_image_cut = name_image_path.split(".")
    if(len(name_image_cut) == 2):
        name_image = name_image_cut[0]
    elif(len(name_image_cut) > 2):
        name_image_cut.pop(-1)
        name_image = ".".join(name_image_cut)
    
    labelFiles = open(os.path.join(label_path, name_image + ".txt"), "w", encoding="utf-8")
    
    for obj in root.iter("object"):
        # 排除困难标签
        difficult = obj.find("difficult").text
        if int(difficult) == 1: continue

        # class and offsets
        cls_name = obj.find("name").text
        if cls_name not in labels:
            labels.append(cls_name)
        x_idx = labels.index(cls_name) # 量化类别
        bndbox = obj.find("bndbox")
        x_b = (
            float(bndbox.find("xmin").text),
            float(bndbox.find("xmax").text),
            float(bndbox.find("ymin").text),
            float(bndbox.find("ymax").text)
        )
        
        x_bb = convert((w, h), x_b)
        labelFiles.write(str(x_idx) + " " + " ".join([str(x_a) for x_a in x_bb]) + "\n")
    
    labelFiles.close()
    return name_image
    
     
###############
#### start ####
###############
### 生成配置文件  
"""
    datasets
       |-----JPEGImage     # 原始图像
       |-----Annotations   # xml文件
    ---> 转变为
    datasets
        |-----images
            ---- train
                    --- 1.jpg
                    --- 2.jpg        
            ---- val 
                    --- 1.jpg
                    --- 2.jpg
        |-----labels
            ---- train
                    --- 1.txt
                    --- 2.txt
            ---- val 
                    --- 1.txt
                    --- 2.txt
"""
parser = argparse.ArgumentParser(description="export training datasets of Yolo serial")
parser.add_argument("--src_path", default=None, help="datasets source path")
parser.add_argument("--save_path",default=None, help="generate dataset save path")
parser.add_argument("--namespace", default=None, help="Generate file names")
parser.add_argument("--val_numbers", default=10, help="Randomly select validation set")
args = parser.parse_args()
    
print("------------- start export labels and configs-------------")
annotations_dirs = os.path.join(args.src_path, "Annotations")
jpegImages_dirs = os.path.join(args.src_path, "JPEGImages")


# 生成文件
new_file_dataset_main_dirs = os.path.join(args.save_path, args.namespace)
os.makedirs(new_file_dataset_main_dirs, exist_ok=True)

new_train_img_dirs = os.path.join(new_file_dataset_main_dirs, "images", "train")
new_train_txt_dirs = os.path.join(new_file_dataset_main_dirs, "labels", "train")
new_val_img_dirs = os.path.join(new_file_dataset_main_dirs, "images", "val")
new_val_txt_dirs = os.path.join(new_file_dataset_main_dirs, "labels", "val")

os.makedirs(new_train_img_dirs, exist_ok=True)
os.makedirs(new_train_txt_dirs, exist_ok=True)
os.makedirs(new_val_img_dirs, exist_ok=True)
os.makedirs(new_val_txt_dirs, exist_ok=True)

# 生成训练集和验证集
labels = []
for xml_name in tqdm.tqdm(os.listdir(annotations_dirs), desc="create train sets"):
    xml_path = os.path.join(annotations_dirs, xml_name)
    name_image = analysis_xml_from_app_labelme(xml_path, new_train_txt_dirs, labels) 
    if name_image is not None:
        shutil.copy(os.path.join(jpegImages_dirs, name_image+".jpg"), new_train_img_dirs) 

import random
random.seed(1024)

train_images = os.listdir(new_train_img_dirs)
val_images = random.sample(train_images, min(int(args.val_numbers), len(train_images)))
for img_name in tqdm.tqdm(val_images, desc="create val sets"):
    img_src_path = os.path.join(new_train_img_dirs, img_name)
    shutil.copy(img_src_path, new_val_img_dirs)
    os.remove(img_src_path) # train中移除图像
    
    label_name = img_name.replace(".jpg", ".txt")
    label_src_path = os.path.join(new_train_txt_dirs, label_name)
    shutil.copy(label_src_path, new_val_txt_dirs)
    os.remove(label_src_path) # train中移除标签

# 生成data.yaml

yaml_file_path = os.path.join(new_file_dataset_main_dirs, args.namespace + ".yaml")
with open(yaml_file_path, "w", encoding="utf-8") as yaml_file:
    yaml_file.write("#Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]\n")
    yaml_file.write(f"path: {new_file_dataset_main_dirs}\n")
    yaml_file.write(f"train: images/train\n")
    yaml_file.write(f"val: images/val\n")
    yaml_file.write(f"test: \n")
    
    #classes
    yaml_file.write("\n#classes\n")
    yaml_file.write(f"nc: {len(labels)}\n")
    yaml_file.write("names: [")
    yaml_file.write(" ,".join([f"'{label}'" for label in labels]))
    yaml_file.write("]\n")
    
print("The yolov5 dataset Yaml configuration file has been generated!")