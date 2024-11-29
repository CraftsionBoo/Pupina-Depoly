import argparse
import os 
import tqdm
import shutil
import xml.etree.cElementTree as ET


def normalize_coordinate_(line, width, height):
    elements = line.split()
    class_index = elements[0]
    coordinates = [
        "{:.5f}".format(float(coord) / width) if i % 2 == 0 else "{:.5f}".format(float(coord) / height)
        for i, coord in enumerate(elements[1:])
    ]
    normalized_line = ' '.join([class_index] + coordinates)
    return normalized_line

def analysis_xml_from_roboflow_(xml_file, label_folder, labels):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    size = root.find("size")
    
    w = int(size.find("width").text)
    h = int(size.find("height").text)
    
    name_image_path = root.find("filename").text
    name_image_cut = name_image_path.split(".")
    if(len(name_image_cut) == 2):
        name_image = name_image_cut[0]
    elif(len(name_image_cut) > 2):
        name_image_cut.pop(-1)
        name_image = ".".join(name_image_cut)
    
    labelFiles = open(os.path.join(label_folder, name_image + ".txt"), "w", encoding="utf-8")
    
    for obj in root.iter("object"):
        
        difficult = obj.find("difficult").text
        if (int(difficult)) == 1 : continue
        
        # cls and seg
        line = ""
        # cls
        cls_name = obj.find("name").text
        if cls_name not in labels:
            labels.append(cls_name)
        idx = labels.index(cls_name)
        line += str(idx) + " "

        # seg
        polygon = obj.find("polygon")
        if polygon is not None:
            for point in polygon:
                if point.tag.startswith("x"):
                    x = point.text
                    y = polygon.find('y{}'.format(point.tag[1:])).text
                    line += "{} {}".format(x, y) + " "
        line += "\n"
        normalize = normalize_coordinate_(line, w, h)
        labelFiles.write(normalize + "\n")    
    labelFiles.close()
    return name_image
    
################
#### start ####
###############
### yolov8 coco8-seg
"""
coco8-seg
    |----- images
             |----- train
             |----- val
    |----- labels
             |----- train
             |----- val
"""
parser = argparse.ArgumentParser(description="export training datasets of Yolo serial")
parser.add_argument("--src_path", default=None, help="datasets source path")
parser.add_argument("--save_path",default=None, help="generate dataset save path")
parser.add_argument("--namespace", default=None, help="Generate file names")
args = parser.parse_args()

print("------------- start export labels and configs-------------")
train_dirs = os.path.join(args.src_path, "train")
valid_dirs = os.path.join(args.src_path, "valid")
test_dirs = os.path.join(args.src_path, "test")

# 生成文件
new_file_dataset_main_dirs = os.path.join(args.save_path, args.namespace)
os.makedirs(new_file_dataset_main_dirs, exist_ok=True)

new_train_img_dirs = os.path.join(new_file_dataset_main_dirs, "images", "train")
new_train_txt_dirs = os.path.join(new_file_dataset_main_dirs, "labels", "train")
new_val_img_dirs = os.path.join(new_file_dataset_main_dirs, "images", "val")
new_val_txt_dirs = os.path.join(new_file_dataset_main_dirs, "labels", "val")
new_test_img_dirs = os.path.join(new_file_dataset_main_dirs, "images", "test")
new_test_txt_dirs = os.path.join(new_file_dataset_main_dirs, "labels", "test")

os.makedirs(new_train_img_dirs, exist_ok=True)
os.makedirs(new_train_txt_dirs, exist_ok=True)
os.makedirs(new_val_img_dirs, exist_ok=True)
os.makedirs(new_val_txt_dirs, exist_ok=True)
os.makedirs(new_test_img_dirs, exist_ok=True)
os.makedirs(new_test_txt_dirs, exist_ok=True)

# 生成训练集和验证集
labels = []
for file_name in tqdm.tqdm(os.listdir(train_dirs), desc = "create train sets"):
    if file_name.endswith("xml"):
        xml_path = os.path.join(train_dirs, file_name)
        jpg_name = analysis_xml_from_roboflow_(xml_path, new_train_txt_dirs, labels)
        shutil.copy(os.path.join(train_dirs, jpg_name+".jpg"), new_train_img_dirs)

for file_name in tqdm.tqdm(os.listdir(valid_dirs), desc = "create valid sets"):
    if file_name.endswith("xml"):
        xml_path = os.path.join(valid_dirs, file_name)
        jpg_name = analysis_xml_from_roboflow_(xml_path, new_val_txt_dirs, labels)
        shutil.copy(os.path.join(valid_dirs, jpg_name+".jpg"), new_val_img_dirs)    
        
for file_name in tqdm.tqdm(os.listdir(test_dirs), desc = "create test sets"):
    if file_name.endswith("xml"):
        xml_path = os.path.join(test_dirs, file_name)
        jpg_name = analysis_xml_from_roboflow_(xml_path, new_test_txt_dirs, labels)
        shutil.copy(os.path.join(test_dirs, jpg_name+".jpg"), new_test_img_dirs) 


yaml_file_path = os.path.join(new_file_dataset_main_dirs, args.namespace + ".yaml")
with open(yaml_file_path, "w", encoding="utf-8") as yaml_file:
    yaml_file.write("#Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]\n")
    yaml_file.write(f"path: {new_file_dataset_main_dirs}\n")
    yaml_file.write(f"train: images/train\n")
    yaml_file.write(f"val: images/val\n")
    yaml_file.write(f"test: # images/test(optional)\n")
    
    #classes
    yaml_file.write("\n#classes\n")
    yaml_file.write("names: \n")
    yaml_file.write("   ")
    yaml_file.write("   ".join([f"{i}: {label} \n" for i, label in enumerate(labels)]))

print("--------------------> greate done")