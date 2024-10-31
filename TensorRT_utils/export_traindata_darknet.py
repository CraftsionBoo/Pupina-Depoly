import xml.etree.cElementTree as ET
import argparse
import os 
import tqdm

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
    
    return name_image
    
     
###############
#### start ####
###############
parser = argparse.ArgumentParser(description="export training datasets of Yolo serial")
parser.add_argument("--src_path", default=None, help="datasets source path")
parser.add_argument("--save_path", default=None, help="The path where the tag summary information is stored")
parser.add_argument("--namespace", default=None, help="Generate file names")
args = parser.parse_args()
    
print("------------- start export labels and configs-------------")
annotations_dirs = os.path.join(args.src_path, "Annotations")
jpegImages_dirs = os.path.join(args.src_path, "JPEGImages")

if not os.path.exists(args.save_path):
    print("Creating new folder {}".format(args.save_path))
    os.makedirs(args.save_path)

if args.namespace == None:
    file_train_txt = os.path.join(args.save_path, "train.txt")
    file_train_data = os.path.join(args.save_path, "train.data")
    file_train_names = os.path.join(args.save_path, "train.names")
else:
    file_train_txt = os.path.join(args.save_path, args.namespace + ".txt")
    file_train_data = os.path.join(args.save_path, args.namespace + ".data")
    file_train_names = os.path.join(args.save_path, args.namespace + ".names")

labels = []
with open(file_train_txt, "w", encoding="utf-8") as f:
    for xml_name in tqdm.tqdm(os.listdir(annotations_dirs), desc="train"):
        xml_path = os.path.join(annotations_dirs, xml_name)
        name_image = analysis_xml_from_app_labelme(xml_path, jpegImages_dirs, labels) 
        if (name_image == None):
            continue
        else:
            image_path = os.path.join(jpegImages_dirs, name_image + ".jpg")  
            f.write(image_path + "\n")
print("Data generation completed. All classes : {}\n".format(labels))

# train.names
with open(file_train_names, "w", encoding="utf-8") as f:
    for label in labels:
        f.write(label + "\n")
        
# train.data 
with open(file_train_data, "w", encoding="utf-8") as f:
    f.write("classes=" + str(len(labels)) + "\n")
    f.write("train=" + file_train_txt + "\n")
    f.write("valid=" + file_train_txt + "\n")
    f.write("names=" + file_train_names + "\n")