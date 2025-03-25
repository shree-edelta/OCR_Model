import os
import glob as glob
import matplotlib.pyplot as plt
import numpy as np
import cv2
import requests
import random

seed = 42
np.random.seed(seed)
 
train = True
epochs = 25
 

class_name = ['Button', 'Carousel', 'CheckBox', 'Heading', 'Icon', 'Image', 'InputText', 'Paragraph', 'RadioButton', 'SelectBox', 'TextArea', 'button', 'checkbox', 'image', 'label', 'paragraph', 'radiobutton', 'select', 'textbox','dropdown']
colors = np.random.uniform(0, 255, size=(len(class_name), 3))

def yolo2bbox(bboxes):
    xmin,ymin = bboxes[0]-bboxes[2]/2,bboxes[1]-bboxes[3]/2
    xmax,ymax = bboxes[0]+bboxes[2]/2,bboxes[1]+bboxes[3]/2
    return [xmin,ymin,xmax,ymax]

def plot_box(image, bboxes, labels):
    h,w,_ = image.shape
    for box_num, box in enumerate(bboxes):
        x1, y1, x2, y2 = yolo2bbox(box)
        xmin, ymin, xmax, ymax = int(x1*w), int(y1*h), int(x2*w), int(y2*h)
        width = xmax-xmin
        height = ymax-ymin
        class_name = class_name[int(labels[box_num])]
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color = colors[class_name.index(class_name)],thickness=2)
        font_scale = min(1,max(3,int(w/500)))
        font_thickness = min(2,max(10,int(w/50)))
        
        p1,p2 = (int(xmin),int(ymin)),(int(xmax),int(ymax))
        
        tw,th = cv2.getTextSize(class_name, 0, fontScale=font_scale, thickness=font_thickness)[0]
        p2 = p1[0]+tw,p1[1]+-th-10
        cv2.rectangle(image,p1,p2,colors[class_name.index(class_name)],-1)
        cv2.putText(image,class_name,(xmin+1,ymin-10),cv2.FONT_HERSHEY_SIMPLEX,font_scale,(255,255,255),font_thickness)
        return image

def plot(image_paths,label_paths,num_samples):
    all_training_images = glob.glob(image_paths)
    all_training_labels = glob.glob(label_paths)
    all_training_images.sort()
    all_training_labels.sort()
    
    num_images = len(all_training_images)
    
    plt.figure(figsize=(15,12))
    for i in range(num_samples):
        j = random.randint(0,num_images-1)
        image = cv2.imread(all_training_images[j])
        with open(all_training_labels[j],'r') as f:
            bboxes = []
            labels = []
            label_lines = f.readlines()
            for label_line in label_lines:
                label = label_line[0]
                bbox_string = label_line[2:]
                x_c,y_c,w,h = bbox_string.split(' ')
                x_c,y_c,w,h = float(x_c),float(y_c),float(w),float(h)
                bboxes.append([x_c,y_c,w,h])
                labels.append(label)
        image = plot_box(image,bboxes,labels)
        plt.subplot(1,num_samples,i+1)
        plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
        plt.title('Image '+str(i+1))
        plt.axis('off')
        plt.show()
plot(image_paths="object.v2i.yolov5pytorch/train/images/0bb2725b-88b4-4734-98f1-3d7d860434e9_png_jpg.rf.0f00d0525d192a01d6adb67b0bcb5b80.jpg",label_paths="object.v2i.yolov5pytorch/train/labels/0bb2725b-88b4-4734-98f1-3d7d860434e9_png_jpg.rf.0f00d0525d192a01d6adb67b0bcb5b80.txt",num_samples=1)
                
               