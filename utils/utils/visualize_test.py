#Purely a testing script to visualize the result of splitting the images to onlu contain the strawberries

import cv2
import random

img_path = "/Users/larsmoan/Documents/Datasets/StrawDI_Db1/test/img/8.png"
txt_path = "/Users/larsmoan/Documents/Datasets/StrawDI_Db1/test/annotations/8.txt"

def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def yolo_to_pascal_voc(x_center, y_center, w, h,  image_w, image_h):
    w = w * image_w
    h = h * image_h
    x1 = ((2 * x_center * image_w) - w)/2
    y1 = ((2 * y_center * image_h) - h)/2
    x2 = x1 + w
    y2 = y1 + h
    return [x1, y1, x2, y2]

def visualize(img_path, txt_path):
    #Opening and reading the image
    

    img = cv2.imread(img_path)
    image_w = img.shape[1]
    image_h = img.shape[0]

    boxes = []
    #Opening and reading the txt file
    file = open(txt_path, "r")
    lines = file.readlines()
    for line in lines:
        #Create a list equal to the box
        box = line.split(" ")
        box = box[1:]
        
        #Convert the box to float
        box = [float(i) for i in box]
        print(box)
        #Convert the box to pascal voc format
        box = yolo_to_pascal_voc(box[0], box[1], box[2], box[3], image_w, image_h)
        print(box)

        boxes.append(box)
    
    for box in boxes:
        plot_one_box(box, img, label="Strawberry", color=[0, 255, 0])

    #Showing the image 
    cv2.imshow("Image", img)
    cv2.waitKey(0)


visualize(img_path, txt_path)