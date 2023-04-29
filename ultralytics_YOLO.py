#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import torch
from ultralytics import YOLO
import os
import cv2
import numpy as np
from predictors.YOLOv3 import YOLOv3Predictor

colors = [['black', 0, 0, 0], ['white', 255, 255, 255], ['red', 255, 0, 0],
          ['lime color', 0, 255, 0], ['blue', 0, 0, 255], ['yellow', 255, 255, 0],
          ['blue aqua', 0, 255, 255], ['pink', 255, 0, 255], ['burgundy', 128, 0, 0], ['green', 0, 128, 0],
          ['violet', 128, 0, 128], ['turquoise', 0, 128, 128], ['navy blue', 0, 0, 128],
          ['navy orange', 255, 140, 0], ['orange', 255, 69, 0], ['olive', 107, 142, 35],
          ['blue', 70, 130, 180], ['light blue', 173, 216, 30],
          ['light blue', 135, 206, 250], ['violet', 138, 43, 226], ['violet', 75, 0, 130],
          ['navy pink', 255, 20, 147], ['beige', 245, 245, 20], ['brown', 139, 69, 19]]


def get_color(image, x, y):
    b, g, r = image[x, y]
    minimum = 1000
    color_name = 'black'
    for i in range(len(colors)):
        d = abs(colors[i][1] - r) + abs(colors[i][2] - g) + abs(colors[i][3] - b)
        if d <= minimum:
            minimum = d
            color_name = colors[i][0]

    return color_name


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

yolo_modanet_params = {"model_def": "yolo/modanetcfg/yolov3-modanet.cfg",
                       "weights_path": "yolo/weights/yolov3-modanet_last.weights",
                       "class_path": "yolo/modanetcfg/modanet.names",
                       "conf_thres": 0.8,
                       "nms_thres": 0.7,
                       "img_size": 600,
                       "device": device}

# Начало загрузки модели Clothing detection
yolo_params = yolo_modanet_params
# Classes
clothing_classes = ['bag', 'belt', 'boots', 'footwear', 'outer', 'dress', 'sunglasses', 'pants', 'top', 'shorts',
                     'skirt', 'headwear', 'scarf/tie']
detectron = YOLOv3Predictor(params=yolo_params)
# Конец загрузки clothing detection

model = YOLO("yolov8n.pt")
test_path = "input_dir"
videos = os.listdir(test_path)

for file_name in videos:
    path_to_file_name = test_path + '/' + file_name

    results = model(path_to_file_name,
                    conf=0.7, vid_stride=True, device="mps", task='detect', stream=True)

    classes = []
    clothing = []
    color_clothing = []
    for r in results:
        # print(len(r.boxes.cls))
        boxes = r.boxes
        for box in boxes:
            if model.names[int(box.cls)] == "person":
                x_0 = int(box.xyxy[0][0])
                y_0 = int(box.xyxy[0][1])
                x_1 = int(box.xyxy[0][2])
                y_1 = int(box.xyxy[0][3])
                img = r.orig_img[y_0:y_1, x_0:x_1]
                detections = detectron.get_detections(img)
                if len(detections) != 0:
                    detections.sort(reverse=False, key=lambda x: x[4])
                    for x1, y1, x2, y2, cls_conf, cls_pred in detections:
                        cloth_name = clothing_classes[int(cls_pred)]
                        if cloth_name not in clothing:
                            x, y = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                            color_name = get_color(img, y, x)
                            clothing.append(cloth_name)
                            color_clothing.append(color_name)

        if len(r.boxes.cls) != 0:
            cls = r.boxes.cls.cpu().detach().numpy()
            clas = [r.names[int(c)] for c in cls]
            classes.append(clas)

    classes = [item for sublist in classes for item in sublist]
    classes = list(set(classes))
    print(classes)
    ind = 0
    ans = ""
    while ind < len(clothing):
        ans += color_clothing[ind] + " " + clothing[ind] + ", "
        ind += 1
    print(ans[:-2])
