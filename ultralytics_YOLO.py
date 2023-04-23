#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import torch
from ultralytics import YOLO
import os
import cv2
import numpy as np
from predictors.YOLOv3 import YOLOv3Predictor


colors = [['black', 0,0,0], ['white', 255, 255, 255], ['red', 255, 0, 0],
 ['lime color', 0, 255, 0], ['blue', 0, 0, 255], ['yellow', 255, 255, 0],
 ['blue aqua', 0, 255, 255], ['pink', 255, 0, 255], ['burgundy', 128, 0, 0], ['green', 0, 128, 0], 
 ['violet', 128, 0, 128], ['turquoise', 0, 128, 128], ['navy blue', 0, 0, 128],
 ['navy orange', 255, 140, 0], ['orange', 255, 69, 0], ['olive', 107, 142, 35],
 ['blue', 70, 130, 180], ['light blue', 173, 216, 30],
 ['light blue', 135, 206, 250], ['violet', 138, 43, 226], ['violet', 75, 0, 130],
 ['navy pink', 255, 20, 147], ['beige', 245, 245, 20], ['brown', 139, 69, 19]]
def get_color(image, X, Y):
    
    B, G, R = image[X, Y]
    minimum = 1000
    color_name = 'black'
    for i in range(len(colors)):
        d = abs(colors[i][1] - R) + abs(colors[i][2]- G)+ abs(colors[i][3]- B)
        if d <= minimum:
          minimum = d
          color_name = colors[i][0]
    
    return color_name


def get_saving_frames_durations(cap, saving_fps):
    """Функция, которая возвращает список длительностей, в которые следует сохранять кадры."""
    s = []
    # получаем продолжительность клипа, разделив количество кадров на количество кадров в секунду
    clip_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    # используйте np.arange () для выполнения шагов с плавающей запятой
    for i in np.arange(0, clip_duration, 1 / saving_fps):
        s.append(i)
    return s

def Cloathing_detection_video_to_img(video_file):
    SAVING_FRAMES_PER_SECOND = 0.01
    filename = "video_to_img"
    if not os.path.isdir(filename):
        os.mkdir(filename)
    # читать видео файл    
    cap = cv2.VideoCapture(video_file)
    # получить FPS видео
    fps = cap.get(cv2.CAP_PROP_FPS)
    # если SAVING_FRAMES_PER_SECOND выше видео FPS, то установите его на FPS (как максимум)
    saving_frames_per_second = min(fps, SAVING_FRAMES_PER_SECOND)
    # получить список длительностей для сохранения
    saving_frames_durations = get_saving_frames_durations(cap, saving_frames_per_second)
    # запускаем цикл
    count = 0
    while True:
        is_read, frame = cap.read()
        if not is_read:
            # выйти из цикла, если нет фреймов для чтения
            break
        # получаем продолжительность, разделив количество кадров на FPS
        frame_duration = count / fps
        try:
            # получить самую раннюю продолжительность для сохранения
            closest_duration = saving_frames_durations[0]
        except IndexError:
            # список пуст, все кадры длительности сохранены
            break
        if frame_duration >= closest_duration:
            # если ближайшая длительность меньше или равна длительности кадра,
            # затем сохраняем фрейм
            
            cv2.imwrite(os.path.join(filename, "tmp.jpg"), frame) 
            # удалить точку продолжительности из списка, так как эта точка длительности уже сохранена
            try:
                saving_frames_durations.pop(0)
            except IndexError:
                pass
        # увеличить количество кадров
        count += 1


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

yolo_modanet_params = {   "model_def" : "yolo/modanetcfg/yolov3-modanet.cfg",
"weights_path" : "yolo/weights/yolov3-modanet_last.weights",
"class_path":"yolo/modanetcfg/modanet.names",
"conf_thres" : 0.5,
"nms_thres" :0.4,
"img_size" : 886,
"device" : device}

#Начало загрузки модели Clothing detection
yolo_params = yolo_modanet_params
#Classes
cloathing_classes = ['bag', 'belt', 'boots', 'footwear', 'outer', 'dress', 'sunglasses', 'pants', 'top', 'shorts', 'skirt', 'headwear', 'scarf/tie']
detectron = YOLOv3Predictor(params=yolo_params)
#Конец загрузки cloathing detection

model = YOLO("yolov8n.pt")
test_path = "input_dir"
videos = os.listdir(test_path)

for file_name in videos:
    path_to_file_name = test_path + '/' + file_name
    Cloathing_detection_video_to_img(path_to_file_name)

    img = cv2.imread("video_to_img/tmp.jpg") 
    
    results = model(path_to_file_name,
               conf=0.7, vid_stride=True, device="mps", task='detect', stream=True)
    
    classes = []

    for r in results:
        #print(len(r.boxes.cls))
        if len(r.boxes.cls)!=0:
            cls = r.boxes.cls.cpu().detach().numpy()
            clas = [r.names[int(c)] for c in cls]
            classes.append(clas)
            
    classes = [item for sublist in classes for item in sublist]
    classes = list(set(classes))
    print(classes)
    
    detections = detectron.get_detections(img)
    
    if len(detections) != 0 :
        detections.sort(reverse=False ,key = lambda x:x[4])
        for x1, y1, x2, y2, cls_conf, cls_pred in detections:
            x, y = (int((x1 + x2)/2), int((y1 + y2)/2))
            image = cv2.imread("video_to_img/tmp.jpg")
            color_name = get_color(image, x, y)
            print(color_name + " " + "%s" % (cloathing_classes[int(cls_pred)]))

