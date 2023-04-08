import torch
import os
from sys import argv
import cv2
from yolo.utils.utils import *
from predictors.YOLOv3 import YOLOv3Predictor
# from predictors.DetectronModels import Predictor
import glob
from tqdm import tqdm
import sys
from datetime import timedelta

def get_saving_frames_durations(cap, saving_fps):
    """Функция, которая возвращает список длительностей, в которые следует сохранять кадры."""
    s = []
    # получаем продолжительность клипа, разделив количество кадров на количество кадров в секунду
    clip_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    # используйте np.arange () для выполнения шагов с плавающей запятой
    for i in np.arange(0, clip_duration, 1 / saving_fps):
        s.append(i)
    return s

def main(video_file):
    
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


#YOLO PARAMS

yolo_modanet_params = {   "model_def" : "yolo/modanetcfg/yolov3-modanet.cfg",
"weights_path" : "yolo/weights/yolov3-modanet_last.weights",
"class_path":"yolo/modanetcfg/modanet.names",
"conf_thres" : 0.5,
"nms_thres" :0.4,
"img_size" : 416,
"device" : device}


#DATASET MODANET
dataset = 'modanet'
print("Начало загрузки модели")
yolo_params = yolo_modanet_params


#Classes
classes = load_classes(yolo_params["class_path"])
#Colors
cmap = plt.get_cmap("rainbow")
colors = np.array([cmap(i) for i in np.linspace(0, 1, 13)])
#np.random.shuffle(colors)

model = 'yolo'

detectron = YOLOv3Predictor(params=yolo_params)
print("Модель загрузилась")

script, path = argv
if not os.path.exists(path):
    print('Img does not exists..')
    print(path)
SAVING_FRAMES_PER_SECOND = 0.01
main(path)

img = cv2.imread("video_to_img/tmp.jpg")

detections = detectron.get_detections(img)


if len(detections) != 0 :
    detections.sort(reverse=False ,key = lambda x:x[4])
    for x1, y1, x2, y2, cls_conf, cls_pred in detections:

            print("%s" % (classes[int(cls_pred)]))
            ans_file = open("../answer.txt", "a")
            ans_file.write("%s" % (classes[int(cls_pred)]) + "\n")
            ans_file.close()




      