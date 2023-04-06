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

def format_timedelta(td):
    """Служебная функция для классного форматирования объектов timedelta (например, 00:00:20.05)
    исключая микросекунды и сохраняя миллисекунды"""
    result = str(td)
    try:
        result, ms = result.split(".")
    except ValueError:
        return result + ".00".replace(":", "-")
    ms = int(ms)
    ms = round(ms / 1e4)
    return f"{result}.{ms:02}".replace(":", "-")


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
    # создаем папку по названию видео файла
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
            frame_duration_formatted = format_timedelta(timedelta(seconds=frame_duration))
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
yolo_df2_params = {   "model_def" : "yolo/df2cfg/yolov3-df2.cfg",
"weights_path" : "yolo/weights/yolov3-df2_15000.weights",
"class_path":"yolo/df2cfg/df2.names",
"conf_thres" : 0.5,
"nms_thres" :0.4,
"img_size" : 416,
"device" : device}

yolo_modanet_params = {   "model_def" : "yolo/modanetcfg/yolov3-modanet.cfg",
"weights_path" : "yolo/weights/yolov3-modanet_last.weights",
"class_path":"yolo/modanetcfg/modanet.names",
"conf_thres" : 0.5,
"nms_thres" :0.4,
"img_size" : 416,
"device" : device}


#DATASET
dataset = 'modanet'


if dataset == 'df2': #deepfashion2
    yolo_params = yolo_df2_params

if dataset == 'modanet':
    yolo_params = yolo_modanet_params


#Classes
classes = load_classes(yolo_params["class_path"])

#Colors
cmap = plt.get_cmap("rainbow")
colors = np.array([cmap(i) for i in np.linspace(0, 1, 13)])
#np.random.shuffle(colors)

model = 'yolo'

if model == 'yolo':
    detectron = YOLOv3Predictor(params=yolo_params)
else:
    # detectron = Predictor(model=model,dataset= dataset, CATEGORIES = classes)
    print("Что-то пошло не так")

#Faster RCNN / RetinaNet / Mask RCNN



script, path = argv
if not os.path.exists(path):
    print('Img does not exists..')
    print(path)
SAVING_FRAMES_PER_SECOND = 0.01
main(path)

img = cv2.imread("video_to_img/tmp.jpg")

detections = detectron.get_detections(img)
#detections = yolo.get_detections(img)
# print(detections)



#unique_labels = np.array(list(set([det[-1] for det in detections])))

#n_cls_preds = len(unique_labels)
#bbox_colors = colors[:n_cls_preds]


if len(detections) != 0 :
    detections.sort(reverse=False ,key = lambda x:x[4])
    for x1, y1, x2, y2, cls_conf, cls_pred in detections:

            #feat_vec =detectron.compute_features_from_bbox(img,[(x1, y1, x2, y2)])
            #feat_vec = detectron.extract_encoding_features(img)
            #print(feat_vec)
            #print(a.get_field('features')[0].shape)
            print("%s" % (classes[int(cls_pred)]))
            ans_file = open("../answer.txt", "a")
            ans_file.write("%s" % (classes[int(cls_pred)]) + "\n")
            ans_file.close()

            #color = bbox_colors[np.where(unique_labels == cls_pred)[0]][0]
            color = colors[int(cls_pred)]

            color = tuple(c*255 for c in color)
            color = (.7*color[2],.7*color[1],.7*color[0])

            font = cv2.FONT_HERSHEY_SIMPLEX


            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            text =  "%s conf: %.3f" % (classes[int(cls_pred)] ,cls_conf)

            cv2.rectangle(img,(x1,y1) , (x2,y2) , color,3)
            y1 = 0 if y1<0 else y1
            y1_rect = y1-25
            y1_text = y1-5

            if y1_rect<0:
                y1_rect = y1+27
                y1_text = y1+20
            cv2.rectangle(img,(x1-2,y1_rect) , (x1 + int(8.5*len(text)),y1) , color,-1)
            cv2.putText(img,text,(x1,y1_text), font, 0.5,(255,255,255),1,cv2.LINE_AA)




# cv2.imshow('Detections',img)


# img_id = path.split('/')[-1].split('.')[0]
# cv2.imwrite('output/ouput-test_{}_{}_{}.jpg'.format(img_id,model,dataset),img)


# cv2.waitKey(0)