import argparse
import os
import sys
from pathlib import Path
import torch
from yolo.utils.utils import *
from predictors.YOLOv3 import YOLOv3Predictor
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.torch_utils import select_device, smart_inference_mode

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
    SAVING_FRAMES_PER_SECOND = 0.1
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
            
            cv2.imwrite(os.path.join(filename, "tmp" + str(count) + ".jpg"), frame) 
            # удалить точку продолжительности из списка, так как эта точка длительности уже сохранена
            try:
                saving_frames_durations.pop(0)
            except IndexError:
                pass
        # увеличить количество кадров
        count += 1



FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.8,  # confidence threshold
        iou_thres=0.4,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        save_crop=False,  # save cropped prediction boxes
        nosave=True,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        line_thickness=3,  # bounding box thickness (pixels)
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):

    print("Начало загрузки модели YOLOv5")
    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    print("Модель YOLOv5 загрузилась")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()

    yolo_modanet_params = {   "model_def" : "yolo/modanetcfg/yolov3-modanet.cfg",
    "weights_path" : "yolo/weights/yolov3-modanet_last.weights",
    "class_path":"yolo/modanetcfg/modanet.names",
    "conf_thres" : 0.5,
    "nms_thres" :0.4,
    "img_size" : 416,
    "device" : device}

    print("Начало загрузки модели Clothing detection")
    yolo_params = yolo_modanet_params
    #Classes
    cloathing_classes = load_classes(yolo_params["class_path"])

    detectron = YOLOv3Predictor(params=yolo_params)
    print("Модель Clothing detection загрузилась")
    print()
      
    for source_img in os.listdir(source):
        if (str(source_img) != ".ipynb_checkpoints"):
            source_img = str(source) + "/" + str(source_img)
            print(source_img)


            path = source_img
            if not os.path.exists(path):
                print('Img does not exists..')
                print(path)
            
            Cloathing_detection_video_to_img(path)

            img = cv2.imread("video_to_img/tmp300.jpg")

            detections = detectron.get_detections(img)

            # Dataloader
            dataset = LoadImages(source_img, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)

            

            # Run inference


            model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
            seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
            ans = []
            for path, im, im0s, vid_cap, s in dataset:
                with dt[0]:
                    im = torch.from_numpy(im).to(model.device)
                    im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                    im /= 255  # 0 - 255 to 0.0 - 1.0
                    if len(im.shape) == 3:
                        im = im[None]  # expand for batch dim

                # Inference
                with dt[1]:
                    visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
                    pred = model(im, augment=augment, visualize=visualize)

                # NMS
                with dt[2]:
                  
                    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

                # Second-stage classifier (optional)
                # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)
                
                # Process predictions
                for i, det in enumerate(pred):  # per image
                    seen += 1
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                        # Print results
                        for c in det[:, 5].unique():
                            n = (det[:, 5] == c).sum()  # detections per class
                            detection_item = f"{n} {names[int(c)]}{'s' * (n > 1)}"[2:]
                            if not ((detection_item[:-1] in ans) or (detection_item + "s" in ans) or (detection_item in ans)):
                                ans.append(detection_item)

            # Print results

            answer = ""
            for _ in ans:
              answer += _ + ", "
            ans_file = open("../answer.txt", "w")
            ans_file.write(answer[:-2] + "\n")
            ans_file.close()
            print(answer[:-2])

            if len(detections) != 0 :
                detections.sort(reverse=False ,key = lambda x:x[4])
                for x1, y1, x2, y2, cls_conf, cls_pred in detections:

                        print("%s" % (cloathing_classes[int(cls_pred)]))
                        x = (x1 + x2)/2
                        y = (y1 + y2)/2
                        print(x, y)
                        ans_file = open("../answer.txt", "a")
                        ans_file.write("%s" % (cloathing_classes[int(cls_pred)]) + "\n")
                        ans_file.close()
            print()

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
