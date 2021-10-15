"""Run inference with a YOLOv5 model on images, videos, directories, streams

Usage:
    $ python path/to/detect.py --source path/to/img.jpg --weights yolov5s.pt --img 640
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

import numpy as np
from .models.augmentations import letterbox

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

from .models.experimental import attempt_load
from .models.datasets import LoadStreams, LoadImages
from .models.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from .models.plots import colors, plot_one_box
from .models.torch_utils import select_device, load_classifier, time_sync


DEFAULT_WEIGHT_FILE = {'s':FILE.parents[0].as_posix() + '/models/pt/yolov5s.pt',
                       'm':FILE.parents[0].as_posix() + '/models/pt/yolov5m.pt',
                       'l':FILE.parents[0].as_posix() + '/models/pt/yolov5l.pt',
                       'x':FILE.parents[0].as_posix() + '/models/pt/yolov5x.pt'}


class Detector(object):

    def __init__(self,
                 weights = DEFAULT_WEIGHT_FILE['s'],
                 conf_thres = 0.25,
                 iou_thres = 0.45,
                 max_det = 1000,
                 classes = None,
                 augment = False,
                 classify = False,
                 half = False,
                 agnostic_nms = False):

        self.conf_thres = conf_thres  # confidence threshold
        self.iou_thres = iou_thres  # NMS IOU threshold
        self.max_det = max_det  # maximum detections per image
        self.classes = classes
        self.augment = augment
        self.classify = classify
        self.imgsz = 640
        self.half = half
        self.weights = weights
        self.agnostic_nms = agnostic_nms

        self.first_frame = True

        set_logging()
        self.device = select_device()

        self.half &= self.device.type != 'cpu'  # half precision only supported on CUDA



    @torch.no_grad()
    def load_model(self):
        detect_shape = np.shape(self.img_detect)
        self.x_p, self.y_p = 1. / detect_shape[1], 1. / detect_shape[0]

        # Load model
        self.w = self.weights[0] if isinstance(self.weights, list) else self.weights
        self.classify, self.pt, self.onnx = False, self.w.endswith('.pt'), self.w.endswith('.onnx')  # inference type
        self.stride, self.names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
        if self.pt:
            self.model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
            self.stride = int(self.model.stride.max())  # model stride
            self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names  # get class names
            if self.half:
                self.model.half()  # to FP16
            if self.classify:  # second-stage classifier
                self.modelc = load_classifier(name='resnet50', n=2)  # initialize
                self.modelc.load_state_dict(torch.load('resnet50.pt', map_location=self.device)['model']).to(self.device).eval()
            # Run inference
            if self.device.type != 'cpu':
                # imgsz = check_img_size(imgsz, s=stride)  # check image size
                self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once
        elif self.onnx:
            check_requirements(('onnx', 'onnxruntime'))
            import onnxruntime
            self.session = onnxruntime.InferenceSession(self.w, None)

    @torch.no_grad()
    def detect(self, img, fixed = False):
        # self.img_ori = img.copy()
        self.img_detect = img.copy()

        if fixed:
            self.img_detect = cv2.resize(self.img_detect, (640, 480))
        else:
            self.img_detect = cv2.resize(self.img_detect, (self.img_detect.shape[1], int(self.img_detect.shape[1]*0.75)))
            # pass

        if self.first_frame:
            if not fixed:
                self.imgsz = np.shape(self.img_detect)[1]
            self.first_frame = False
            self.load_model()

        # img = cv2.resize(img, (640,480))
        # cv2.imshow('detect_img',img)
        # print(imgsz, np.shape(img)[0])

        self.img_detect = [letterbox(self.img_detect, self.imgsz, auto=True, stride=self.stride)[0]]

        # Stack
        self.img_detect = np.stack(self.img_detect, 0)

        # Convert
        self.img_detect = self.img_detect[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
        self.img_detect = np.ascontiguousarray(self.img_detect)

        if self.pt:
            self.img_detect = torch.from_numpy(self.img_detect).to(self.device)
            self.img_detect = self.img_detect.half() if self.half else self.img_detect.float()  # uint8 to fp16/32
        elif self.onnx:
            self.img_detect = self.img_detect.astype('float32')
        self.img_detect /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(self.img_detect.shape) == 3:
            self.img_detect = self.img_detect[None]  # expand for batch dim

        # Inference
        t1 = time_sync()
        if self.pt:
            self.visualize = False
            pred = self.model(self.img_detect, augment=self.augment, visualize=self.visualize)[0]
        elif self.onnx:
            pred = torch.tensor(self.session.run([self.session.get_outputs()[0].name], {self.session.get_inputs()[0].name: self.img_detect}))

        # NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)
        t2 = time_sync()
        return pred[0]

    def draw_bb(self, img, pred, type_limit = None, line_thickness = 2):

        if type_limit is None:
            type_limit = self.names
        for *xyxy, conf, cls in pred:
            # print(xyxy)
            xyxy = [xyxy[0]*self.x_p, xyxy[1]*self.y_p, xyxy[2]*self.x_p, xyxy[3]*self.y_p]
            # print(xyxy, self.x_p,self.y_p)
            if self.names[int(cls)] in type_limit:
                label = f'{self.names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, img, label=label, color=colors(int(cls), True), line_thickness=line_thickness)



if __name__ == "__main__":
    pass
