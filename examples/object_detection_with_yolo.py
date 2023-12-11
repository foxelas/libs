import warnings
warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument*")

import numpy as np
from PIL import Image, ImageFile
import torch
from numpy import random
from pathlib import Path
from libs.foxutils.utils import core_utils
from libs.foxutils.utils.image_utils import read_open_cv
from libs.yolov7.models.experimental import attempt_load
from libs.yolov7.utils.datasets import letterbox
from libs.yolov7.utils.general import check_img_size, non_max_suppression, scale_coords, increment_path
from libs.yolov7.utils.plots import plot_one_box
from libs.yolov7.utils.torch_utils import select_device, load_classifier, TracedModel
from os.path import join as pathjoin

import logging
logger = logging.getLogger("utils.object_detection")

ImageFile.LOAD_TRUNCATED_IMAGES = True

DEVICE = core_utils.device
MODELS_DIR = pathjoin("D:\\",  "git", "github", "yourdirectoryhere", "custom_models")
WEIGHTS_DIR = pathjoin(MODELS_DIR, "yolov7", "weights")
YOLO_MODEL = pathjoin(WEIGHTS_DIR, "yolov7_training")
RUNS_FOLDER = pathjoin("runs", "detect", "exp")

logger.debug(f"Models dir: {MODELS_DIR}")


def LoadModel(options, device, half, classify=False):
    logger.info(f"Load from {options.weights}")
    model = attempt_load(options.weights, map_location=device)
    stride = int(model.stride.max())
    _ = check_img_size(options.img_size, s=stride)
    if options.trace:
        model = TracedModel(model, device, options.img_size)
    if half:
        model.half()

    if classify:
        modelc = load_classifier(name="resnet101", n=2)  # initialize
        modelc.load_state_dict(torch.load("weights/resnet101.pt", map_location=device)["model"]).to(device).eval()
    else:
        modelc = None

    return model, stride, modelc


class InParams:
    def __init__(self, d=None):
        self.names = None
        self.device = None
        self.half = None
        self.img_size = 640
        self.exist_ok = True
        self.name = "exp"
        self.project = "runs/detect"
        self.weights = None
        self.trace = False
        self.colors = None
        self.stride = 0
        self.augment = False
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.classes = None
        self.agnostic_nms = False
        if d is not None:
            for key, value in d.items():
                setattr(self, key, value)


def load_object_detection_model(save_img=True, save_txt=True, device="cuda"):
    classify = False
    if device == "cuda" or device == "gpu":
        device = select_device("0")
        half = device.type != "cpu"
    elif device == "cpu":
        half = False
    else:
        half = True

    opt = InParams(dict(agnostic_nms=False,
                        augment=False,
                        classes=None,
                        conf_thres=0.25,
                        device=device,
                        exist_ok=True,
                        img_size=640,
                        iou_thres=0.45,
                        name="exp",
                        nosave=False,
                        project="runs/detect",
                        save_conf=False,
                        save_txt=save_txt,
                        source="",
                        trace=False,
                        update=False,
                        view_img=False,
                        save_img=save_img,
                        save_dir="",
                        classify=classify,
                        half=half,
                        stride=0,
                        names=[],
                        colors=[],
                        weights=YOLO_MODEL + ".pt"))

    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)
    opt.save_dir = save_dir

    model, stride, modelc = LoadModel(opt, device, half, classify)
    opt.stride = stride
    logger.info(f"New object detection model loaded from Yolov7 on device {device}. Model type {type(model)}.\n")

    opt.names = model.module.names if hasattr(model, "module") else model.names
    opt.colors = [[random.randint(0, 255) for _ in range(3)] for _ in opt.names]
    if opt.half:
        model(torch.zeros(1, 3, opt.img_size, opt.img_size).to(opt.device).type_as(next(model.parameters())))

    return model, opt


def detect_from_image(imgs, od_model, od_opt, device):
    names = od_opt.names
    colors = od_opt.colors
    od_img = None
    od_dict_list = []
    for img in imgs:
        # Padded resize
        od_img = letterbox(img, od_opt.img_size, od_opt.stride)[0]
        # Convert
        od_img = od_img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        od_img = np.ascontiguousarray(od_img)
        od_img = torch.from_numpy(od_img).to(device)
        od_img = od_img.half() if od_opt.half else od_img.float()
        od_img /= 255.0
        if od_img.ndimension() == 3:
            od_img = od_img.unsqueeze(0)
        od_pred = od_model(od_img, augment=od_opt.augment)[0]
        od_pred = non_max_suppression(od_pred, od_opt.conf_thres, od_opt.iou_thres, classes=od_opt.classes,
                                      agnostic=od_opt.agnostic_nms)

        od_dict = {}
        for i, det in enumerate(od_pred):
            im0 = img.copy()
            if len(det):
                det[:, :4] = scale_coords(od_img.shape[2:], det[:, :4], im0.shape).round()
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()
                    od_dict[names[int(c)]] = int(n)

                for *xyxy, conf, cls in reversed(det):
                    label = f"{names[int(cls)]} {conf:.2f}"
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2)

        if len(od_pred) > 0:
            od_dict_list.append(od_dict)
            od_img = Image.fromarray(im0[:, :, ::-1])

    return od_img, od_dict_list


def detect_objects(img_list):
    od_model, od_opt = load_object_detection_model(save_img=True, save_txt=True, device=DEVICE)
    logger.debug(f"Available classes: {od_opt.names}")

    for x in img_list:
        logger.info(f"Detecting objects in {x}")
        img = read_open_cv(x)
        res_img, od_dict_list = detect_from_image([img], od_model, od_opt, DEVICE)
        res_img.save(pathjoin('runs', x))

    logger.debug(f"Finito!")



