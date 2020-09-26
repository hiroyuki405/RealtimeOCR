
# 参考
# https://qiita.com/KTake/items/dbce1e7361fe0f03139a
import sys
sys.path.append("text-detect")
sys.path.append("text-recognition")


import string
import argparse
import torch
import torch.utils.data
import torch.nn.functional as F
from utils import CTCLabelConverter, AttnLabelConverter
from dataset_img import RawDataset, AlignCollate
from model import Model


import time
from PIL import Image

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import cv2
from skimage import io
import numpy as np
import craft_utils
import imgproc
import file_utils
import json
import zipfile
from craft import CRAFT
from collections import OrderedDict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

POINT_INDEX_LEFT_TOP=0
POINT_INDEX_RIGHT_TOP=1
POINT_INDEX_RIGHT_BOTTOM=2
POINT_INDEX_LEFT_BOTTOM=3


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

class Recognition_Option():
    def __init__(self):
        # self.image_folder', required=True, help='path to image_folder which contains text images')
        self.workers=4
        self.batch_size=192
        self.saved_model="text-recognition/TPS-ResNet-BiLSTM-Attn.pth"
        self.batch_max_length=25
        self.imgH=32
        self.imgW=100
        self.rgb=False
        self.character='0123456789abcdefghijklmnopqrstuvwxyz'
        self.sensitive=None
        self.PAD=False
        self.Transformation="TPS"
        self.FeatureExtraction="ResNet"
        self.SequenceModeling="BiLSTM"
        self.Prediction="Attn"
        self.num_fiducial=20
        self.input_channel=1
        self.output_channel=512
        self.hidden_size=256

class TextRecongtion():
    def __init__(self):
        self.opt = Recognition_Option()
        # self.opt = parser.parse_args()
        if self.opt.sensitive:
            self.opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).
        cudnn.benchmark = False
        cudnn.deterministic = True
        self.opt.num_gpu = torch.cuda.device_count()
    
    def load_net(self):
        """ model configuration """
        if 'CTC' in self.opt.Prediction:
            self.converter = CTCLabelConverter(self.opt.character)
        else:
            self.converter = AttnLabelConverter(self.opt.character)
        self.opt.num_class = len(self.converter.character)

        if self.opt.rgb:
            self.opt.input_channel = 3
        print("input channle :{}".format(self.opt.input_channel))
        self.model = Model(self.opt)
        print('model input parameters', self.opt.imgH, self.opt.imgW, self.opt.num_fiducial, self.opt.input_channel, self.opt.output_channel,
            self.opt.hidden_size, self.opt.num_class, self.opt.batch_max_length, self.opt.Transformation, self.opt.FeatureExtraction,
            self.opt.SequenceModeling, self.opt.Prediction)
        self.model = torch.nn.DataParallel(self.model).to(device)

        # load model
        print('loading pretrained model from %s' % self.opt.saved_model)
        state_dict  = torch.load(self.opt.saved_model, map_location=device)
        self.model.load_state_dict(state_dict)

        # prepare data. two demo images from https://github.com/bgshih/crnn#run-demo
        self.AlignCollate_demo = AlignCollate(imgH=self.opt.imgH, imgW=self.opt.imgW, keep_ratio_with_pad=self.opt.PAD)

    def predict(self, image_list):
        if len(image_list) <= 0 :
            return [""]
        demo_data = RawDataset(image_list, opt=self.opt)  # use RawDataset
        demo_loader = torch.utils.data.DataLoader(
            demo_data, batch_size=self.opt.batch_size,
            shuffle=False,
            num_workers=int(self.opt.workers),
            collate_fn=self.AlignCollate_demo, pin_memory=True)

        # predict
        ret = []
        self.model.eval()
        with torch.no_grad():
            for image_tensors, image_path_list in demo_loader:
                batch_size = image_tensors.size(0)
                image = image_tensors.to(device)
                # For max length prediction
                length_for_pred = torch.IntTensor([self.opt.batch_max_length] * batch_size).to(device)
                text_for_pred = torch.LongTensor(batch_size, self.opt.batch_max_length + 1).fill_(0).to(device)

                if 'CTC' in self.opt.Prediction:
                    preds = self.model(image, text_for_pred)

                    # Select max probabilty (greedy decoding) then decode index to character
                    preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                    _, preds_index = preds.max(2)
                    preds_index = preds_index.view(-1)
                    preds_str = self.converter.decode(preds_index.data, preds_size.data)

                else:
                    preds = self.model(image, text_for_pred, is_train=False)

                    # select max probabilty (greedy decoding) then decode index to character
                    _, preds_index = preds.max(2)
                    preds_str = self.converter.decode(preds_index, length_for_pred)


                # log = open(f'./log_demo_result.txt', 'a')
                # dashed_line = '-' * 80
                # head = f'{"image_path":25s}\t{"predicted_labels":25s}\tconfidence score'
                
                # print(f'{dashed_line}\n{head}\n{dashed_line}')
                # log.write(f'{dashed_line}\n{head}\n{dashed_line}\n')

                preds_prob = F.softmax(preds, dim=2)
                preds_max_prob, _ = preds_prob.max(dim=2)
                for img_name, pred, pred_max_prob in zip(image_path_list, preds_str, preds_max_prob):
                    if 'Attn' in self.opt.Prediction:
                        pred_EOS = pred.find('[s]')
                        pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                        pred_max_prob = pred_max_prob[:pred_EOS]

                    # calculate confidence score (= multiply of pred_max_prob)
                    confidence_score = pred_max_prob.cumprod(dim=0)[-1]
                    if confidence_score <= 0.4:
                        pred ="None"
                    ret.append(pred)
                    # ret.append(preds_str)
                    print(f'{img_name:25s}\t{pred:25s}\t{confidence_score:0.4f}')
        return ret

class TextDetector():
    def __init__(self):
        self.trained_model = "text-detect/craft_mlt_25k.pth"
        self.text_threshold=0.7
        self.low_text=0.4
        self.link_threshold=0.4
        self.cuda=True
        self.canvas_size=1280
        self.mag_ratio=1.5
        self.poly=False
        self.show_time=False
        self.refine=False
        self.refine_model="text-detect/craft_refiner_CTW1500.pth"
        # parser =argparse.ArgumentParser(description='CRAFT Text Detection')
        # self.trained_model', default='weights/craft_mlt_25k.pth', type=str, help='pretrained model')
        # self.text_threshold', default=0.7, type=float, help='text confidence threshold')
        # self.low_text', default=0.4, type=float, help='text low-bound score')
        # self.link_threshold', default=0.4, type=float, help='link confidence threshold')
        # self.cuda', default=True, type=str2bool, help='Use cuda for inference')
        # self.canvas_size', default=1280, type=int, help='image size for inference')
        # self.mag_ratio', default=1.5, type=float, help='image magnification ratio')
        # self.poly', default=False, action='store_true', help='enable polygon type')
        # self.show_time', default=False, action='store_true', help='show processing time')
        # self.test_folder', default='/data/', type=str, help='folder path to input images')
        # self.refine', default=False, action='store_true', help='enable link refiner')
        # self.refiner_model', default='weights/craft_refiner_CTW1500.pth', type=str, help='pretrained refiner model')

    def test_net(self, net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):
        t0 = time.time()

        # resize
        img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, self.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=self.mag_ratio)
        ratio_h = ratio_w = 1 / target_ratio

        # preprocessing
        x = imgproc.normalizeMeanVariance(img_resized)
        x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
        x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
        if cuda:
            x = x.cuda()

        # forward pass
        with torch.no_grad():
            y, feature = net(x)

        # make score and link map
        score_text = y[0,:,:,0].cpu().data.numpy()
        score_link = y[0,:,:,1].cpu().data.numpy()

        # refine link
        if refine_net is not None:
            with torch.no_grad():
                y_refiner = refine_net(y, feature)
            score_link = y_refiner[0,:,:,0].cpu().data.numpy()

        t0 = time.time() - t0
        t1 = time.time()

        # Post-processing
        boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

        # coordinate adjustment
        boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
        polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
        for k in range(len(polys)):
            if polys[k] is None: polys[k] = boxes[k]

        t1 = time.time() - t1

        # render results (optional)
        render_img = score_text.copy()
        render_img = np.hstack((render_img, score_link))
        ret_score_text = imgproc.cvt2HeatmapImg(render_img)

        if self.show_time : print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

        return boxes, polys, ret_score_text

    def test_net(self, net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):
        t0 = time.time()
        # resize
        img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, self.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=self.mag_ratio)
        ratio_h = ratio_w = 1 / target_ratio

        # preprocessing
        x = imgproc.normalizeMeanVariance(img_resized)
        x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
        x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
        if cuda:
            x = x.cuda()

        # forward pass
        with torch.no_grad():
            y, feature = net(x)

        # make score and link map
        score_text = y[0,:,:,0].cpu().data.numpy()
        score_link = y[0,:,:,1].cpu().data.numpy()

        # refine link
        if refine_net is not None:
            with torch.no_grad():
                y_refiner = refine_net(y, feature)
            score_link = y_refiner[0,:,:,0].cpu().data.numpy()

        t0 = time.time() - t0
        t1 = time.time()

        # Post-processing
        boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

        # coordinate adjustment
        boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
        polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
        for k in range(len(polys)):
            if polys[k] is None: polys[k] = boxes[k]

        t1 = time.time() - t1

        # render results (optional)
        render_img = score_text.copy()
        render_img = np.hstack((render_img, score_link))
        ret_score_text = imgproc.cvt2HeatmapImg(render_img)

        if self.show_time : print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

        return boxes, polys, ret_score_text

    def let_load(self):
        self.net = CRAFT()     # initialize
        print('Loading weights from checkpoint (' + self.trained_model + ')')
        if self.cuda:
            self.net.load_state_dict(copyStateDict(torch.load(self.trained_model)))
        else:
            self.net.load_state_dict(copyStateDict(torch.load(self.trained_model, map_location='cpu')))

        if self.cuda:
            self.net = self.net.cuda()
            self.net = torch.nn.DataParallel(self.net)
            cudnn.benchmark = False

        self.net.eval()

        self.refine_net = None
        if self.refine:
            from refinenet import RefineNet
            refine_net = RefineNet()
            print('Loading weights of refiner from checkpoint (' + self.refiner_model + ')')
            if self.cuda:
                refine_net.load_state_dict(copyStateDict(torch.load(self.refiner_model)))
                refine_net = refine_net.cuda()
                refine_net = torch.nn.DataParallel(refine_net)
            else:
                refine_net.load_state_dict(copyStateDict(torch.load(self.refiner_model, map_location='cpu')))

            refine_net.eval()
            self.poly = True

        t = time.time()

    def text_detect(self, image):
        bboxes, polys, score_text = self.test_net(self.net, image, self.text_threshold, self.link_threshold, self.low_text, self.cuda, self.poly, self.refine_net)
        # cv2.imwrite(mask_file, score_text)

        # file_utils.saveResult(image_path, image[:,:,::-1], polys, dirname=result_folder)
        return bboxes, polys, score_text
def cv2np(image):
    ''' OpenCV型 -> PIL型 '''
    new_image = image.copy()
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = new_image[:, :, ::-1]
    elif new_image.shape[2] == 4:  # 透過
        new_image = new_image[:, :, [2, 1, 0, 3]]
    new_image = Image.fromarray(new_image)
    ret = np.array(new_image)
    return ret

def np2cv(image):
    ''' PIL型 -> OpenCV型 '''
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = new_image[:, :, ::-1]
    elif new_image.shape[2] == 4:  # 透過
        new_image = new_image[:, :, [2, 1, 0, 3]]
    return new_image

def transform_by4(img, points):
    """ 4点を指定してトリミングする。 """
    # points = sorted(points, key=lambda x:x[1])  # yが小さいもの順に並び替え。
    # top = sorted(points[:2], key=lambda x:x[0])  # 前半二つは四角形の上。xで並び替えると左右も分かる。
    # bottom = sorted(points[2:], key=lambda x:x[0], reverse=True)  # 後半二つは四角形の下。同じくxで並び替え。
    # points = np.array(top + bottom, dtype='float32')  # 分離した二つを再結合。
    # width = max(np.sqrt(((points[0][0]-points[2][0])**2)*2), np.sqrt(((points[1][0]-points[3][0])**2)*2))
    # height = max(np.sqrt(((points[0][1]-points[2][1])**2)*2), np.sqrt(((points[1][1]-points[3][1])**2)*2))
    # print("befor w:{} after h:{}".format(width, height))

    left_top = np.array(points[POINT_INDEX_LEFT_TOP])
    right_top = np.array(points[POINT_INDEX_RIGHT_TOP])
    right_btm = np.array(points[POINT_INDEX_RIGHT_BOTTOM])
    left_btm = np.array(points[POINT_INDEX_LEFT_BOTTOM])
    width = np.linalg.norm(left_top - right_top)
    height = np.linalg.norm(left_top - left_btm)
    points = np.array(points, dtype='float32')
    print("after w:{} after h:{}".format(width, height))
    dst = np.array([
            np.array([0, 0]),
            np.array([width-1, 0]),
            np.array([width-1, height-1]),
            np.array([0, height-1]),
            ], np.float32)

    trans = cv2.getPerspectiveTransform(points, dst)  # 変換前の座標と変換後の座標の対応を渡すと、透視変換行列を作ってくれる。
    # return trans
    return cv2.warpPerspective(img, trans, (int(width), int(height)))  # 透視変換行列を使って切り抜く。

def transform_projective(image, points):
    M = cv2.getPerspectiveTransform

def triming(image, boxes, mode="L", gray=True):
    ret = []
    tmp_img = image.copy()
    tmp_img = cv2.cvtColor(tmp_img, cv2.COLOR_BGR2GRAY)
    for i, box in enumerate(boxes):
        poly = np.array(box).astype(np.int32).reshape((-1))
        poly = poly.reshape(-1, 2)
        img = transform_by4(tmp_img, poly)
        img = Image.fromarray(img, mode)
        ret.append(img)
    return ret

def ocr(img, boxes, texts, verticals=None):
    """ save text detection result one by one
    Args:
        img_file (str): image file name
        img (array): raw image context
        boxes (array): array of result file
            Shape: [num_detections, 4] for BB output / [num_detections, 4] for QUAD output
    Return:
        None
    """
    if len(boxes) <= 0:
        return img

    img = np.array(img)
    for i, box in enumerate(boxes):
        poly = np.array(box).astype(np.int32).reshape((-1))
        strResult = ','.join([str(p) for p in poly]) + '\r\n'
        print(strResult)

        poly = poly.reshape(-1, 2)
        poly_recv = poly.reshape((-1, 1, 2))
        tras = transform_by4(img, poly)
        # cv2.imshow("trim", tras)
        cv2.polylines(img, [poly_recv], True, color=(0, 0, 255), thickness=2)
        ptColor = (0, 255, 255)
        if verticals is not None:
            if verticals[i]:
                ptColor = (255, 0, 0)

        if texts is not None:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5

            # cv2.putText(img, "{}".format(texts), (poly[0][0]+1, poly[0][1]+1), font, font_scale, (0, 0, 0), thickness=1)
            cv2.putText(img, "{}".format(texts[i]), tuple(poly[0]), font, font_scale, (193, 12, 23), thickness=2)
            # cv2.putText(img, "{}".format(texts[i]), tuple(poly[1]), font, font_scale, (0, 255, 255), thickness=1)
            # cv2.putText(img, "{}".format("te2"), tuple(poly[2]), font, font_scale, (0, 255, 255), thickness=1)
            # cv2.putText(img, "{}".format("te3"), tuple(poly[3]), font, font_scale, (0, 255, 255), thickness=1)
    return img
def main():
    text_recognition = TextRecongtion()
    text_recognition.load_net()
    text_detector = TextDetector()
    text_detector.let_load()
    cap = cv2.VideoCapture(-1)
    while True:
        for _ in range(4):
            ret, frame = cap.read()
        np_image = cv2np(frame)
        bboxes, polys, score_text = text_detector.text_detect(np_image)
        trim_images = triming(frame, bboxes)
        label_list = text_recognition.predict(trim_images)
        # label_list=[""]
        ret = ocr(np_image, polys, label_list)
        ret = np2cv(ret)
        cv2.imshow("test", ret)
        cv2.waitKey(1)    

if __name__ == "__main__":
    main()