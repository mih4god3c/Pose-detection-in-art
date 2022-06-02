import json
from turtle import st
from lib.config import cfg, update_config
from lib.utils.paf_to_pose import paf_to_pose_cpp
from lib.utils.common import Human, BodyPart, CocoPart, CocoColors, CocoPairsRender, draw_humans
from evaluate.coco_eval import get_outputs, handle_paf_and_heat
from lib.network import im_transform
from lib.network.rtpose_vgg import get_model
from collections import OrderedDict
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch
import pylab as plt
import numpy as np
import matplotlib
import argparse
import scipy
import time
import math
import cv2
import os
import re
import sys
import merge
sys.path.append('.')
# from scipy.ndimage.morphology import generate_binary_structure
# from scipy.ndimage.filters import gaussian_filter, maximum_filter


def check_aspect(img_shape):
    aspect_ratio = img_shape[0] / img_shape[1]
    if aspect_ratio <= 0.4:
        return [0, 0.5, 0]
    elif 0.4 < aspect_ratio <= 0.8:
        return [0.5, 0, 0.5]
    elif 1 < aspect_ratio <= 1.5:
        return [0, 0.5, 0.5]
    elif aspect_ratio > 1.5:
        return [0, 0, 0.5]
    else:
        return [0, 0, 0]


def check_portrait(body_parts, human):
    portrait_points = [0, 1, 2, 5, 14, 15, 16, 17]
    b_len = len(body_parts)
    p_len = len(portrait_points)
    ratio = p_len/b_len
    match = 0
    for b in body_parts:
        if b in portrait_points:
            match += 1
    match_ratio = match/b_len
    if match_ratio < 0.5:
        return [1, 0, 1]
    elif 0.8 < match_ratio:
        return [0, 1, 0]
    else:
        return [0, 0, 0]


def check_laying_standing(body_parts, human, body_box):
    lower_body_points = [9, 10, 12, 13]
    start = body_box[0]
    end = body_box[1]
    width = end[0] - start[0]
    height = end[1] - start[1]
    aspect = width/height
    b_len = len(body_parts)
    p_len = len(lower_body_points)

    if any(x in body_parts for x in lower_body_points):
        if 1.5 < aspect:
            return [3, 0, 0]
        elif 1.2 < aspect <= 1.5:
            return [1, 0, 0]
        elif 0.8 < aspect <= 1.2:
            return [1, 0, 1]
        elif 0.4 < aspect <= 0.8:
            return [0, 0, 1]
        elif aspect <= 0.4:
            return [0, 0, 3]
        else:
            return [0, 0, 0]
    else:
        return [0, 0, 0]


def write_to_folder_by_pose(human, img_shape, img, img_raw, period, filename):
    img_cp = np.copy(img)
    img_cp_raw = np.copy(img_raw)
    body_parts = list(human.body_parts.keys())
    num_body_parts = len(body_parts)
    face = human.get_face_box(img_shape[1], img_shape[0])
    upper_body = human.get_upper_body_box(img_shape[1], img_shape[0])
    body_box = human.get_body_box(img_shape[1], img_shape[0])
    if upper_body is None or body_box is None or face is None:
        return None
    start = body_box[0]
    end = body_box[1]
    img_cp = cv2.rectangle(img_cp, start, end, (0, 255, 0), 3)
    img_cp_raw = cv2.rectangle(img_cp_raw, start, end, (0, 255, 0), 3)
    img_cp_raw = img_cp_raw[start[1]:end[1], start[0]:end[0]]

    probabilities = [0, 0, 0]
    probabilities = [
        x + y for (x, y) in zip(probabilities, check_aspect(img_shape))]
    probabilities = [
        x + y for (x, y) in zip(probabilities, check_portrait(body_parts, human))]
    probabilities = [x + y for (x, y) in zip(probabilities,
                                             check_laying_standing(body_parts, human, body_box))]
    ix = np.argmax(probabilities)
    if ix == 0:
        pose = 'laying'
    elif ix == 1:
        pose = 'portrait'
    elif ix == 2:
        pose = 'standing'

    # cv2.imwrite('results/{}/{}/{}'.format(period, pose, filename), img_cp)
    cv2.imwrite('results-raw/{}/{}/{}'.format(period,
                pose, filename), img_cp_raw)


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='experiment configure file name',
                        default='./experiments/vgg19_368x368_sgd.yaml', type=str)
    parser.add_argument('--weight', type=str,
                        default='pose_model.pth')
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    return args


def make_folders(period):
    # crete folder if not exist
    if not os.path.exists('results/' + period):
        os.makedirs('results/' + period)
    if not os.path.exists('results-raw/' + period):
        os.makedirs('results-raw/' + period)
    for p in ['laying', 'portrait', 'standing']:
        if not os.path.exists('results/' + period + '/' + p):
            os.makedirs('results/' + period + '/' + p)
        if not os.path.exists('results-raw/' + period + '/' + p):
            os.makedirs('results-raw/' + period + '/' + p)


def main():
    args = arguments()
    update_config(cfg, args)

    model = get_model('vgg19')
    model.load_state_dict(torch.load(args.weight))
    model = torch.nn.DataParallel(model).cuda()
    model.float()
    model.eval()

    data_folder = 'wikiart'

    print('Pose estimation...')
    for period in os.listdir(data_folder):
        if os.path.isdir(data_folder + '/' + period):
            print(period)
            for filename in os.listdir(data_folder + '/' + period):
                if not (filename.endswith('.jpg') or filename.endswith('.png')):
                    continue

                test_image = data_folder + '/{}/{}'.format(period, filename)
                oriImg = cv2.imread(test_image)  # B,G,R order
                shape_dst = np.min(oriImg.shape[0:2])
                img_shape = np.shape(oriImg)
                # Get results of original image

                with torch.no_grad():
                    paf, heatmap, im_scale = get_outputs(
                        oriImg, model,  'rtpose')
                humans = paf_to_pose_cpp(heatmap, paf, cfg)
                out = draw_humans(oriImg, humans)
                black_img = np.zeros_like(oriImg)
                outRaw = draw_humans(black_img, humans)

                make_folders(period)

                for i, human in enumerate(humans):
                    filename = '{}-{}'.format(i, filename)
                    write_to_folder_by_pose(
                        human, img_shape, out, outRaw, period, filename)


if __name__ == '__main__':
    main()
