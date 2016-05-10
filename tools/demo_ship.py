#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg, cfg_from_file
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms

from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse

CLASSES = ('__background__',
           'ship')

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'vgg': ('VGG_CNN_M_1024',
                   'VGG_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel')}



def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()

def demo(net, im_file):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image as gray scale
    gim = cv2.imread(im_file, flags= cv2.CV_LOAD_IMAGE_GRAYSCALE)
    # convert to rgb repeated in each channel
    im = cv2.cvtColor(gim, cv2.COLOR_GRAY2BGR)
    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections(im, cls, dets, thresh=CONF_THRESH)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [zf]',
                        choices=NETS.keys(), default='vgg')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)

    args = parser.parse_args()

    return args

if __name__ == '__main__':

    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    prototxt = os.path.join(cfg.ROOT_DIR, 'models', NETS[args.demo_net][0],
                            'faster_rcnn_end2end', 'test_ships.prototxt')
    #caffemodel = os.path.join(cfg.ROOT_DIR, 'output', 'faster_rcnn_end2end',
    #                         'ak47_train', 'zf_faster_rcnn_iter_70000.caffemodel')
    caffemodel = os.path.join(cfg.ROOT_DIR, 'output', 'faster_rcnn_end2end',
                             'ships_train', 'vgg_cnn_m_1024_faster_rcnn_iter_500000.caffemodel')

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you train it?'
                       ).format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    #im = 128 * np.ones((300, 500, 1), dtype=np.uint8)
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)

    print("test positive images")
    #im_test_dir = '/media/tomb/newvol/images/ships/HLD_distro/tanker/tanker'
    im_test_dir = '/home/tomb/images/ships/HLD_distro/tanker/tanker_test'
    #im_test_dir = '/home/tomb/images/ships/HLD_distro/non-ships/non-ship_test'
    posCnt = 0
    negCnt = 0
    for root, dirs, files in os.walk(im_test_dir):
        for name in files:
            nextf = os.path.join(root, name)
            print('detection for ' + nextf)
            demo(net, nextf)
            posCnt += 1

    #im_test_dir = '/media/tomb/newvol/images/ships/HLD_distro/non-ships/waves'

    #for root, dirs, files in os.walk(im_test_dir):
    #    for name in files:
    #        nextf = os.path.join(root, name)
    #        print('detection for ' + nextf)
    #        demo(net, nextf)
    #        negCnt += 1

    print("Tested {0} pos samples, {1} neg samples".format(posCnt, negCnt))

    plt.show()
