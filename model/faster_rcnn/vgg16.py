#!/usr/bin/env python3
import torch
from torch import nn
import torchvision.models as models


class VGG16(nn.Module):

    def __init__(self, num_classes, class_agnostic=False, pretrained=False, weight_path='data/pretrained/vgg16_caffe.pth'):
        nn.Module.__init__(self)
        self.num_classes = num_classes
        self.class_agnostic = class_agnostic
        self.pretrained = pretrained
        self.weight_path = weight_path
        self._assemble_modules()

    def _assemble_modules(self):
        vgg = models.vgg16()
        if self.pretrained:
            print('Loading weights from %s ...' % self.weight_path)
            vgg.load_state_dict(torch.load(self.weight_path))
        # remove the last maxpool layer as RCNN_base
        self.RCNN_base = vgg.features[:-1]
        # remove the last fc as RCNN_top
        self.RCNN_top = vgg.classifier[:-1]
        del vgg.avgpool

        # classifier fc layer
        self.RCNN_cls_fc = nn.Linear(4096, self.num_classes)
        # bbox predict fc layer
        if self.class_agnostic:
            self.RCNN_bbox_fc = nn.Linear(4096, 4)
        else:
            self.RCNN_bbox_fc = nn.Linear(4096, 4 * self.num_classes)


print(VGG16(1))
