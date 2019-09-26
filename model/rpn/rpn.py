# -*- coding: utf-8 -*-
from __future__ import absolute_import
from torch import nn
from torch.nn import functional as F
from model.config import cfg
from model.rpn.kits import generate_single_anchors
from model.rpn.kits import generate_all_anchors
from model.rpn.kits import anchors_adjusted_on_deltas
from model.rpn.kits import anchors_adjusted_on_border


class RPN(nn.Module):
    """ region proposal network """
    def __init__(self, depth, train=False):
        nn.Module.__init__(self)
        self.depth = depth  # depth of input feature map
        self.train = train
        self.feature_stride = cfg.feature_stride
        self.anchor_scales = cfg.ANCHOR_SCALES  # 矩形框参考大小
        self.anchor_ratios = cfg.ANCHOR_RATIOS  # 矩形框长宽比例
        self.anchor_num = len(self.anchor_scales) * len(self.anchor_ratios)

        # bg/fg分类和bbox回归共用的rpn网络第一层卷积
        self.rpn_base_conv = nn.Conv2d(self.depth, 512, 3, 1, 1, bias=True)

        # bg/fg二分类卷基层,kernel_size = (1,1)
        self.rpn_cls_conv = nn.Conv2d(512, self.anchor_num * 2, 1)

        # bbox偏移量卷基层,kernel_size = (1,1)
        self.rpn_bbox_conv = nn.Conv2d(512, self.anchor_num * 4, 1)

        # proposal layer
        self.rpn_proposal_layer = ProposalLayer(self.feature_stride,
                                                self.anchor_ratios,
                                                self.anchor_scales,
                                                self.feature_stride)

        # anchor target layer
        self.rpn_target_anchor_layer = None

    def forward(self, base_feature):
        # bg/fg分类和bbox回归共用的特征
        rpn_base_feature = F.relu(self.rpn_base_conv(base_feature),
                                  inplace=True)

        # bg/fg分类pipline
        rpn_cls_scores = self.rpn_cls_conv(rpn_base_feature)
        cls_feature_shape = rpn_cls_scores.size()
        # reshape: (batch_size, anchor_num * 2, h, w) -> (batch_size, 2, -1)
        rpn_cls_scores = rpn_cls_scores.view(rpn_cls_scores.size(0), 2, -1)
        rpn_cls_scores = F.softmax(rpn_cls_scores, 1)
        rpn_cls_scores = rpn_cls_scores.view(cls_feature_shape)

        # bbox回归pipline
        rpn_bbox_adjusts = self.rpn_bbox_conv(rpn_base_feature)


class ProposalLayer(nn.Module):
    def __init__(self, feature_stride, scales, ratios, base_feat=16):
        nn.Module.__init__(self)

        self.feature_stride = feature_stride
        self.anchors = generate_single_anchors(feature_stride, ratios, scales)
        self.num_anchors = self.anchors.size(0)

    def forward(self, rpn_cls_scores, rpn_bbox_adjusts, img_shapes, train):
        """
        定义：
            特征(:, :num_anchors, :, :)关联bg，
            特征(:, num_anchors:, :, :)关联fg
        Args:
            rpn_cls_scores---size=(batch_size, num_anchors*2, h, w)
            rpn_bbox_adjusts---size=(batch_size, num_anchors*4, h, w)
            img_shapes---输入图片大小
            train---boolean,是否训练
        """
        batch_size = rpn_cls_scores.size(0)
        rpn_cls_scores = rpn_cls_scores[:, self.num_anchors:, :, :]
        feature_h, feature_w = rpn_cls_scores.size()[2:4]
        all_anchors = generate_all_anchors(self.anchors, batch_size, feature_h,
                                           feature_w)
        anchors_adjusted_on_deltas(all_anchors, rpn_bbox_adjusts)
        anchors_adjusted_on_border(all_anchors, img_shapes)

        cfg_key = 'TRAIN' if train else 'TEST'
        pre_nms_topN = cfg[cfg_key].RPN_PRE_NMS_TOP_N
        post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N
        nms_thresh = cfg[cfg_key].RPN_NMS_THRESH
        min_size = cfg[cfg_key].RPN_MIN_SIZE