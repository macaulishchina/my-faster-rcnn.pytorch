# -*- coding: utf-8 -*-
import torch
import numpy as np
from model.config import cfg


def generate_single_anchors(base_size=16,
                            ratios=[0.5, 1, 2],
                            scales=[8, 16, 32]):
    """产生锚点
    Args：
        base_size---基础尺寸
        ratios---矩形长宽比例
        scales---放大比例
    Return:
        [[-h/2, -w/2, h/2, w/2],...],shape=(len(ratios)*len(scales), 4)
    """
    scales = np.array(scales)
    anchors = []
    for area in (base_size * scales)**2:
        for r in ratios:
            h = (area / r)**0.5
            w = r * h
            anchors.append([-w / 2, -h / 2, w / 2, h / 2])
    return torch.from_numpy(np.array(anchors, np.float32))


def generate_all_anchors(single_anchors, batch_size, feature_height,
                         feature_width):
    """生成初始的anchors
    Args：
        batch_size---批次大小
        feature_height---特征高度
        feature_width---特征宽度
    """
    shift_x = np.arange(0, feature_width) * cfg.FEATURE_STRIDE
    shift_y = np.arange(0, feature_height) * cfg.FEATURE_STRIDE
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = torch.from_numpy(
        np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(),
                   shift_y.ravel())).transpose())
    shifts = shifts.contiguous().float()
    # A---anchor个数，K---feature size
    A, K = single_anchors.size(0), shifts.size(0)
    all_anchors = single_anchors.view(1, A, 4) + shifts.view(K, 1, 4)
    all_anchors = all_anchors.view(1, A * K, 4).expand(batch_size, A * K, 4)
    return all_anchors


def anchors_adjusted_on_deltas(anchors, deltas):
    """第一次调整anchors
    Args:
        anchors---输入图片上所有锚点(x_min, y_min, x_max, y_max)
                    size=(batch_size, anchor_num * feature_size, 4)
        deltas---相对锚点归一化后的调整量(dx, dy, dw, dh)
                    size=(batch_size, anchor_num * feature_size, 4)
    """
    batch_size = anchors.size(0)
    deltas = deltas.permute(0, 2, 3, 1).contiguous()
    deltas = deltas.view(batch_size, -1, 4).float()
    widths = anchors[:, :, 2] - anchors[:, :, 0] + 1.0
    heights = anchors[:, :, 3] - anchors[:, :, 1] + 1.0
    adjust_ctr_x = (anchors[:, :, 0] + widths / 2).unsqueeze(2)
    adjust_ctr_y = (anchors[:, :, 1] + heights / 2).unsqueeze(2)
    dx = deltas[:, :, 0::4]  # 中心点X坐标偏移量
    dy = deltas[:, :, 1::4]  # 中心点Y坐标偏移量
    dw = deltas[:, :, 2::4]  # bbox宽度调整
    dh = deltas[:, :, 3::4]  # bbox高度调整
    adjust_ctr_x += dx * widths.unsqueeze(2)
    adjust_ctr_y += dy * heights.unsqueeze(2)
    adjust_w = torch.exp(dw) * widths.unsqueeze(2)
    adjust_h = torch.exp(dh) * heights.unsqueeze(2)

    # x_min
    anchors[:, :, 0::4] = adjust_ctr_x - adjust_w / 2
    # y_min
    anchors[:, :, 1::4] = adjust_ctr_y - adjust_h / 2
    # x_max
    anchors[:, :, 2::4] = adjust_ctr_x + adjust_w / 2
    # y_max
    anchors[:, :, 3::4] = adjust_ctr_y + adjust_h / 2

    return anchors


def anchors_adjusted_on_border(anchors, img_shapes):
    """对超过图片边界的锚点进行调整，使其被图像包含
    Args:
        anchors---经过第一次调整后的anchors
        img_shapes---图片的尺寸信息
    """
    for idx, img_shape in enumerate(img_shapes):
        anchors[idx, :, 0::2].clamp_(0, img_shape[1] - 1)
        anchors[idx, :, 1::2].clamp_(0, img_shape[0] - 1)
    return anchors


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    plt.xlim(-500, 1000)
    plt.ylim(-500, 1000)
    batch_size = 3
    feature_height = 8
    feature_width = 10
    img_shape = np.array((feature_height, feature_width)) * cfg.FEATURE_STRIDE
    anchors = generate_all_anchors(generate_single_anchors(), batch_size,
                                   feature_height, feature_width)

    bbox_adjusts = torch.from_numpy(
        np.random.randn(batch_size, 9 * 4, feature_height, feature_width) / 10)

    anchors_adjusted_on_deltas(anchors, bbox_adjusts)
    img_shapes = torch.from_numpy(np.array(img_shape)).unsqueeze(0).expand(
        batch_size, 2)
    anchors_adjusted_on_border(anchors, img_shapes)
    for group in anchors:
        for anchor in group:
            rect = patches.Rectangle(anchor[0:2],
                                     anchor[2] - anchor[0],
                                     anchor[3] - anchor[1],
                                     linewidth=1,
                                     edgecolor='r',
                                     facecolor='none')
            plt.gca().add_patch(rect)
    plt.show()
