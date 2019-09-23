# -*- coding: utf-8 -*-
import torch
import numpy as np


def generate_anchors(base_size=16,
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
    for area in (base_size * scales) ** 2:
        for r in ratios:
            h = (area / r) ** 0.5
            w = r * h
            anchors.append([-h / 2, -w / 2, h / 2, w / 2])
    return torch.from_numpy(np.array(anchors, np.float32))


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    anchors = generate_anchors()
    for anchor in anchors:
        print(anchor)
        rect = patches.Rectangle(anchor[0:2], anchor[2] * 2, anchor[3] * 2, linewidth=1, edgecolor='r', facecolor='none')
        plt.gca().add_patch(rect)
    plt.xlim(-500, 500)
    plt.ylim(-500, 500)
    plt.show()
