from __future__ import absolute_import


from easydict import EasyDict as edict

cfg = __C = edict()


# Anchor scales for RPN
__C.ANCHOR_SCALES = [8, 16, 32]

# Anchor ratios for RPN
__C.ANCHOR_RATIOS = [0.5, 1, 2]

# Feature stride for RPN
__C.FEATURE_STRIDE = 16

#
# Training options
#
__C.TRAIN = edict()
# NMS threshold used on RPN proposals
__C.TRAIN.RPN_NMS_THRESH = 0.7
# Number of top scoring boxes to keep before apply NMS to RPN proposals
__C.TRAIN.RPN_PRE_NMS_TOP_N = 12000
# Number of top scoring boxes to keep after applying NMS to RPN proposals
__C.TRAIN.RPN_POST_NMS_TOP_N = 2000
# Proposal height and width both need to be greater than RPN_MIN_SIZE (at orig image scale)
__C.TRAIN.RPN_MIN_SIZE = 8


#
# Testing options
#
__C.TEST = edict()
# NMS threshold used on RPN proposals
__C.TEST.RPN_NMS_THRESH = 0.7
# Number of top scoring boxes to keep before apply NMS to RPN proposals
__C.TEST.RPN_PRE_NMS_TOP_N = 6000
# Number of top scoring boxes to keep after applying NMS to RPN proposals
__C.TEST.RPN_POST_NMS_TOP_N = 300
# Proposal height and width both need to be greater than RPN_MIN_SIZE (at orig image scale)
__C.TEST.RPN_MIN_SIZE = 16
