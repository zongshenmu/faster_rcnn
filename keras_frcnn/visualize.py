"""
this is common visualize utils to show boxes in detection or tracking,
this file support both cv2 or PIL library, with separately methods
"""
import cv2
import numpy as np
import colorsys

#hsv色系得到不一样的颜色
def _create_unique_color_uchar(tag, hue_step=0.41):
    h, v = (tag * hue_step) % 1, 1. - (int(tag * hue_step) % 4) / 5.
    r, g, b = colorsys.hsv_to_rgb(h, 1., v)
    return int(255 * r), int(255 * g), int(255 * b)

#画边界盒子和标签
def draw_boxes_and_label_on_image_cv2(img, class_label_map, class_boxes_map):
    """
    this method using cv2 to show boxes on image with various class labels
    :param img:
    :param class_label_map: {1: 'Car', 2: 'Pedestrian'}
    :param class_boxes_map: {1: [box1, box2..], 2: [..]}, in every box is [bb_left, bb_top, bb_width, bb_height, prob]
    :return:
    """
    # for c, boxes in class_boxes_map.items():
    #     for box in boxes:
    #         assert len(box) == 5, 'class_boxes_map every item must be [bb_left, bb_top, bb_width, bb_height, prob]'
    #         # checking box order is bb_left, bb_top, bb_width, bb_height
    #         # make sure all box should be int for OpenCV
    #         bb_left = int(box[0])
    #         bb_top = int(box[1])
    #         bb_width = int(box[2])
    #         bb_height = int(box[3])
    #
    #         # 类标签得到不一样的颜色
    #         unique_color = _create_unique_color_uchar(c)
    #         #画边界
    #         cv2.rectangle(img, (bb_left, bb_top), (bb_width, bb_height), unique_color, 2)
    #         #类别+概率
    #         prob = round(box[4], 2)
    #         text_label = '{} {}'.format(class_label_map[c], prob)
    #         text_org = (bb_left, bb_top - 0)
    #         #加文字
    #         cv2.putText(img, text_label, text_org, cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))

    for c,box in enumerate(class_boxes_map):
        assert len(box) == 4, 'class_boxes_map every item must be [bb_left, bb_top, bb_width, bb_height]'
        # checking box order is bb_left, bb_top, bb_width, bb_height
        # make sure all box should be int for OpenCV
        bb_left = int(box[0])
        bb_top = int(box[1])
        bb_width = int(box[2])
        bb_height = int(box[3])

        # 类标签得到不一样的颜色
        unique_color = _create_unique_color_uchar(c)
        # 画边界
        cv2.rectangle(img, (bb_left, bb_top), (bb_width, bb_height), unique_color, 2)
    return img





