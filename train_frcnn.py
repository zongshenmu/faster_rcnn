# ==============================================================================
#
#this code will train on MSCOCO dataset
#dataset include 80 classes
#model based on resnet 50 and weights pretrained on ImageNet
#created by zongshenCS
#date:2018-4-20
#
# ==============================================================================
from __future__ import division

import os
import pickle
import random
import time

import numpy as np
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import generic_utils

import keras_frcnn.roi_helpers as roi_helpers
from keras_frcnn import config, data_generators
from keras_frcnn import losses as losses_fn
from keras_frcnn import resnet as nn
from keras_frcnn.simple_parser import get_data
from utils.Logger import Logger


def train_mscoco():
    # ===========================模型的配置和加载======================================
    # config for data argument
    cfg = config.Config()
    cfg.use_horizontal_flips = True
    cfg.use_vertical_flips = True
    cfg.rot_90 = True
    cfg.num_rois = 32
    #resnet前四卷积部分的权值
    cfg.base_net_weights = nn.get_weight_path()
    #保存模型的权重值
    cfg.model_path = './model/mscoco_frcnn.hdf5'
    #all_images, class_mapping = get_data()
    #加载训练的图片
    train_imgs, class_mapping = get_data('train')


    cfg.class_mapping = class_mapping
    print('Num classes (including bg) = {}'.format(len(class_mapping)))
    #保存所有的配置文件
    with open(cfg.config_save_file, 'wb') as config_f:
        pickle.dump(cfg, config_f)
        print('Config has been written to {}, and can be loaded when testing to ensure correct results'.format(
            cfg.config_save_file))
    #图片随机洗牌
    random.shuffle(train_imgs)
    print('Num train samples {}'.format(len(train_imgs)))
    data_gen_train = data_generators.get_anchor_gt(train_imgs, class_mapping, cfg, nn.get_img_output_length,
                                                   K.image_dim_ordering(), mode='train')
    # ==============================================================================

    # ===============================模型的定义======================================
    #keras内核为tensorflow
    input_shape_img = (None, None, 3)
    img_input = Input(shape=input_shape_img)
    roi_input = Input(shape=(None, 4))
    # define the base resnet50 network
    shared_layers = nn.nn_base(img_input, trainable=False)
    # define the RPN, built on the base layers
    num_anchors = len(cfg.anchor_box_scales) * len(cfg.anchor_box_ratios)
    rpn = nn.rpn(shared_layers, num_anchors)
    classifier = nn.classifier(shared_layers, roi_input, cfg.num_rois, nb_classes=len(class_mapping), trainable=True)
    #model(input=,output=)
    model_rpn = Model(img_input, rpn[:2])
    model_classifier = Model([img_input, roi_input], classifier)
    # this is a model that holds both the RPN and the classifier, used to load/save weights for the models
    model_all = Model([img_input, roi_input], rpn[:2] + classifier)
    # ==============================================================================

    # ===========================基本模型加载ImageNet权值=============================
    try:
        print('loading base model weights from {}'.format(cfg.base_net_weights))
        model_rpn.load_weights(cfg.base_net_weights, by_name=True)
        model_classifier.load_weights(cfg.base_net_weights, by_name=True)
    except Exception as e:
        print('基本模型加载ImageNet权值: ',e)
        print('Could not load pretrained model weights on ImageNet.')
    # ==============================================================================

    # ===============================模型优化========================================
    #在调用model.compile()之前初始化一个优化器对象，然后传入该函数
    optimizer = Adam(lr=1e-5)
    optimizer_classifier = Adam(lr=1e-5)
    model_rpn.compile(optimizer=optimizer,
                      loss=[losses_fn.rpn_loss_cls(num_anchors), losses_fn.rpn_loss_regr(num_anchors)])
    model_classifier.compile(optimizer=optimizer_classifier,
                             loss=[losses_fn.class_loss_cls, losses_fn.class_loss_regr(len(class_mapping) - 1)],
                             metrics={'dense_class_{}'.format(len(class_mapping)): 'accuracy'})
    model_all.compile(optimizer='sgd', loss='mae')
    # ==============================================================================

    # ================================训练、输出设置==================================
    epoch_length = len(train_imgs)
    num_epochs = int(cfg.num_epochs)
    iter_num = 0
    losses = np.zeros((epoch_length, 5))
    rpn_accuracy_rpn_monitor = []
    rpn_accuracy_for_epoch = []
    start_time = time.time()
    best_loss = np.Inf

    logger = Logger(os.path.join('.', 'log.txt'))
    # ==============================================================================

    print('Starting training')
    for epoch_num in range(num_epochs):

        progbar = generic_utils.Progbar(epoch_length)
        logger.write('Epoch {}/{}'.format(epoch_num + 1, num_epochs))

        while True:
            try:
                if len(rpn_accuracy_rpn_monitor) == epoch_length and cfg.verbose:
                    mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor)) / len(rpn_accuracy_rpn_monitor)
                    rpn_accuracy_rpn_monitor = []
                    print(
                        'Average number of overlapping bounding boxes from RPN = {} for {} previous iterations'.format(
                            mean_overlapping_bboxes, epoch_length))
                    if mean_overlapping_bboxes == 0:
                        print('RPN is not producing bounding boxes that overlap'
                              ' the ground truth boxes. Check RPN settings or keep training.')
                #图片，标准的cls、rgr，盒子数据
                X, Y, img_data = next(data_gen_train)

                #训练rpn
                loss_rpn = model_rpn.train_on_batch(X, Y)

                #边训练rpn得到的区域送入roi
                #x_class, x_regr, base_layers
                P_rpn = model_rpn.predict_on_batch(X)

                result = roi_helpers.rpn_to_roi(P_rpn[0], P_rpn[1], cfg, K.image_dim_ordering(), use_regr=True,
                                                overlap_thresh=0.7,
                                                max_boxes=300)
                # note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
                #区域、cls、rgr、iou
                X2, Y1, Y2, IouS = roi_helpers.calc_iou(result, img_data, cfg, class_mapping)

                if X2 is None:
                    rpn_accuracy_rpn_monitor.append(0)
                    rpn_accuracy_for_epoch.append(0)
                    continue

                neg_samples = np.where(Y1[0, :, -1] == 1)
                pos_samples = np.where(Y1[0, :, -1] == 0)

                if len(neg_samples) > 0:
                    neg_samples = neg_samples[0]
                else:
                    neg_samples = []

                if len(pos_samples) > 0:
                    pos_samples = pos_samples[0]
                else:
                    pos_samples = []

                rpn_accuracy_rpn_monitor.append(len(pos_samples))
                rpn_accuracy_for_epoch.append((len(pos_samples)))

                if cfg.num_rois > 1:
                    if len(pos_samples) < cfg.num_rois // 2:
                        selected_pos_samples = pos_samples.tolist()
                    else:
                        selected_pos_samples = np.random.choice(pos_samples, cfg.num_rois // 2, replace=False).tolist()
                    try:
                        selected_neg_samples = np.random.choice(neg_samples, cfg.num_rois - len(selected_pos_samples),
                                                                replace=False).tolist()
                    except:
                        selected_neg_samples = np.random.choice(neg_samples, cfg.num_rois - len(selected_pos_samples),
                                                                replace=True).tolist()

                    sel_samples = selected_pos_samples + selected_neg_samples
                else:
                    # in the extreme case where num_rois = 1, we pick a random pos or neg sample
                    selected_pos_samples = pos_samples.tolist()
                    selected_neg_samples = neg_samples.tolist()
                    if np.random.randint(0, 2):
                        sel_samples = random.choice(neg_samples)
                    else:
                        sel_samples = random.choice(pos_samples)

                #训练classifier
                loss_class = model_classifier.train_on_batch([X, X2[:, sel_samples, :]],
                                                             [Y1[:, sel_samples, :], Y2[:, sel_samples, :]])

                losses[iter_num, 0] = loss_rpn[1]
                losses[iter_num, 1] = loss_rpn[2]

                losses[iter_num, 2] = loss_class[1]
                losses[iter_num, 3] = loss_class[2]
                losses[iter_num, 4] = loss_class[3]

                iter_num += 1

                progbar.update(iter_num,
                               [('rpn_cls', np.mean(losses[:iter_num, 0])), ('rpn_regr', np.mean(losses[:iter_num, 1])),
                                ('detector_cls', np.mean(losses[:iter_num, 2])),
                                ('detector_regr', np.mean(losses[:iter_num, 3]))])

                if iter_num == epoch_length:
                    loss_rpn_cls = np.mean(losses[:, 0])
                    loss_rpn_regr = np.mean(losses[:, 1])
                    loss_class_cls = np.mean(losses[:, 2])
                    loss_class_regr = np.mean(losses[:, 3])
                    class_acc = np.mean(losses[:, 4])

                    mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
                    rpn_accuracy_for_epoch = []

                    if cfg.verbose:
                        logger.write('Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(
                            mean_overlapping_bboxes))
                        logger.write('Classifier accuracy for bounding boxes from RPN: {}'.format(class_acc))
                        logger.write('Loss RPN classifier: {}'.format(loss_rpn_cls))
                        logger.write('Loss RPN regression: {}'.format(loss_rpn_regr))
                        logger.write('Loss Detector classifier: {}'.format(loss_class_cls))
                        logger.write('Loss Detector regression: {}'.format(loss_class_regr))
                        logger.write('Elapsed time: {}'.format(time.time() - start_time))

                    curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
                    iter_num = 0
                    start_time = time.time()

                    if curr_loss < best_loss:
                        if cfg.verbose:
                            logger.write('Total loss decreased from {} to {}, saving weights'.format(best_loss, curr_loss))
                        best_loss = curr_loss
                        model_all.save_weights(cfg.model_path)
                    break

            except Exception as e:
                print('Exception: {}'.format(e))
                # save model
                model_all.save_weights(cfg.model_path)
                continue
    print('Training complete, exiting.')


if __name__ == '__main__':
    train_mscoco()
