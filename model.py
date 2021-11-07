# 羊咩咩
# happy happy happy
import torch
import torch.nn as nn
import torch.nn.functional as F

import tensorflow as tf
import tf_slim as slim
print()

'''
建议将文件名称命名为 attention
'''

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def combined_static_and_dynamic_shape(tensor):
    """Returns a list containing static and dynamic values for the dimensions.

    Returns a list of static and dynamic values for shape dimensions. This is
    useful to preserve static shapes when available in reshape operation.

    Args:
      tensor: A tensor of any type.

    Returns:
      A list of size tensor.shape.ndims containing integers or a scalar tensor.
    """

    static_tensor_shape = tensor.shape
    dynamic_tensor_shape = tensor.shape
    combined_shape = []
    for index, dim in enumerate(static_tensor_shape):
        if dim is not None:
            combined_shape.append(dim)
        else:
            combined_shape.append(dynamic_tensor_shape[index])
    return combined_shape


def convolutional_block_attention_module(feature_map, index, inner_units_ratio=0.5):
    """
    CBAM: convolution block attention_a module, which is described in "CBAM: Convolutional Block Attention Module"
    If you want to use this module, just plug this module into your network
    :param feature_map : input feature map
    :param index : the index of convolution block attention_a module
    :param inner_units_ratio: output units number of fully connected layer: inner_units_ratio*feature_map_channel
    :return:feature map with channel and spatial attention_a
    """

    with tf.compat.v1.variable_scope("cbam"):
        feature_map_shape = combined_static_and_dynamic_shape(feature_map)
        # channel attention_a
        # print(feature_map_shape)
        channel_avg_weights = tf.nn.avg_pool(
            input=feature_map.cpu().detach().numpy(),
            ksize=[1, feature_map_shape[1], feature_map_shape[2], 1],
            strides=[1, 1, 1, 1],
            padding='VALID'
        )
        channel_max_weights = tf.nn.max_pool(
            input=feature_map.cpu().detach().numpy(),
            ksize=[1, feature_map_shape[1], feature_map_shape[2], 1],
            strides=[1, 1, 1, 1],
            padding='VALID'
        )
        channel_avg_reshape = tf.reshape(channel_avg_weights,
                                         [feature_map_shape[0], 1, feature_map_shape[3]])
        channel_max_reshape = tf.reshape(channel_max_weights,
                                         [feature_map_shape[0], 1, feature_map_shape[3]])
        channel_w_reshape = tf.concat([channel_avg_reshape, channel_max_reshape], axis=1)

        fc_1 = tf.compat.v1.layers.dense(
            inputs=channel_w_reshape,
            units=feature_map_shape[3] * inner_units_ratio,
            name="fc_1",
            activation=tf.nn.relu
        )
        fc_2 = tf.compat.v1.layers.dense(
            inputs=fc_1,
            units=feature_map_shape[3],
            name="fc_2",
            activation=tf.nn.sigmoid
        )
        channel_attention = tf.reduce_sum(fc_2, axis=1, name="channel_attention_sum")
        channel_attention = tf.reshape(channel_attention, shape=[feature_map_shape[0], 1, 1, feature_map_shape[3]])
        feature_map_with_channel_attention = tf.multiply(feature_map.cpu().detach().numpy(), channel_attention)
        # spatial attention_a
        channel_wise_avg_pooling = tf.reduce_mean(feature_map_with_channel_attention, axis=3)
        channel_wise_max_pooling = tf.reduce_max(feature_map_with_channel_attention, axis=3)

        channel_wise_avg_pooling = tf.reshape(channel_wise_avg_pooling,
                                              shape=[feature_map_shape[0], feature_map_shape[1], feature_map_shape[2],
                                                     1])
        channel_wise_max_pooling = tf.reshape(channel_wise_max_pooling,
                                              shape=[feature_map_shape[0], feature_map_shape[1], feature_map_shape[2],
                                                     1])

        channel_wise_pooling = tf.concat([channel_wise_avg_pooling, channel_wise_max_pooling], axis=3)
        spatial_attention = slim.conv2d(
            channel_wise_pooling,
            1,
            [3, 3],
            padding='SAME',
            activation_fn=tf.nn.sigmoid,
            scope="spatial_attention_conv"
        )
        feature_map_with_attention = tf.multiply(feature_map_with_channel_attention, spatial_attention)

        # print(feature_map_with_attention)
        # with tf.Session() as sess:
        #    init = tf.global_variables_initializer()
        #    sess.run(init)
        #    result = sess.run(feature_map_with_attention)

        return feature_map_with_attention


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()

        # (输入通道数，输出通道数卷积核个数，卷积核尺寸，零填充)
        self.backbone_layer1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, padding=1),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                             nn.ReLU(inplace=True))

        self.backbone_layer2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(128, 128, kernel_size=3, padding=1),
                                             nn.ReLU(inplace=True))

        self.backbone_layer3 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, padding=1),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(256, 256, kernel_size=3, padding=1),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(256, 256, kernel_size=3, padding=1),
                                             nn.ReLU(inplace=True))

        self.backbone_layer4 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, padding=1),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(512, 512, kernel_size=3, padding=1),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(512, 512, kernel_size=3, padding=1),
                                             nn.ReLU(inplace=True))

        self.backbone_layer5 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(512, 512, kernel_size=3, padding=1),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(512, 512, kernel_size=3, padding=1),
                                             nn.ReLU(inplace=True))

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        # 定义反卷积上采样层 ASPP层采用3x3conv  rate= 2,4,8,16
        self.d2conv_ReLU = nn.Sequential(nn.Conv2d(512, 128, kernel_size=3, padding=2, dilation=2),
                                         nn.ReLU(inplace=True))
        self.d4conv_ReLU = nn.Sequential(nn.Conv2d(512, 128, kernel_size=3, padding=4, dilation=4),
                                         nn.ReLU(inplace=True))
        self.d8conv_ReLU = nn.Sequential(nn.Conv2d(512, 128, kernel_size=3, padding=8, dilation=8),
                                         nn.ReLU(inplace=True))
        self.d16conv_ReLU = nn.Sequential(nn.Conv2d(512, 128, kernel_size=3, padding=16, dilation=16),
                                          nn.ReLU(inplace=True))
        # 将conv3,conv4,conv5和ASPP层都使用1*1卷积，并调整成conv3的输出尺寸
        self.conv1 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=1), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=1), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=1), nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1), nn.ReLU(inplace=True))
        # 将融合特征图连续应用三个1*1的卷积
        self.predict_layer = nn.Sequential(nn.Conv2d(1024, 512, kernel_size=1),
                                           nn.ReLU(inplace=True),
                                           nn.Conv2d(512, 512, kernel_size=1),
                                           nn.ReLU(inplace=True),
                                           nn.Conv2d(512, 2, kernel_size=1))

    def forward(self, x):
        input_size = x.size()[2:]
        stage1 = self.backbone_layer1(x)
        stage1_maxpool = self.maxpool(stage1)

        stage2 = self.backbone_layer2(stage1_maxpool)
        stage2_maxpool = self.maxpool(stage2)

        stage3 = self.backbone_layer3(stage2_maxpool)
        stage3_maxpool = self.maxpool(stage3)
        tmp_size = stage3.size()[2:]

        stage4 = self.backbone_layer4(stage3_maxpool)
        stage4_maxpool = self.maxpool(stage4)

        #stage4_maxpool = torch.from_numpy(convolutional_block_attention_module(stage4_maxpool, index=0).cpu().numpy()).to(device)

        stage5 = self.backbone_layer5(stage4_maxpool)

        # ASPP
        d2conv_ReLU = self.d2conv_ReLU(stage5)
        d4conv_ReLU = self.d4conv_ReLU(stage5)
        d8conv_ReLU = self.d8conv_ReLU(stage5)
        d16conv_ReLU = self.d16conv_ReLU(stage5)
        # 将4个扩张的tensor按维数1进行拼接（横着拼）
        dilated_conv_concat = torch.cat((d2conv_ReLU, d4conv_ReLU, d8conv_ReLU, d16conv_ReLU), 1)

        # 上采样：conv3,4,5 以及ASPP采用1X1卷积，调整成conv3大小，并使用双线性插值上采样预测BPD
        sconv1 = self.conv1(dilated_conv_concat)
        sconv1 = F.interpolate(sconv1, size=tmp_size, mode='bilinear', align_corners=True)

        sconv2 = self.conv2(stage5)
        sconv2 = F.interpolate(sconv2, size=tmp_size, mode='bilinear', align_corners=True)

        sconv3 = self.conv3(stage4)
        sconv3 = F.interpolate(sconv3, size=tmp_size, mode='bilinear', align_corners=True)

        sconv4 = self.conv4(stage3)
        sconv4 = F.interpolate(sconv4, size=tmp_size, mode='bilinear', align_corners=True)

        sconcat = torch.cat((sconv1, sconv2, sconv3, sconv4), 1)

        pred_flux = self.predict_layer(sconcat)
        pred_flux = F.interpolate(pred_flux, size=input_size, mode='bilinear', align_corners=True)

        return pred_flux







