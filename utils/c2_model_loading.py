#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Caffe2 to PyTorch checkpoint name converting utility."""

import re


def get_name_convert_func():
    """
    Get the function to convert Caffe2 layer names to PyTorch layer names.
    Returns:
        (func): function to convert parameter name from Caffe2 format to PyTorch
        format.
    """
    pairs = [
        # ------------------------------------------------------------


        # 't_res_conv1_bn_b' -> 'fast_bn1.bias'
        [
            r"^t_res_conv1_bn(.*)",
            # r"s\1.pathway0_res\2.branch\3.\4_\5",
            r"fast_bn1.\1",
        ],
        # 'res_conv1_bn_b' -> 'slow_bn1.bias'
        [
            r"^res_conv1_bn(.*)",
            # r"s\1.pathway0_res\2.branch\3.\4_\5",
            r"slow_bn1.\1",
        ],
        # 't_conv1_w' -> 'fast_conv1.weight'
        [
            r"^t_conv1_w",
            # r"s\1.pathway0_res\2.branch\3.\4_\5",
            r"fast_conv1_w",
        ],
        # 'conv1_w' -> 'slow_conv1.weight'
        [
            r"^conv1_w",
            # r"s\1.pathway0_res\2.branch\3.\4_\5",
            r"slow_conv1_w",
        ],
        # ------------------------------------------------------------
        # 'res4_4_branch_2c_bn_b' -> 's4.pathway0_res4.branch2.c_bn_b'
        [
            r"^res([0-9]+)_([0-9]+)_branch([0-9]+)([a-z])_bn(.*)",
            # r"s\1.pathway0_res\2.branch\3.\4_\5",
            r"slow_res\1.\2.bn\3\4.\5",
        ],

        # ------------------------------------------------------------
        # 'res4_4_branch_2c_bn_b' -> 's4.pathway0_res4.branch2.c_bn_b' 'fast_res4.4.bn2c_b'
        [
            r"^t_res([0-9]+)_([0-9]+)_branch([0-9]+)([a-z])_bn(.*)",
            # r"s\1.pathway1_res\2.branch\3.\4_\5",
            r"fast_res\1.\2.bn\3\4.\5",
        ],
        #t_res4_1_branch2b_w  ->  fast_res4.1.conv2.weight
        [r"^t_res([0-9]+)_([0-9]+)_branch([0-9]+)([a-z])_w",
            r"fast_res\1.\2.conv\3\4_w",
         ],
        [r"^res([0-9]+)_([0-9]+)_branch([0-9]+)([a-z])_w",
         r"slow_res\1.\2.conv\3\4_w",
         ],
        # ------------------------------------------------------------
        # '.bn_b' -> '.weight'
        [r"(.*)_b\Z", r"\1bias"],
        # '.bn_s' -> '.weight'
        [r"(.*)_s\Z", r"\1weight"],
        # '_bn_rm' -> '.running_mean'
        [r"(.*)_rm\Z", r"\1running_mean"],
        # '_bn_riv' -> '.running_var'
        [r"(.*)_riv\Z", r"\1running_var"],
        # '_b' -> '.bias'
        # [r"(.*)[\._]b\Z", r"\1bias"],
        # '_w' -> '.weight'
        [r"(.*)[\._]w\Z", r"\1.weight"],
        # 'bn2a' -> '1'
        [r"(.*)bn2a(.*)\Z", r"\1bn1\2"],
        [r"(.*)bn2b(.*)\Z", r"\1bn2\2"],
        [r"(.*)bn2c(.*)\Z", r"\1bn3\2"],
        [r"(.*)conv2a(.*)\Z", r"\1conv1\2"],
        [r"(.*)conv2b(.*)\Z", r"\1conv2\2"],
        [r"(.*)conv2c(.*)\Z", r"\1conv3\2"],
    ]

    def convert_caffe2_name_to_pytorch(caffe2_layer_name):
        """
        Convert the caffe2_layer_name to pytorch format by apply the list of
        regular expressions.
        Args:
            caffe2_layer_name (str): caffe2 layer name.
        Returns:
            (str): pytorch layer name.
        """
        for source, dest in pairs:
            caffe2_layer_name = re.sub(source, dest, caffe2_layer_name)
        return caffe2_layer_name

    return convert_caffe2_name_to_pytorch
