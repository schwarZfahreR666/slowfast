import os
import time
import numpy as np
import torch
from config import params
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from lib.dataset import Test_Dataset
from lib import slowfastnet
from tensorboardX import SummaryWriter
from fvcore.common.file_io import PathManager
from utils.c2_model_loading import get_name_convert_func
import pickle
from collections import OrderedDict

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res




def test(model, val_dataloader):

    model.eval()

    with torch.no_grad():
        for step, (inputs, nums) in enumerate(val_dataloader):
            index_path = 'name2label.txt'
            save_path = 'submission.txt'
            index = {}
            with open(index_path,'r') as f:
                rows = f.readlines()
                for row in rows:
                    label = row.split(': ')[1].replace('\n', '')
                    name = row.split(': ')[0]
                    index[label] = name
            inputs = inputs.cuda()
            outputs = model(inputs)
            maxk = 1

            _, pred = outputs.topk(maxk, 1, True, True)
            res = int(pred[0])
            num = nums[0].split('/')[1]
            res_name = index[str(res)]
            new_row = num + res_name + '\n'
            print(new_row)

            with open(save_path, 'a') as f:
                f.writelines(new_row)


            # measure accuracy and record loss



def c2_normal_to_sub_bn(key, model_keys):
    """
    Convert BN parameters to Sub-BN parameters if model contains Sub-BNs.
    Args:
        key (OrderedDict): source dict of parameters.
        mdoel_key (OrderedDict): target dict of parameters.
    Returns:
        new_sd (OrderedDict): converted dict of parameters.
    """
    if "bn.running_" in key:
        if key in model_keys:
            return key

        new_key = key.replace("bn.running_", "bn.split_bn.running_")
        if new_key in model_keys:
            return new_key
    else:
        return key

def main():
    cudnn.benchmark = False



    test_dataloader = \
        DataLoader(
            Test_Dataset(params['test_dataset'], action='validation', clip_len=params['clip_len'], frame_sample_rate=params['frame_sample_rate']),
            batch_size=1, shuffle=False, num_workers=params['num_workers'])

    model = slowfastnet.resnet50(class_num=params['num_classes'])
    
    if params['pretrained'] is not None and params['from_caffe']:
        # pretrained_dict = torch.load(params['pretrained'], map_location='cpu', encoding='latin1')

        # try:
        #     model_dict = model.module.state_dict()
        # except AttributeError:
        #     model_dict = model.state_dict()
        #     # for key in model_dict:
        #         # print(key,model_dict[key].shape)
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        print("load pretrain model")
        # model_dict.update(pretrained_dict)
        # model.load_state_dict(model_dict)
        with PathManager.open(params['pretrained'], "rb") as f:
            caffe2_checkpoint = pickle.load(f, encoding="latin1")
        state_dict = OrderedDict()
        name_convert_func = get_name_convert_func()
        for key in caffe2_checkpoint["blobs"].keys():
            converted_key = name_convert_func(key)
            converted_key = c2_normal_to_sub_bn(converted_key, model.state_dict())
            if converted_key in model.state_dict():

                c2_blob_shape = caffe2_checkpoint["blobs"][key].shape
                model_blob_shape = model.state_dict()[converted_key].shape

                # expand shape dims if they differ (eg for converting linear to conv params)
                if len(c2_blob_shape) < len(model_blob_shape):
                    c2_blob_shape += (1,) * (
                            len(model_blob_shape) - len(c2_blob_shape)
                    )
                    caffe2_checkpoint["blobs"][key] = np.reshape(
                        caffe2_checkpoint["blobs"][key], c2_blob_shape
                    )
                # Load BN stats to Sub-BN.
                if (
                        len(model_blob_shape) == 1
                        and len(c2_blob_shape) == 1
                        and model_blob_shape[0] > c2_blob_shape[0]
                        and model_blob_shape[0] % c2_blob_shape[0] == 0
                ):
                    caffe2_checkpoint["blobs"][key] = np.concatenate(
                        [caffe2_checkpoint["blobs"][key]]
                        * (model_blob_shape[0] // c2_blob_shape[0])
                    )
                    c2_blob_shape = caffe2_checkpoint["blobs"][key].shape

                if c2_blob_shape == tuple(model_blob_shape):
                    state_dict[converted_key] = torch.tensor(
                        caffe2_checkpoint["blobs"][key]
                    ).clone()
                    print(
                        "{}: {} => {}: {}".format(
                            key,
                            c2_blob_shape,
                            converted_key,
                            tuple(model_blob_shape),
                        )
                    )
                else:

                    print(
                        "!! {}: {} does not match {}: {}".format(
                            key,
                            c2_blob_shape,
                            converted_key,
                            tuple(model_blob_shape),
                        )
                    )
            else:
                if not any(
                        prefix in key for prefix in ["momentum", "lr", "model_iter"]
                ):

                    print(
                        "!! {}: can not be converted, got {}".format(
                            key, converted_key
                        )
                    )
        diff = set(model.state_dict()) - set(state_dict)
        diff = {d for d in diff if "num_batches_tracked" not in d}
        if len(diff) > 0:

            print("Not loaded {}".format(diff))
        print(state_dict.keys())
        model.load_state_dict(state_dict, strict=False)

    elif params['pretrained'] is not None and not params['from_caffe']:
        pretrained_dict = torch.load(params['pretrained'], map_location='cpu', encoding='latin1')

        try:
            model_dict = model.module.state_dict()
        except AttributeError:
            model_dict = model.state_dict()
            # for key in model_dict:
                # print(key,model_dict[key].shape)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        print("load pretrain model")
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    model = model.cuda(params['gpu'][0])
    model = nn.DataParallel(model, device_ids=params['gpu'])  # multi-Gpu

    test(model,test_dataloader)





if __name__ == '__main__':
    main()
