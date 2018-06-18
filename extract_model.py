import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import re
import _pickle as pickle

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
}

f_alias_map = {
    'conv/weight': 'conv/kernel',
    'bn/bias': 'bn/beta',
    'bn/weight': 'bn/gamma',
    'running_mean': 'moving_mean',
    'running_var': 'moving_variance',
}

alias_map_pre = {
    'bn1': 'bn_1',
    'bn2': 'bn_2',
    'downsample/0': 'shortcut',
    'downsample/1': 'bn_0'
}

alias_map = {
    'running_mean': 'moving_mean',
    'running_var': 'moving_variance',
    'conv1/weight': 'conv_1/kernel',
    'conv2/weight': 'conv_2/kernel',
    'bn_1/bias': 'bn_1/beta',
    'bn_1/weight': 'bn_1/gamma',
    'bn_2/bias': 'bn_2/beta',
    'bn_2/weight': 'bn_2/gamma',
    'layer1': 'conv2',
    'layer2': 'conv3',
    'layer3': 'conv4',
    'layer4': 'conv5',
    '/0/': '_1/',
    '/1/': '_2/',
    'shortcut/weight': 'shortcut/kernel',
    'bn_0/bias': 'bn_0/beta',
    'bn_0/weight': 'bn_0/gamma',
}

pretrained_model = model_zoo.load_url(model_urls['resnet18'])

to_pickle = {}

for key, value in pretrained_model.items():
    key = key.replace(".", "/")
    value = value.data.numpy()

    if key.startswith('fc'):
        continue

    if not key.startswith('layer'):
        key = "conv1/" + re.sub(r"\d", "", key)
        for _f, _r in f_alias_map.items():
            key = key.replace(_f, _r)
    else:
        for _f, _r in alias_map_pre.items():
            key = key.replace(_f, _r)
        for _f, _r in alias_map.items():
            key = key.replace(_f, _r)

    if key in to_pickle:
        print("ERROR duplicate key")
        import sys
        sys.exit(1)
    to_pickle[key] = value

# print(to_pickle)
pickle.dump(to_pickle, open("../image_classify/models/resnet-18.pkl", "wb"))