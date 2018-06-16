import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
from torch.autograd import Variable
from PIL import Image
import progressbar

from Resnet import resnet
from Dataset import CUBDataset
from utils import *

GPU_ONLY = True
CPU_MODE = False if torch.cuda.is_available() else True # Determine MODE: CUDA / GPU
CUDA_OR_GPU = lambda x: x.cpu() if CPU_MODE else x.cuda()
if not GPU_ONLY:
    plt.ion()

datasets = load_data()

# Training

dataloaders = {split: torch.utils.data.DataLoader(
                datasets[split], batch_size=32,shuffle=(split=='train'),
                num_workers=2, pin_memory=True) for split in ('train', 'test')}

# construct model
model = resnet()
model = CUDA_OR_GPU(model)
criterion = CUDA_OR_GPU(nn.SmoothL1Loss())

fine_tune_layers = set(model.fc.parameters())
fine_tune_layers |= set(model.layer4.parameters())
fine_tune_layers |= set(model.layer3.parameters())

pretrained_layers = set(model.parameters()) - fine_tune_layers

optimizer = torch.optim.Adam([
    {'params': list(pretrained_layers), 'lr': 1e-4},
    {'params': list(fine_tune_layers), 'lr': 1e-3},
], lr=1e-3)
scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

best_model_state = model.state_dict()
best_epoch = -1
best_acc = 0.0

epoch_loss = {'train': [], 'test': []}
epoch_acc = {'train': [], 'test': []}
epochs = 20
for epoch in range(epochs):
    if epoch == 5:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    accs = AverageMeter()
    losses = AverageMeter()
    for phase in ('train', 'test'):
        if phase == 'train':
            scheduler.step()
            model.train(True)
        else:
            model.train(False)

        end = time.time()
        bar = progressbar.ProgressBar()
        for ims, boxes, im_sizes in bar(dataloaders[phase]):
            boxes = crop_boxes(boxes, im_sizes)
            boxes = box_transform(boxes, im_sizes)

            inputs = Variable(CUDA_OR_GPU(ims))
            targets = Variable(CUDA_OR_GPU(boxes))

            optimizer.zero_grad()

            # forward
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            acc = compute_acc(outputs.data.cpu(), targets.data.cpu(), im_sizes)

            nsample = inputs.size(0)
            accs.update(acc, nsample)
            losses.update(loss.data.item(), nsample)

            if phase == 'train':
                loss.backward()
                optimizer.step()

        if phase == 'test' and accs.avg > best_acc:
            best_acc = accs.avg
            best_epoch = epoch
            best_model_state = model.state_dict()

        elapsed_time = time.time() - end
        print('[{}]\tEpoch: {}/{}\tLoss: {:.4f}\tAcc: {:.2%}\tTime: {:.3f}'.format(
            phase, epoch+1, epochs, losses.avg, accs.avg, elapsed_time))
        epoch_loss[phase].append(losses.avg)
        epoch_acc[phase].append(accs.avg)

    print('[Info] best test acc: {:.2%} at {}th epoch'.format(best_acc, best_epoch))
    torch.save(best_model_state, 'best_model_state.path.tar')

if not GPU_ONLY:
    # Plot statistics
    plt.figure(figsize=(15, 10))
    for phase in ('train', 'test'):
        plt.plot(range(len(epoch_loss[phase])), epoch_loss[phase], label=(phase + '_loss'))
        plt.plot(range(len(epoch_acc[phase])), epoch_acc[phase], label=(phase + '_acc'))
    plt.legend(prop={'size': 15})

    # Visualize predicting result
    model.load_state_dict(best_model_state)
    model = model.cpu()

    ind = random.choice(range(len(datasets['test'])))
    im, box, im_size = datasets['test'][ind]
    path, _ = datasets['test'].imgs[ind]
    box = box_transform(box, im_size)[0]

    pred_box = model(Variable(im.unsqueeze(0))).data[0]

    ori_im = np.array(Image.open(path))

    inp = im.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    imshow(ori_im, box, pred_box)
