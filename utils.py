import random
import math
import torch
import matplotlib.pyplot as plt

def to_2d_tensor(inp):
    inp = torch.Tensor(inp)
    if len(inp.size()) < 2:
        inp = inp.unsqueeze(0)
    return inp

def xywh_to_x1y1x2y2(boxes):
    boxes = to_2d_tensor(boxes)
    boxes[:, 2] += boxes[:, 0] - 1
    boxes[:, 3] += boxes[:, 1] - 1
    return boxes

def x1y1x2y2_to_xywh(boxes):
    boxes = to_2d_tensor(boxes)
    boxes[:, 2] -= boxes[:, 0] - 1
    boxes[:, 3] -= boxes[:, 1] - 1
    return boxes

def crop_boxes(boxes, im_sizes):
    boxes = to_2d_tensor(boxes)
    im_sizes = to_2d_tensor(im_sizes)
    boxes = xywh_to_x1y1x2y2(boxes)
    zero = torch.Tensor([0])
    boxes[:, 0] = torch.max(torch.min(boxes[:, 0], im_sizes[:, 0]), zero)
    boxes[:, 1] = torch.max(torch.min(boxes[:, 1], im_sizes[:, 1]), zero)
    boxes[:, 2] = torch.max(torch.min(boxes[:, 2], im_sizes[:, 0]), zero)
    boxes[:, 3] = torch.max(torch.min(boxes[:, 3], im_sizes[:, 1]), zero)
    boxes = x1y1x2y2_to_xywh(boxes)
    return boxes

def box_transform(boxes, im_sizes):
    # box in (x, y, w, h) format
    boxes = to_2d_tensor(boxes)
    im_sizes = to_2d_tensor(im_sizes)
    boxes[:, 0] = 2 * boxes[:, 0] / im_sizes[:, 0] - 1
    boxes[:, 1] = 2 * boxes[:, 1] / im_sizes[:, 1] - 1
    boxes[:, 2] = 2 * boxes[:, 2] / im_sizes[:, 0]
    boxes[:, 3] = 2 * boxes[:, 3] / im_sizes[:, 1]
    return boxes

def box_transform_inv(boxes, im_sizes):
    # box in (x, y, w, h) format
    boxes = to_2d_tensor(boxes)
    im_sizes = to_2d_tensor(im_sizes)
    boxes[:, 0] = (boxes[:, 0] + 1) / 2 * im_sizes[:, 0]
    boxes[:, 1] = (boxes[:, 1] + 1) / 2 * im_sizes[:, 1]
    boxes[:, 2] = boxes[:, 2] / 2 * im_sizes[:, 0]
    boxes[:, 3] = boxes[:, 3] / 2 * im_sizes[:, 1]
    return boxes

def compute_IoU(boxes1, boxes2):
    boxes1 = to_2d_tensor(boxes1)
    boxes1 = xywh_to_x1y1x2y2(boxes1)
    boxes2 = to_2d_tensor(boxes2)
    boxes2 = xywh_to_x1y1x2y2(boxes2)

    intersec = boxes1.clone()
    intersec[:, 0] = torch.max(boxes1[:, 0], boxes2[:, 0])
    intersec[:, 1] = torch.max(boxes1[:, 1], boxes2[:, 1])
    intersec[:, 2] = torch.min(boxes1[:, 2], boxes2[:, 2])
    intersec[:, 3] = torch.min(boxes1[:, 3], boxes2[:, 3])

    def compute_area(boxes):
        # in (x1, y1, x2, y2) format
        dx = boxes[:, 2] - boxes[:, 0]
        dx[dx < 0] = 0
        dy = boxes[:, 3] - boxes[:, 1]
        dy[dy < 0] = 0
        return dx * dy

    a1 = compute_area(boxes1)
    a2 = compute_area(boxes2)
    ia = compute_area(intersec)
    assert((a1 + a2 - ia <= 0).sum() == 0)

    return ia / (a1 + a2 - ia)

def compute_acc(preds, targets, im_sizes, theta=0.75):
    preds = box_transform_inv(preds.clone(), im_sizes)
    preds = crop_boxes(preds, im_sizes)
    targets = box_transform_inv(targets.clone(), im_sizes)
    IoU = compute_IoU(preds, targets)
    corr = (IoU >= theta).sum()
    return corr.item() / preds.size(0)

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

def imshow(img, gt_box, pred_box=None):
    plt.imshow(img)

    def draw_box(box, color='green'):
        x, y, w, h = box_transform_inv(box, img.shape[:2][::-1])[0]
        if x == 0:
            x = 1
        if y == 0:
            y = 1
        plt.gca().add_patch(
            plt.Rectangle((x, y), w, h,
                          fill=False, edgecolor=color, linewidth=2, alpha=0.5)
        )

    draw_box(gt_box)
    if pred_box is not None:
        draw_box(pred_box, 'red')

def split(ratio):
    with open('data/images.txt') as f:
        lines = f.read().splitlines()
    class_groups = dict()
    for line in lines:
        value, line = line.split(' ', 1)
        key = line.split('.', 1)[0]
        value = value
        if key in class_groups:
            class_groups[key].append(value)
        else:
            class_groups[key] = [value]

    test_id = []
    for _, group in class_groups.items():
        test_id.extend(random.sample(group, int(math.ceil(len(group)*ratio))))
    train_id = [i for i in map(str, range(1, len(lines)+1)) if i not in test_id]

    return train_id, test_id