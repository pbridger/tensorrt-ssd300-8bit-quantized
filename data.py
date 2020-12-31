'adapted from https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Detection/SSD/src/utils.py'

import os
import math
import json, pickle, bz2
import itertools
from coco import COCO

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms


def init_dboxes(model_dtype=torch.float32):
    fig_size = 300
    feat_size = [38, 19, 10, 5, 3, 1]
    steps = [8, 16, 32, 64, 100, 300]
    scales = [21, 45, 99, 153, 207, 261, 315]
    aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]

    fk = fig_size / torch.tensor(steps).float()

    dboxes = []
    # size of feature and number of feature
    for idx, sfeat in enumerate(feat_size):
        sk1 = scales[idx] / fig_size
        sk2 = scales[idx + 1] / fig_size
        sk3 = math.sqrt(sk1 * sk2)
        all_sizes = [(sk1, sk1), (sk3, sk3)]

        for alpha in aspect_ratios[idx]:
            w, h = sk1 * math.sqrt(alpha), sk1 / math.sqrt(alpha)
            all_sizes.append((w, h))
            all_sizes.append((h, w))

        for w, h in all_sizes:
            for i, j in itertools.product(range(sfeat), repeat=2):
                cx, cy = (j + 0.5) / fk[idx], (i + 0.5) / fk[idx]
                dboxes.append((cx, cy, w, h))

    return torch.tensor(
        dboxes,
        dtype=model_dtype,
        device='cuda'
    ).clamp(0, 1)


class COCODetection(Dataset):
    def __init__(self, img_folder, annotate_file, transform):
        self.img_folder = img_folder
        self.annotate_file = annotate_file

        # Start processing annotation
        with open(annotate_file) as fin:
            self.data = json.load(fin)

        self.images = {}

        self.label_map = {}
        self.label_info = {}
        # 0 stand for the background
        cnt = 0
        self.label_info[cnt] = "background"
        for cat in self.data["categories"]:
            cnt += 1
            self.label_map[cat["id"]] = cnt
            self.label_info[cnt] = cat["name"]

        # build inference for images
        for img in self.data["images"]:
            img_id = img["id"]
            img_name = img["file_name"]
            img_size = (img["height"],img["width"])
            if img_id in self.images: raise Exception("dulpicated image record")
            self.images[img_id] = (img_name, img_size, [])

        # read bboxes
        for bboxes in self.data["annotations"]:
            img_id = bboxes["image_id"]
            category_id = bboxes["category_id"]
            bbox = bboxes["bbox"]
            bbox_label = self.label_map[bboxes["category_id"]]
            self.images[img_id][2].append((bbox, bbox_label))

        for k, v in list(self.images.items()):
            if len(v[2]) == 0:
                self.images.pop(k)

        self.img_keys = list(self.images.keys())
        self.transform = transform

    @property
    def labelnum(self):
        return len(self.label_info)

    @staticmethod
    def load(pklfile):
        with bz2.open(pklfile, "rb") as fin:
            ret = pickle.load(fin)
        return ret

    def save(self, pklfile):
        with bz2.open(pklfile, "wb") as fout:
            pickle.dump(self, fout)


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_id = self.img_keys[idx]
        img_data = self.images[img_id]
        fn = img_data[0]
        img_path = os.path.join(self.img_folder, fn)
        img = Image.open(img_path).convert("RGB")

        htot, wtot = img_data[1]
        bbox_sizes = []
        bbox_labels = []

        #for (xc, yc, w, h), bbox_label in img_data[2]:
        for (l,t,w,h), bbox_label in img_data[2]:
            r = l + w
            b = t + h
            #l, t, r, b = xc - 0.5*w, yc - 0.5*h, xc + 0.5*w, yc + 0.5*h
            bbox_size = (l/wtot, t/htot, r/wtot, b/htot)
            bbox_sizes.append(bbox_size)
            bbox_labels.append(bbox_label)

        bbox_sizes = torch.tensor(bbox_sizes)
        bbox_labels =  torch.tensor(bbox_labels)

        max_num = 200
        bbox_out = torch.zeros(max_num, 4)
        label_out =  torch.zeros(max_num, dtype=torch.long)
        bbox_out[:bbox_sizes.size(0), :] = bbox_sizes
        label_out[:bbox_labels.size(0)] = bbox_labels

        img = self.transform(img)
        return img, img_id, (htot, wtot), bbox_out, label_out



def get_val_dataloader(args):
    transformer = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = COCODetection(
        os.path.join(args.data, "val2017"),
        os.path.join(args.data, "annotations/instances_val2017.json"),
        transformer
    )

    return DataLoader(
        dataset,
        batch_size=args.batch_dim,
        shuffle=False,  # Note: distributed sampler is shuffled :(
        sampler=None,
        num_workers=args.num_workers
    )


def get_coco_ground_truth(args):
    return COCO(annotation_file=os.path.join(args.data, "annotations/instances_val2017.json"))

