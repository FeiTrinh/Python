import os
from PIL import Image
from xml.dom.minidom import parse
import torch
import numpy as np
import transforms as T
import json

class DatasetGen(object):
    def __init__(self, root, transforms, train=True):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.train = train
        if self.train:
            name = "FudanPed"
        else:
            name = "PennPed"
        self.imgs = list(sorted([_ for _ in  os.listdir(os.path.join(root, "PNGImages")) if name in _] ))
        self.annotations = list(sorted([_ for _ in  os.listdir(os.path.join(root, "PedMasks")) if name in _] ))
        assert len(self.imgs) == len(self.annotations)

    def __getitem__(self, idx):
        # load images
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        
        annotation_path = os.path.join(self.root, "PedMasks", self.annotations[idx])
        mask = Image.open(annotation_path)
        mask = np.array(mask)
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]
        masks = mask == obj_ids[:, None, None]
        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
        labels = torch.ones((num_objs,), dtype=torch.int64)
        
        #import pdb;pdb.set_trace()

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([idx])
        area = torch.as_tensor((boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]), dtype=torch.float32)
        # iscrowd is needed in evaluation, which converts everything into coco datatype
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


def get_transform(horizontal_flip):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    
    if horizontal_flip:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)
