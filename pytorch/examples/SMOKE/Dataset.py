import os
from PIL import Image
from xml.dom.minidom import parse
import torch
import numpy as np
import transforms as T
import json
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET


class DatasetGen(object):
    def __init__(self, root, transforms, train=True):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.train = train
        if self.train:
            pass
        else:
            pass
        self.annotations = os.listdir(os.path.join(self.root,"Annotations"))

    def __getitem__(self, idx):
        label_path =  os.path.join(self.root, "Annotations", self.annotations[idx])
        img_path =  os.path.join(self.root, "images", self.annotations[idx].split('.')[0]+".jpg")
        # load images
        img = Image.open(img_path).convert("RGB")
        
        xml_fp = ET.parse(label_path)
        objs = xml_fp.findall("object")
        num_objs = len(objs)
        size = (num_objs, int(xml_fp.find("size").find("height").text), int(xml_fp.find("size").find("width").text))
        masks = np.zeros(size)
        
        boxes = []
        for i in range(num_objs):
            obj = objs[i]
            (xmin,ymin) = (int(obj.find("bndbox").find("xmin").text),int(obj.find("bndbox").find("ymin").text))
            (xmax,ymax) = (int(obj.find("bndbox").find("xmax").text),int(obj.find("bndbox").find("ymax").text))
            masks[i, ymin:ymax,xmin:xmax] = 1
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
        return len(self.annotations)


def get_transform(horizontal_flip):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    
    if horizontal_flip:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)
