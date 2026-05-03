# -*- coding: utf-8 -*-

# For reading data
import os
import numpy as np
from xml.dom import minidom

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# For visualizing
import plotly.express as px
from torchvision.io import read_image
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F

# For model building
import torch
import torch.nn as nn

# for videos
import cv2 as cv

class BaseballVideos(torch.utils.data.Dataset):
    def __init__(self, root=None, transforms=None):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        if root==None:
            self.vids = list(sorted([os.path.join("Model Data", i) for i in os.listdir(os.path.join(os.path.curdir, "Model Data")) if '.mov' in i]))[:5]
            self.notes = list(sorted([os.path.join("Model Data", i) for i in os.listdir(os.path.join(os.path.curdir, "Model Data")) if '.xml' in i]))[:5]
            if len(self.vids)!=len(self.notes):
                raise RuntimeError("Mismatch of annotation files and video files.\nPlease confirm that you have one annotation file for each video and try again.")
        imgs = []
        notes = []
        for i, k in zip(self.vids, self.notes):
            cap = cv.VideoCapture(i)
            note = minidom.parse(k)
            ret = True
            frame_count = 0
            while ret:
              ret, frame = cap.read()
              if ret:
                frame_count += 1
                frame = np.moveaxis(frame, -1, 0) # Pivot image so color channels first, then H then W
                imgs.append(torch.from_numpy(frame))
                canvas_size = list(frame.shape[1:])

            for f in range(frame_count):
                frame_i = [j for j in note.getElementsByTagName("box") if int(j.attributes['frame'].value)==f]
                boxes = []
                labels = []
                areas = []
                movings = []

                for j in frame_i:
                    moving = j.getElementsByTagName('attribute')[0].firstChild.data=='true'

                    xtl = float(j.attributes['xtl'].value)
                    ytl = float(j.attributes['ytl'].value)
                    xbr = float(j.attributes['xbr'].value)
                    ybr = float(j.attributes['ybr'].value)
                    box = (xtl, ytl, xbr, ybr)

                    label = 'baseball'
                    area = (xbr - xtl) * (ybr - ytl)

                    boxes.append(box)
                    labels.append(label)
                    areas.append(area)
                    movings.append(moving)
                    # if moving null, say moving?

                target = {}
                target["boxes"] = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=canvas_size)
                target["labels"] = labels
                target["area"] = areas
                target["moving"] = movings

                notes.append(target)
        self.imgs = imgs
        self.notes = notes

            # target = {}
            # target["boxes"] = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(img))
            # target["masks"] = tv_tensors.Mask(masks)
            # target["labels"] = labels
            # target["image_id"] = image_id
            # target["area"] = area
            # target["iscrowd"] = iscrowd
        self.imgs
        self.notes

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):

        img = self.imgs[idx]
        target = self.notes[idx]

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

data = BaseballVideos()

frame = np.moveaxis(np.array(data.__getitem__(35)[0]), 0, -1)

cap = cv.VideoCapture('dusty_1.mov')
ret, frame = cap.read()

#cv2_imshow(frame)


