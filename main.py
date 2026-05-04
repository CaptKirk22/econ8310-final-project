# -*- coding: utf-8 -*-
"""
Main 

Handles custom data loading, NN specification, and various accuracy checks

"""

# For reading data
import os
import gc
import numpy as np
from xml.dom import minidom
import matplotlib.pyplot as plt
import matplotlib.patches as patches


from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# For visualizing
#import plotly.express as px
from torchvision.io import read_image
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F

# For model building
import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import FasterRCNN
#from torchvision.models.detection.rpn import AnchorGenerator

# for videos
import cv2 as cv
from ultralytics.utils.metrics import box_iou
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

#define nn

BaseballNN = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

num_classes = 3

in_features = BaseballNN.roi_heads.box_predictor.cls_score.in_features

BaseballNN.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

def collate_fn(batch):
    return tuple(zip(*batch))

if torch.xpu.is_available():
    device = torch.device("xpu")
    print(f"Using device: {torch.xpu.get_device_name(0)}")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using device: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("GPUs not found, using CPU")


params = [p for p in BaseballNN.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(
    BaseballNN.parameters(),
    lr=0.005,
    momentum=0.9,
    weight_decay=0.0005
)

#computing intersection overlap union. Between 0 and 1 where closer to 1 means the box is closer to being properly placed 
def compute_iou(boxA, boxB):
        # boxA, boxB: [x1, y1, x2, y2]
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
        return iou

# attempting 2 epochs
num_epochs = 2
#define image count for naming images later
img_count = 0
end = 0

#TODO: remove later
#os.chdir(r'C:\Users\mackm\Documents\Other\School\UNO\Semester 3\Forecasting\Project Git\econ8310-final-project')

#if continuing from a previous save, continue training where last left off.
if 'iteration.txt' and 'baseball_model.pt' in os.listdir():
    with open("iteration.txt", "r") as f:
     iteration = int(f.read().strip())
     BaseballNN = torch.load('baseball_model.pt')
else: iteration = 0

#defining range for case where training from beginning
#4 videos to be used for training per loop for 13 total loops.
if iteration == 0:
    upper = 13
else:
    #when starting from previous save state, begin at correct index
    upper = 13 - iteration
    start = (iteration +1) * 4 + 1


for iteration in range(upper):
    if end == 0:
        start = 0
    else:
        start = end 

    if iteration == 0:
        end = 5
    else:    
        end = (iteration +1) * 4 + 1
    
    print(f'{start}:{end}')

    #redefine the class every iteration to get the next set of data
    class BaseballVideos(torch.utils.data.Dataset):
        def __init__(self, root=None, transforms=None):
            self.root = root
            self.transforms = transforms
            # load all image files, sorting them to
            # ensure that they are aligned
            if root==None:
                self.vids = list(sorted([os.path.join("Model Data", i) for i in os.listdir(os.path.join(os.path.curdir, "Model Data")) if '.mov' in i]))[start:end]
                self.notes = list(sorted([os.path.join("Model Data", i) for i in os.listdir(os.path.join(os.path.curdir, "Model Data")) if '.xml' in i]))[start:end]
                if len(self.vids)!=len(self.notes):
                    raise RuntimeError("Mismatch of annotation files and video files.\nPlease confirm that you have one annotation file for each video and try again.")
            imgs = []
            notes = []
            index = 0
            for i, k in zip(self.vids, self.notes):
                cap = cv.VideoCapture(i)
                note = minidom.parse(k)
                ret = True
                frame_count = 0
                print(index)
                index += 1
                while ret:
                    ret, frame = cap.read()
                    if ret:
                        frame_count += 1
                        frame = np.moveaxis(frame, -1, 0) # Pivot image so color channels first, then H then W
                        imgs.append(torch.from_numpy(frame).float()/255)
                        canvas_size = list(frame.shape[1:])

                for f in range(frame_count):
                    # Collect all boxes for this frame, regardless of 'moving' attribute
                    frame_i = [j for j in note.getElementsByTagName("box") if int(j.attributes['frame'].value)==f]
                    boxes = []
                    labels = []
                    areas = []
                    movings = []

                    for j in frame_i:
                        attrs = j.getElementsByTagName('attribute')
                        if attrs is not None and len(attrs) > 0 and attrs[0].firstChild is not None:
                            moving = attrs[0].firstChild.data == 'true'
                        else:
                            moving = False  # default if attribute is missing

                        xtl = float(j.attributes['xtl'].value)
                        ytl = float(j.attributes['ytl'].value)
                        xbr = float(j.attributes['xbr'].value)
                        ybr = float(j.attributes['ybr'].value)
                        box = (xtl, ytl, xbr, ybr)

                        if moving:
                            label = 1
                        else:
                            label = 2
                        area = (xbr - xtl) * (ybr - ytl)

                        boxes.append(box)
                        labels.append(label)
                        areas.append(area)
                        movings.append(moving)

                    target = {}
                    if len(boxes) > 0:
                        target["boxes"] = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=canvas_size)
                        target["labels"] = torch.tensor(labels, dtype=torch.int64)
                        target["area"] = torch.tensor(areas, dtype=torch.float32)
                        target["moving"] = torch.tensor(movings, dtype=torch.int64)
                    else:
                        target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
                        target["labels"] = torch.zeros((0,), dtype=torch.int64)
                        target['area'] = torch.zeros((0,), dtype=torch.float32)
                        target['moving'] = torch.zeros((0,), dtype=torch.int64)
                    notes.append(target)
                 
            self.imgs = imgs
            self.notes = notes
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
    #call newly defined class to get dataset object
    dataset = BaseballVideos()


# define training and validation data loaders

    #moving NN to gpu if available otherwise cpu
    BaseballNN.to(device)
    

    alpha = .05 # iou threshold

    total_balls =0

    correct_balls =0
    #on iterations 4, 8, and 13 use the data to produce test results. Train on all other iterations
    if iteration+1 in [4,8,13]:
        #define test data loader
        data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn
        )
        BaseballNN.eval()
        size = len(data_loader.dataset)
        with torch.no_grad():

            for images, targets in data_loader:
                images = [img.to(device) for img in images]
                predictions = BaseballNN(images)
                
                for img, pred, targ in zip(images,predictions, targets):
                    p_labels = pred['labels'].cpu().numpy()
                    t_labels = targ['labels'].cpu().numpy()
                    p_boxes = pred['boxes'].cpu().numpy()
                    t_boxes = targ['boxes'].cpu().numpy()
                    """ found = False
                    for targ in t_boxes:
                        for pred in p_boxes:
                            if box_iou(targ, pred) > alpha:
                                found = True
                                break
                        if found:
                            break
                    if found:
                        correct += 1
                    total += 1"""

                    total_balls += len(t_boxes)

                    img_np = img.detach().cpu().permute(1, 2, 0).numpy()
                    matches= []
                    for i, t_box in enumerate(t_boxes):
                        found_this_ball = False
                        for j, p_box in enumerate(p_boxes):
                            # verify it's the moving vs not moving
                            if p_labels[j] == t_labels[i]:
                                # verify the box is tight enough
                                iou = compute_iou(p_box, t_box)
                                print(iou)
                                if iou > alpha:
                                    found_this_ball = True
                                    matches.append((p_box,t_labels[i],iou))
                                    correct_balls += 1
                                    break 
                        
                    if len(matches) > 0:
                        fig, ax = plt.subplots(1, figsize=(8, 8))
                        ax.imshow(img_np)
                        for box, label, iou in matches:
                            x1, y1, x2, y2 = box
                            rect = patches.Rectangle(
                            (x1, y1),
                            x2 - x1,
                            y2 - y1,
                            linewidth=1,
                            edgecolor="red",
                            facecolor="none"
                            )
                            ax.add_patch(rect)
                            
                        plt.savefig(f'Prediction Images/image{img_count}.png')
                        plt.close(fig)
                    img_count += 1




        #compute percentage of balls accurately classified and placed within the iou threshhold
        accuracy = (correct_balls / total_balls * 100) if total_balls > 0 else 0
        print(f"Ball Detection Accuracy: {accuracy:.2f}%")
        with open('accuracy_log.txt', 'a') as file:
            file.write(f"Iteration: {iteration+1}, Accuracy: {accuracy:.2f}%\n")
                            

        """accuracy = 100.0 * correct / total
        print(f"Detection accuracy: {accuracy:.2f}%")"""

    else: 
        #define training data loader
        data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=collate_fn
        )
        #train for 2 epochs
        for epoch in range(num_epochs):
            BaseballNN.train()
            size = len(data_loader.dataset)
            for batch, (X, label) in enumerate(data_loader):
                pred = BaseballNN(X, label)
                #get loss from predefined loss function from faster R CNN
                loss = sum(loss for loss in pred.values())
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                #report loss values
                if batch % 10 == 0:
                    loss_val, current = loss.item(), (batch + 1) * len(X)
                    print(f"loss: {loss_val:>7f}  [{current:>5d}/{size:>5d}]")
    torch.save(BaseballNN.state_dict(),'baseball_model.pt') #save model after every iteration
    with open('iteration.txt', 'w') as file: #save the iteration to txt file for loading in for training later
        file.write(str(iteration+1))
    print(f"loop {iteration +1} complete")
    #delete temporary data items for memory control
    del dataset, size#, data_loader







#image code for later


#image = read_image("data/PennFudanPed/PNGImages/FudanPed00046.png")
#eval_transform = get_transform(train=False)



#image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)
#image = image[:3, ...]
#pred_labels = [f"pedestrian: {score:.3f}" for label, score in zip(pred["labels"], pred["scores"])]
#pred_boxes = pred["boxes"].long()
#output_image = draw_bounding_boxes(image, pred_boxes, pred_labels, colors="red")

#masks = (pred["masks"] > 0.7).squeeze(1)
#output_image = draw_segmentation_masks(output_image, masks, alpha=0.5, colors="blue")


#plt.figure(figsize=(12, 12))
#plt.imshow(output_image.permute(1, 2, 0))