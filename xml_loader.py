# For reading data
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# For model building
import torch
import torch.nn as nn

from torch.utils.data import random_split

import os

import xml.etree.ElementTree as ET

import cv2

class CustomBaseballLoader(Dataset):
    def __init__(self, xml_folder):
        self.xml_folder = xml_folder
        self.vid_folder = xml_folder

        self.xml_files = []
        for j in os.listdir(xml_folder):
            if j.endswith('.xml'):
                self.xml_files.append(j)

        self.samples = []

        for xml_file in self.xml_files:
            xml_path = xml_folder + "/" + xml_file
            video_name = xml_file[:-3] + "mov"
            video_path = xml_folder + "/" + video_name

            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            baseball_frames = set()
            tree = ET.parse(xml_path)
            root = tree.getroot()
            for track in root.findall('.//track'):
                label = track.attrib['label']
                track_id = track.attrib['id']
                for box in track.findall('box'):
                    frame_id = int(track.attrib['id'])
                    xtl = float(box.attrib['xtl'])
                    ytl = float(box.attrib['ytl'])
                    xbr = float(box.attrib['xbr'])
                    ybr = float(box.attrib['ybr'])
                    occluded = int(box.attrib.get('occluded', 0))
                    outside = int(box.attrib.get('outside', 0))
                    rotation = float(box.attrib.get('rotation', 0.0))
                    boxes = {'label': label,
                             'box_bound': [xtl, ytl, xbr, ybr],
                             'occluded': occluded,
                             'outside': outside,
                             'rotation': rotation}
                    self.samples.append((video_path, frame_id, 1)) #moving baseball

        for frame_id in range(total_frames):
            if frame_id not in baseball_frames:
                self.samples.append((video_path, frame_id, 0)) #no baseball (no non moving baseballs in this dataset)


    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        video_path, frame_id, label = self.samples[idx]
        label = torch.tensor(label)

        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        cap.release()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (64, 64))
        frame = torch.tensor(frame).float().permute(2, 0, 1) / 255

        return frame, label


class BaseballNN(nn.Module):
    def __init__(self):
        super(BaseballNN, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_model = nn.Sequential(
            nn.LazyLinear(2),
        )

    def forward(self, x):
        x = self.flatten(x)
        output = self.linear_relu_model(x)
        return output

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, label) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if batch % 10 == 0:
            loss_val, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss_val:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, label in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, label).item()
            correct += (pred.argmax(1) == label).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

if __name__ == "__main__":
    xml_folder = "VidsAndXMLs"
    dataset = CustomBaseballLoader(xml_folder)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    model = BaseballNN()
    learning_rate = 1e-2
    epochs = 5
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum = .9)

    for t in range(epochs):
        print(f"Epoch {t+1}\n")
        train_loop(train_loader, model, loss_fn, optimizer)
        test_loop(test_loader, model, loss_fn)

    torch.save(model.state_dict(), "baseball_model.pt")
    print("model saved")
