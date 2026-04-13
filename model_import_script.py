from xml_loader import BaseballNN

import torch

PATH = "baseball_model.pt"

model = BaseballNN()

optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

model.load_state_dict(torch.load(PATH))