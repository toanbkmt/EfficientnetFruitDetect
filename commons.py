import io
import PIL
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
import torchvision.transforms as transforms
from geffnet import create_model
import copy
from copy import deepcopy
from torch import optim

def get_model():
    #Config load From Model had use Training
    model = create_model('efficientnet_b3', pretrained=True)
    # Set total fruit index :130 +1
    n_class= 131
    model.classifier = nn.Linear(model.classifier.in_features, n_class)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),
                      lr=0.001,momentum=0.9,
                      nesterov=True,
                      weight_decay=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    CHECK_POINT_PATH = 'D:/Learns/check_point/CHECKPOINT_EFFICIENTNET_B3_TRAIN_SGD.pth'
    checkpoint = torch.load(CHECK_POINT_PATH, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    best_model_wts = copy.deepcopy(model.state_dict())
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    best_loss = checkpoint['best_val_loss']
    best_acc = checkpoint['best_val_accuracy']
    model.eval()
    return model

def get_tensor(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)
