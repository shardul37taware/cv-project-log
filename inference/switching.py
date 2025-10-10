from ultralytics import YOLO
import timm
import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
import pandas as pd

#import all the models
model_shapes = YOLO(r"D:\git\cv-project-log\object\models\best67.pt")
model_target = YOLO(r"D:\git\cv-project-log\target\models\ultra_final_final.pt")

model_disaster = timm.create_model("mobilevit_s", pretrained=False, num_classes=4)
model_disaster.load_state_dict(torch.load("D:/git/learning-cv/disaster classification/MobileViT/mobilevit_s_disaster_II.pth", map_location=torch.device('cpu')))
model_disaster.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

#class names
shapes_names = ["Triangle", "Square", "Circle"]
target_names = ["Target"]
disaster_names = ["Damage", "Fire", "Flood", "Normal"] 