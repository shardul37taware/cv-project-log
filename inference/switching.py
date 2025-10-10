from ultralytics import YOLO
import timm
import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
import pandas as pd

#import all the models
shapes_model = YOLO(r"D:\git\cv-project-log\object\models\best67.pt")
target_model = YOLO(r"D:\git\cv-project-log\target\models\ultra_final_final.pt")

disaster_model = timm.create_model("mobilevit_s", pretrained=False, num_classes=4)
disaster_model.load_state_dict(torch.load("D:/git/learning-cv/disaster classification/MobileViT/mobilevit_s_disaster_II.pth", map_location=torch.device('cpu')))
disaster_model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

#class names
shapes_names = ["Triangle", "Square", "Circle"]
target_names = ["Target"]
disaster_names = ["Damage", "Fire", "Flood", "Normal"] 

#flags
shapes_flag = 0
target_flag = 1
disaster_flag = 1

#camera feed
cap = cv2.VideoCapture(0)

def YOLOinf(model, )

#inference
while True:
    ret, frame = cap.read()

