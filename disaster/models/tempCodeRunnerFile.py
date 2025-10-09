import timm
import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
from collections import Counter

# Load model
model = timm.create_model("mobilevit_s", pretrained=False, num_classes=4)  # change `3` to your number of classes
model.load_state_dict(torch.load("D:/git/learning-cv/disaster classification/MobileViT/mobilevit_s_disaster_II.pth", map_location=torch.device('cpu')))
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])


# Set up video feed (0 = webcam; change to URL or device for drone)
cap = cv2.VideoCapture(0)

class_names = ["Damage", "Fire", "Flood", "Normal"]  # change to your actual class labels

frameCount = 0
predList = []
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess
    input_tensor = transform(frame).unsqueeze(0)  # [1, 3, 256, 256]

    # Inference
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_idx].item()

        predList.append(pred_idx)

        # Apply confidence threshold majority vote
        if frameCount % 5 == 0:
            majorPred = Counter(predList).most_common(1)[0][0]

            if confidence >= 0.5:
                label = class_names[majorPred]
            else:
                label = "Normal"
                pred_idx = class_names.index("Normal")  # Optional: if you want to log it

            predList.clear()


    # Display
    cv2.putText(frame, f"{label} ({confidence:.2f})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Disaster Classification", frame)

    if cv2.waitKey(1) & 0xFF == ord(' '):
        break

    frameCount+=1 

cap.release()
cv2.destroyAllWindows()
