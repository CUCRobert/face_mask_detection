import cv2
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from facenet_pytorch import MTCNN
from albumentations.pytorch import ToTensorV2
from albumentations import (
     Normalize, Cutout, CoarseDropout, ShiftScaleRotate, CenterCrop, Resize
)

cam = cv2.VideoCapture(0)
NUMBER_OF_CLASSES = 2

#face net for mtcnn
mtcnn = MTCNN(image_size = (480,640) )


#data transforms
data_transforms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(100,),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

#masknet
model_ft = models.resnet18(pretrained = False)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, NUMBER_OF_CLASSES)
model_ft.load_state_dict(torch.load("./mask"+str(NUMBER_OF_CLASSES)+".pth"))
# model_ft = model_ft.to("cuda")
model_ft.eval()

while True:
    ret,frame = cam.read()
    # print(frame.shape)
    # exit()

    
    boxes,probs = mtcnn.detect(frame,landmarks = False)
    
    
    image_cropped = None
    
    if probs[0] != None:
        if len(boxes) >=1 :
            try:
                print(boxes,probs)
                y1 = int(boxes[0][0])
                x1 = int(boxes[0][1])
                y2 = int(boxes[0][2])
                x2 = int(boxes[0][3])
                image_cropped = frame[x1:x2,y1:y2]
                cv2.imshow("croopped",image_cropped)
                image_cropped = cv2.resize(frame,(100,100))
                img = image_cropped.copy()
                img = img[:, :, ::-1]
                img = data_transforms(img)
                print(img.shape)
                output = model_ft(img.unsqueeze(0))
                _, preds = torch.max(output, 1)
                print(preds)
                if preds[0] == 0:
                    frame = cv2.rectangle(frame, (x1,y1), (x2,y2), (0, 0, 255), 2)
                else:
                    frame = cv2.rectangle(frame, (x1,y1), (x2,y2), (0, 255, 0), 2)

            except:
                print("image mtcnn error")
                continue


    cv2.imshow("disp",frame)
    
    # print(image_cropped)
    
    
        

    cv2.waitKey(1)