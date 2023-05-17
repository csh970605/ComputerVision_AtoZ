import torch
from torch.autograd import Variable
import cv2
from data import BaseTransform, VOC_CLASSES as labelmap
from ssd import build_ssd
import imageio

# Define a function that will do the detections
def detect(frame, model, transform):
    height, width = frame.shape[:2]
    threshold = 0.6
    frame_t = transform(frame)[0]
    x = torch.from_numpy(frame_t).permute(2, 0, 1)  #convert numpy as torch tensor with RGB
    x = Variable(x.unsqueeze(0)) # Append batch at first dimension.
    
    y = model(x)
    # detections = [batch, number of classes, number of occurence, (score, x0, y0, x1, y1)]
    detections = y.data
    
    scale = torch.Tensor([width, height, width, height]) # represents coordinate of Top&Left / Bottom&Right

    for i in range(detections.size(1)): # i = class
        j = 0 # occurence
        while detections[0, i, j, 0] >= threshold:
            pt = (detections[0, i, j, 1:] * scale).numpy()
            cv2.rectangle(frame, (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3])), [0, 255, 0], 2)
            cv2.putText(frame, labelmap[i-1], (int(pt[0]), int(pt[1])), cv2.FONT_HERSHEY_SIMPLEX, 2, [255, 255, 255], 2, cv2.LINE_AA)
            j += 1
            
    return frame


# Create the SSD neural network
model = build_ssd('test')
model.load_state_dict(torch.load('ssd300_mAP_77.43_v2.pth', map_location=lambda storage, loc:storage))

# Create the transformation
transform = BaseTransform(model.size, (104/256.0, 117/256.0, 123/256.0))

# Detect some object on a video
reader = imageio.get_reader('funny_dog.mp4')
fps = reader.get_meta_data()['fps']
writer = imageio.get_writer('funn_dog_result.mp4', fps = fps)

for i, frame in enumerate(reader):
    frame = detect(frame, model.eval(), transform)
    writer.append_data(frame)
    
