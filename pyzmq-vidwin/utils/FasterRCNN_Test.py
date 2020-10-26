import torch
import torchvision
from PIL import Image
import torchvision.transforms as TF
import cv2
import matplotlib.pyplot as plt

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

model.eval()

# image = Image.open('/home/dhaval/Desktop/Car-Image.jpg')
#
# x = TF.to_tensor(image)
# x.unsqueeze_(0)
# print(x.shape)

# x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]

# predictions = model(x)
#
# print(predictions)


COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


def get_prediction(img_batch, threshold):
    transform = TF.Compose([TF.ToTensor()])  # Defing PyTorch Transform
    frames = [transform(Image.fromarray(frame)) for frame in img_batch]  # Apply the transform to the image
    predictions = model(frames)  # Pass the image to the model
    pred_class = [[COCO_INSTANCE_CATEGORY_NAMES[index] for index in list(pred['labels'].numpy())] for pred in predictions]  # Get the Prediction Score
    pred_boxes = [list(boxes_array) for pred in predictions for boxes_array in pred['boxes'].detach().numpy()]  # Bounding boxes
    pred_score = [list(pred['scores'].detach().numpy()) for pred in predictions]
    pred_t = [[index for index, value in enumerate(score) if value > threshold] for score in pred_score][-1]  # Get list of index with score greater than threshold.
    pred_boxes = pred_boxes[:pred_t + 1]
    pred_class = pred_class[:pred_t + 1]
    return pred_boxes, pred_class


def object_detection_api(img, threshold=0.5, rect_th=3, text_size=1, text_th=3, c=0):
    pred_boxes, pred_class = get_prediction(img, threshold)  # Get predictions
    for bounding_box in pred_boxes:
        cv2.rectangle(img[0], pred_boxes[i][0], pred_boxes[i][1], color=(0, 255, 0),
                      thickness=rect_th)  # Draw Rectangle with the coordinates
        cv2.putText(img[0], pred_class[i], pred_boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 255, 0),
                    thickness=text_th)  # Write the prediction class
    plt.figure(figsize=(20, 30))  # display the output image
    plt.imsave('/home/piyush/images/test_'+str(c)+'.png', img[0])


def stream(video_path):
    cap = cv2.VideoCapture(video_path)
    # get frame dimension
    width = cap.get(3)  # float
    height = cap.get(4)  # float

    print('width, height:', width, height)
    # read the fps so that opencv read with same fps speed
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"FPS for video: {video_path} : {fps}")
    # time to sleep the process
    i =0
    while True:
        frame_batch = []
        # Capture frame-by-frame

        ret, frame = cap.read()
        if i >= 1:
            print('frame: '+str(i))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # thi is done as openc v read bgr while pil read rgb
            #frame = Image.fromarray(frame)
            frame_batch.append(frame)

            if len(frame_batch) == 2:
                object_detection_api(frame_batch, 0.5, c=i)
        i=i+1
        if not ret:
            break
    cap.release()

#object_detection_api('/home/dhaval/Desktop/Car-Image.jpg')
stream('/home/piyush/test1.mp4')
