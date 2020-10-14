import torch
import torchvision
from PIL import Image
import torchvision.transforms.functional as TF

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

model.eval()

image = Image.open('/home/dhaval/Desktop/Car-Image.jpg')

x = TF.to_tensor(image)
x.unsqueeze_(0)
print(x.shape)

#x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]

predictions = model(x)

print(predictions)
