# 完整的模型验证(测试,demo)套路-利用已经训练好的模型,然后给它提供输入

from PIL import Image
import torchvision
from torch import nn
import torch

image_path = "./imgs/dog.png"
image = Image.open(image_path)
print(image)

# png格式是四个通道,除了RGB三通道之外,还有一个透明度通道
image = image.convert('RGB')
# print(image)

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)),
                                            torchvision.transforms.ToTensor()])

image = transform(image)
print(image.shape)

# 创建网络模型
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(in_features=1024, out_features=64),
            nn.Linear(in_features=64, out_features=10)
        )
    
    def forward(self, x):
        x = self.model1(x)
        return x

model = torch.load("model_29_gpu.pth")
print(model)
image = torch.reshape(image, (1, 3, 32, 32))
if torch.cuda.is_available():
    image =  image.cuda()
model.eval()            # 这句话很重要
with torch.no_grad():   # 这句话很重要
    output = model(image)
    print(output)
    print(output.argmax(1))