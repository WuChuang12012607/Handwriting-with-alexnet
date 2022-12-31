import torch.nn as nn
import torch
from PIL import Image
from torchvision import transforms
import json


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, init_weights=False):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(  # 打包
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),  # input[3, 224, 224]  output[48, 55, 55] 自动舍去小数点后
            nn.ReLU(inplace=True),  # inplace 可以载入更大模型
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[48, 27, 27] kernel_num为原论文一半
            nn.Conv2d(48, 128, kernel_size=5, padding=2),  # output[128, 27, 27]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[128, 13, 13]
            nn.Conv2d(128, 192, kernel_size=3, padding=1),  # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),  # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),  # output[128, 13, 13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[128, 6, 6]
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            # 全链接
            nn.Linear(128 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)  # 展平或者view()
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # 何教授方法
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)  # 正态分布赋值
                nn.init.constant_(m.bias, 0)



def get_predict(file):
    data_transform = transforms.Compose( # 数据转换模型
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Lambda(lambda x:x.repeat(3,1,1)),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    img = Image.open(file) # 加载图片，自定义的图片名称
    img = img.convert('1')
    img = data_transform(img) # 图片转换为矩阵
    # 对数据维度进行扩充
    img = torch.unsqueeze(img, dim=0)
    # 创建模型
    model = AlexNet(num_classes=41)
    # 加载模型权重
    model_weight_path = "G:/SL/AlexNet1.pth" #与train.py里的文件名对应
    temp = torch.load(model_weight_path, map_location='cpu')
    model.load_state_dict(temp)
    model.eval()
    hash={"0": "a", "1": "b", "2": "c", "3": "d", "4": "e", "5": "f", "6": "g", "7": "h", "8": "i", "9": "j", "10": "k", "11": "l", "12": "m", "13": "n", "14": "o", "15": "p", "16": "q", "17": "r", "18": "s", "19": "t", "20": "u", "21": "v", "22": "w", "23": "x", "24": "y", "25": "z","26": "A", "27": "B", "28": "D", "29": "E", "30": "F", "31": "G", "32": "H", "33": "I", "34": "J", "35": "L", "36": "M", "37": "N", "38": "Q", "39": "R", "40": "T"}

    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img)) # 图片压缩
        predict = torch.softmax(output, dim=0) # 求softmax值
        predict_cla = torch.argmax(predict).numpy() # 预测分类结果
        #with open("index.json","r")as f:
        #    data = json.load(f)
        print("预测结果为",hash[str(predict_cla)])
        return hash[str(predict_cla)]
# get_predict("C:/Users/luobe/Desktop/9f2739cc659317fcb02a1528cdbdfdb.jpg")
# get_predict("C:/Users/luobe/Desktop/a52331e3d03efa756f4303593eeff14.jpg")