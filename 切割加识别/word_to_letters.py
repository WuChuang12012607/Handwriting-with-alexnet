import cv2
import numpy as np
import torch.nn as nn
import torch
from PIL import Image
from torchvision import transforms
import cv2
import os
import shutil



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
hash={"0": "a", "1": "b", "2": "c", "3": "d", "4": "e", "5": "f", "6": "g", "7": "h", "8": "i", "9": "j", "10": "k", "11": "l", "12": "m", "13": "n", "14": "o", "15": "p", "16": "q", "17": "r", "18": "s", "19": "t", "20": "u", "21": "v", "22": "w", "23": "x", "24": "y", "25": "z","26": "A", "27": "B", "28": "D", "29": "E", "30": "F", "31": "G", "32": "H", "33": "I", "34": "J", "35": "L", "36": "M", "37": "N", "38": "Q", "39": "R", "40": "T"}


def get_predict(filename):
    data_transform = transforms.Compose( # 数据转换模型
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Lambda(lambda x: x.repeat(3,1,1)),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    img = Image.open(filename) # 加载图片，自定义的图片名称
    img = img.convert('1')
    img = data_transform(img) # 图片转换为矩阵
    # 对数据维度进行扩充
    img = torch.unsqueeze(img, dim=0)
    # 创建模型
    model = AlexNet(num_classes=41)
    # 加载模型权重
    model_weight_path = "AlexNet1.pth" #与train.py里的文件名对应
    model.load_state_dict(torch.load(model_weight_path,map_location='cpu'))
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img)) # 图片压缩
        predict = torch.softmax(output, dim=0) # 求softmax值
        predict_cla = torch.argmax(predict).numpy() # 预测分类结果
        #with open("index.json","r")as f:
        #    data = json.load(f)
        print(hash[str(predict_cla)])
        return hash[str(predict_cla)]

def photo_processing(filepath):
    img = cv2.imread(filepath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imwrite(filepath, thresh)
    im = Image.open(filepath)

    # 获取图像的长和宽
    width, height = im.size

    # 计算正方形的边长
    side = max(width, height)

    # 计算填充部分的宽度和高度
    pad_width = (side - width) / 2
    pad_height = (side - height) / 2
    
    # 创建一个白色画布
    new_im = Image.new("RGB", (side, side), (255, 255, 255))

    # 在画布上将图像居中放置
    new_im.paste(im, (int(pad_width), int(pad_height)))
    new_im = new_im.resize((224, 224))
    # 保存图像
    new_im.save(filepath)
    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

    cv2.imwrite(filepath, image)
    # 保存结果图片
    return
def single_predict(filepath):
    photo_processing(filepath)
    return get_predict(filepath)


def cut(filepath):
    # 读入图像
    image = cv2.imread(filepath)

    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 二值化
    threshold, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # 去噪
    kernel = np.ones((3,3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # 检测边缘
    edges = cv2.Canny(binary, threshold, threshold * 2)

    # 形态学处理
    edges = cv2.dilate(edges, kernel)

    # 检测轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    a=[]
    # 遍历每个轮廓
    i= 0
    for contour in contours:
        # 计算轮廓的外接矩形
        x, y, w, h = cv2.boundingRect(contour)

        # 切割字符
        char_image = binary[y:y+h, x:x+w]
        i = i+1
       # 保存字符图像
        cv2.imwrite(f'word_to_letters\\{x}.jpg', char_image)
        image = cv2.imread(f'word_to_letters\\{x}.jpg', cv2.IMREAD_GRAYSCALE)
        inverted = cv2.bitwise_not(image)
        cv2.imwrite(f'word_to_letters\\{x}.jpg', inverted)
    


def read_files(folder_path):
    file_names = []
    for file_name in os.listdir(folder_path):
        file_names.append(file_name)
    return file_names


# 调用函数
def del_file(path_data):
    for i in os.listdir(path_data):  # os.listdir(path_data)#返回一个列表，里面是当前目录下面的所有东西的相对路径
        file_data = path_data + "\\" + i  # 当前文件夹的下面的所有东西的绝对路径
        if os.path.isfile(file_data) == True:  # os.path.isfile判断是否为文件,如果是文件,就删除.如果是文件夹.递归给del_file.
            os.remove(file_data)
        else:
            del_file(file_data)



def final_read(word_path):
    del_file("word_to_letters")
    cut(word_path)
    path = "word_to_letters"
    path_list = os.listdir(path)

    path_list.sort(key=lambda x: int(x.split('.')[0]))

    path_list = ['word_to_letters\\' + s for s in path_list]
    a = []
    for file_path in path_list:
        a.append(single_predict(file_path))
    word = ''.join(a)
    print(word)
    return word
