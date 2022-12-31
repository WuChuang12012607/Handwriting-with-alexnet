from __future__ import print_function
import torch
import torch.nn.parallel
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
import cv2
from torch.autograd import Variable
import segmentation_models_pytorch as smpd


def run(input_path, weigths_path):
    output_file = input_path[:input_path.rfind('.')]
    model = smpd.Unet(in_channels=3, classes=1, activation="sigmoid")
    model.load_state_dict(torch.load(weigths_path, map_location=torch.device('cpu')))
    model = model.to(device)
    model.eval()

    x = cv2.imread(input_path, 1)[:, :, ::-1]
    new_shape = (x.shape[1] - x.shape[1] % 32, x.shape[0] - x.shape[0] % 32)
    x = cv2.resize(x, new_shape)

    x = transforms.ToTensor()(x.copy())
    x = torch.unsqueeze(x, 0)
    x = x.to(device)

    x = Variable(x).to(device)
    y_hat = model(x)

    out_img = y_hat.cpu().data
    out_img = 1 - (out_img - out_img.min()) / (out_img.max() - out_img.min())
    vutils.save_image(out_img, output_file + '_output.jpg', normalize=False)

device = torch.device("cpu")
print("Device used : ", device)


input_path = 'and.jpg'
weigths_path = 'unet.pth'

run(input_path, weigths_path)
