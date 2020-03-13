import torch
from model import edsr
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
from PIL import Image
import os


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} to run the code".format(device))

    transform2tensor = transforms.Compose([transforms.ToTensor()])
    inputfolder = "input"
    outputfolder = "output"
    inputlist = os.listdir(inputfolder)

    if not inputlist:
        print('The input directory is empty')
        return
        
    if not os.path.exists(outputfolder):
        os.makedirs(outputfolder)

    # load the NN 
    chk = torch.load('EDSR_baseline_x4.pt')
    args = model_args()
    model = edsr.EDSR(args)
    model.load_state_dict(chk)
    model = model.to(device)

    with torch.no_grad():
        for imgname in inputlist:
            print("processing {}".format(imgname))
            input_name = os.path.join(inputfolder, imgname)
            input_tensor = transform2tensor(Image.open(input_name))    
            input_tensor = torch.unsqueeze(input_tensor,0)
            input_tensor = input_tensor.to(device) * 255

        
            out = model(input_tensor)
            out = quantize(out, args.rgb_range) / 255

            out = out.cpu()
            output_name = os.path.join(outputfolder, imgname[:-4] + '_sr.png')

            torchvision.utils.save_image(out, output_name)

def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)


class model_args():
    def __init__(self):
        self.n_resblocks = 16
        self.n_feats = 64
        self.scale = [4]
        self.rgb_range= 255
        self.n_colors= 3
        self.res_scale = 1

if __name__ == "__main__":
    main()