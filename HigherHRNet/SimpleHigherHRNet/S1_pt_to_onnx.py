### Export the trained model to ONNX  ###
import torch
from torch.autograd import Variable
import torchvision.models as models
from SimpleHigherHRNet import SimpleHigherHRNet
from models.higherhrnet import HigherHRNet
from collections import OrderedDict


# trained_model = models.resnet152(pretrained=True)
model_structure = HigherHRNet(c=32, nof_joints=17)

checkpoint = torch.load("../output/pose_higher_hrnet_w32_512.pth", map_location='cpu')

if 'model' in checkpoint:
    checkpoint = checkpoint['model']
# fix issue with official high-resolution weights
checkpoint = OrderedDict([(k[2:] if k[:2] == '1.' else k, v) for k, v in checkpoint.items()])
model_structure.load_state_dict(checkpoint)


# Export the trained model to ONNX
dummy_input = Variable(torch.randn(1, 3, 512, 512)) 
inputs = ['images']
outputs = ['scores']
dynamic_axes = {'images': {0: 'batch'}, 'scores': {0: 'batch'}}
# N x 3 x 224 x 224 ---> interpretation: N = batch_size, 3 channels for RGB images, image_height = 224, image_width = 224
torch.onnx.export(model_structure, dummy_input, "../output/model.onnx",  input_names=inputs, output_names=outputs, dynamic_axes=dynamic_axes)