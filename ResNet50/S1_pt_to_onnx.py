### Export the trained model to ONNX  ###
import torch
from torch.autograd import Variable
import torchvision.models as models

trained_model = models.resnet50(pretrained=True)


# Export the trained model to ONNX
dummy_input = Variable(torch.randn(1, 3, 224, 224)) 
inputs = ['images']
outputs = ['scores']
dynamic_axes = {'images': {0: 'batch'}, 'scores': {0: 'batch'}}
# N x 3 x 224 x 224 ---> interpretation: N = batch_size, 3 channels for RGB images, image_height = 224, image_width = 224
torch.onnx.export(trained_model, dummy_input, "output/model.onnx",  input_names=inputs, output_names=outputs, dynamic_axes=dynamic_axes)