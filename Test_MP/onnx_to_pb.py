
import onnx
from onnx_tf.backend import prepare

model = onnx.load('trained_models/blazepose-heatmap-v1-2020-12-14.onnx')


