### Import the ONNX model to Tensorflow ###

import onnx
from onnx_tf.backend import prepare

# Load the ONNX file
model = onnx.load('output/vgg19.onnx')

# Import the ONNX model to Tensorflow
tf_rep = prepare(model)

# Input nodes to the model
print('inputs:', tf_rep.inputs)

# Output nodes from the model
print('outputs:', tf_rep.outputs)

# All nodes in the model
print('tensor_dict:')
print(tf_rep.tensor_dict)


#################################

### Save the Tensorflow model into a file ###

tf_rep.export_graph('output/vgg19.pb')
