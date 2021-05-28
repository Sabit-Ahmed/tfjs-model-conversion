1. pip install -r requirements.txt

2. Install torch, torchvision and torchaudio:

    pip install torch==1.8.0+cpu torchvision==0.9.0+cpu torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html

3. Install onnx-tensorflow by:

    git clone https://github.com/onnx/onnx-tensorflow.git && cd onnx-tensorflow
    pip install -e .

4. `git clone https://github.com/stefanopini/simple-HigherHRNet.git`

5. Move "S1_pt_to_onnx.py" file into "SimpleHigherHRNet" folder.

6. Run `cd SimpleHigherHRNet/`

7. Run `python S1_pt_to_onnx.py` 
