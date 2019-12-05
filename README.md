# MIMAMO-Net
MIMAMO Net: Integrating Micro- and Macro-motion for Video Emotion Recognition

Paper Link: https://arxiv.org/pdf/1911.09784.pdf

Requirements:

1. Pytorch 0.4.1 (or higher version)
2. Numpy
3. [PyTorchSteerablePyramid](https://github.com/tomrunia/PyTorchSteerablePyramid)
4. [pytorch-benchmarks](https://github.com/albanie/pytorch-benchmarks)
5. [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace)


In this paper, we propose to combine the micro- and macro-motion features to improve video emotion recognition, using a two-stream recurrrent network named MIMAMO (Micro-Macro-Motion) Net. This model structure is shown in the picture:
![alt text](https://github.com/wtomin/MIMA-Net/blob/master/model.png)

To run this project, 

(1) Download the pretrained ResNet50 model from this [webpage](https://www.robots.ox.ac.uk/~albanie/pytorch-models.html), which is pretrained on VGGFACE2 and FER_plus. Make sure the [pytorch-benchmarks](https://github.com/albanie/pytorch-benchmarks) is correctly installed and the pretrained model can be imported.

(2) Use [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace) toolkit to crop and align faces in videos, save aligned faces.

(3) Extracted the Pool5 features of ResNet50 model and save features. Using the python script in './scripts/CNN_feature_extraction.py':
```
python CNN_feature_extraction.py --fps 30 --layer_name pool5_7x7_s1 --save_root Extracted_Features --data_root dir-to-aligned-face
```

(4) Before running experiments on [Aff-wild dataset](https://ibug.doc.ic.ac.uk/resources/first-affect-wild-challenge/) (or [OMG emotion dataset](https://github.com/knowledgetechnologyuhh/OMGEmotionChallenge)), make sure dataset is downloaded and processed in step (3).

Run scripts in 'Aff-wild-exps' or 'OMG-exps'.


