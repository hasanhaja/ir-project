% Reading Notes
% 5 May 2020

# Contents

- [Computer Vision for Autonomous Vehicles: Problems, Datasets and State of the Art](#state-of-the-art)

# Computer Vision for Autonomous Vehicles: Problems, Datasets and State of the Art <a name="state-of-the-art"></a>

## Topics covered

- Semantic Segmentation
- Semantic Instance Segmentation

## Key points

- ResNet behave as linear ensemble of shallow networks
- In video applications, temporal frames correlation can be exploited for better image segmentation accuracy, efficiency and robustness. For example, 3D CNN like VoxNet and OctNet.
- "3D reconstruction works relatively well for static scenes but is still an open problem in dynamic scenes."
- There are methods for applications like stree side view, 3D data, road segmentation, freespace estimation, stixels (which are mid-level representation of 3D traffic), aerial image parsing (for automated extraction of urban objects),
- 3D data methods include online methods for batch processing to get real-time performance improvements that also leverage GPU processing.
- Data acquisition method like using OpenStreetMap can be paired with datasets for more information.

## Areas introduced here to explore further

- What are feature maps?
- Optical flow
  - Isn't this only useful in scenes in motion for reconstructive the environment?
- [OpenStreetMap](https://www.openstreetmap.org/#map=5/54.910/-3.432)

## Citations referenced

- MAP and CRF: 284, 674, 612
- Fully connected CRF: 534, 223, 352
- Co-occurence of object classes: 378, 776, 483
- Simultaneous Detection and Segmentation: 271
- Convolutional Feature masking: 146
- Online methods: 733
- Scene understanding: 193, 236, Chapter 14.
- Video application 3D CNN, like VoxNet and OctNet: 449 and 553 respectively.

# Detailed exploration of SotA datasets

The datasets being considered are considered for their use in semantic segmentation for the purpose of transport applications (i.e. autonomous vehicle scene understanding).

Most popular datasets for semantic segmentation are:

- Cityscapes
- PASCAL VOC
- Microsoft COCO

Datasets specifically for autonomous driving scenarios:

- Mapillary Vistas
- Apolloscape
- Berkeley DeepDrive

PASCAL VOC and Microsoft COCO are more general in nature and therefore it might be better to compare between Cityscapes and Apolloscape since they are constructed and tailored towards autonomous vehicle scene parsing [[citation](http://apolloscape.auto/scene.html)]. Mapillary Vistas is omitted due to a licensing paywall to access the full data with all of the annotations.

Selected datasets:

- Cityscapes
- Apolloscape

## Questions

Here are some questions that came after some reading and exploration:

- What is the difference between scene parsing and semantic segmentation? Is it just the application?
- What is panoptic segmentation? _Panoptic segmentation combines semantic segmentation and semantic instance segmentation [[citation](https://arxiv.org/pdf/1801.00868.pdf)]_

## Citations referenced

- Cite PASCAL VOC for why it wasn't considered here: [[citation](https://pjreddie.com/media/files/VOC2012_doc.pdf)]
- Comparison of various datasets including Cityscapes and Apollo [[citation](https://openaccess.thecvf.com/content_CVPR_2020/papers/Caesar_nuScenes_A_Multimodal_Dataset_for_Autonomous_Driving_CVPR_2020_paper.pdf)]. This citation looks to compare existing datasets while constructing a new one specifically for autonomous vehicles with a full suite of sensors.

# Detailed exploration of SotA methods

Some possible methods and the datasets they were trained on:

- PSPNet: Achieves sota performance on Cityscapes and Pascal VOC 2012
  - Implemented using Caffe and model is 260MB [[Source code and model](https://github.com/hszhao/PSPNet)]
- Atrous Spatial Pyramid Pooling: Achieves sota performance on PASCAL-Context, PASCAL-Person-Part, and Cityscapes, and is comparable to PSPNet
  - Built in Caffe [[Source code and model](http://liangchiehchen.com/projects/DeepLab.html)]
  - There are two versions here: [VGG16 based](http://liangchiehchen.com/projects/DeepLabv2_vgg.html) and [ResNet101 based](http://liangchiehchen.com/projects/DeepLabv2_resnet.html)
  - ResNet based method might better
  - Did not exploit multiscale inputs for cityscapes dataset because of limited GPU memory
  - [DeepLabv3](https://towardsdatascience.com/review-deeplabv3-atrous-convolution-semantic-segmentation-6d818bfd1d74)
  - DeepLabv3 can be downloaded and used from Pytorch [[source](https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/)]
- FC-DenseNet: Achieves sota performance on CamVid and Gatech
  - Built in Theano and Lasagne [[Source code and model](https://github.com/SimJeg/FC-DenseNet)]
  - The only model published is the FC-DenseNet103 variant. It is very deep and evaluation was originally done on platforms with Titan X 12GB GPU.

Selected methods:

- PSPNet
- ~~ASPP_VGG16~~
- ~~ASPP_ResNet101~~
- DeepLabv3_ResNet101
- FC-DenseNet103

### Rationale

The rationale for this choice are these are the current state of the art methods and they also have the source code and models publically available. The factor that is still a bit uncertain is the fact that not all the models were trained or even evaluated using the same datasets (some have but not all, and the uncertainty is surrounding why this decision was made). This could be because this research could have been published before the release of Cityscapes and Apolloscape.

## Questions

- Are these models downloadable?
- Are model accuracies and performance measured by IoUs?
- What is the difference between the different IoUs like mIoU and etc? _I think mIoU means 'Mean IoU'._
- What are all the different "methods" listed in the ASPP paper?
- Do I need to consider the other datasets for the comparison, or can I use their purported performance claims as the baseline and just continue with cityscapes and apolloscape?
- What is DeepLabv3+? [[citation](https://github.com/tensorflow/models/tree/master/research/deeplab)]
-

## Citations referenced

- PSPNet [[citation](https://openaccess.thecvf.com/content_cvpr_2017/papers/Zhao_Pyramid_Scene_Parsing_CVPR_2017_paper.pdf)]
- ASPP [[citation](https://arxiv.org/pdf/1606.00915.pdf)]
- FC-DenseNet [[citation](https://arxiv.org/pdf/1611.09326.pdf)]
- Tool to download datasets [[citation](https://github.com/fvisin/dataset_loaders)]

# Evaluation criteria

There are some recurring segmentation [metrics](https://www.jeremyjordan.me/evaluating-image-segmentation-models/) to evaluate the segmentation models:

- mIoU
- mAcc
- aAcc
