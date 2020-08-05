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

# Detailed exploration of SotA datasets and methods

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

## Questions

Here are some questions that came after some reading and exploration:

- What is the difference between scene parsing and semantic segmentation? Is it just the application?
- What is panoptic segmentation? _Panoptic segmentation combines semantic segmentation and semantic instance segmentation [[citation](https://arxiv.org/pdf/1801.00868.pdf)]_

## Citations referenced

- Cite PASCAL VOC for why it wasn't considered here: [[citation](https://pjreddie.com/media/files/VOC2012_doc.pdf)]
- Comparison of various datasets including Cityscapes and Apollo [[citation](https://openaccess.thecvf.com/content_CVPR_2020/papers/Caesar_nuScenes_A_Multimodal_Dataset_for_Autonomous_Driving_CVPR_2020_paper.pdf)]. This citation looks to compare existing datasets while constructing a new one specifically for autonomous vehicles with a full suite of sensors.
