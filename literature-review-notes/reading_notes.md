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
