# Things to address

- Performance of SoTA methods on different datasets, and this could be addressing:
    - Robustness of the models
    - Diversity of the datasets
    - How can you tell the difference?
- Novelty factor

# Plan

1. Run all the identified models with evaluation data from Cityscapes.
2. Document:
    1. If Cityscapes was the dataset used to train the model, then record it as baseline performance.
    2. If not, then record it against the performance on cityscapes along with the literture value for baseline performance.
3. Run all the identified models with evaluation data from Apolloscape.
4. Run those experiments on different hardware.
    1. Explore the impact of (dedicated) GPU vs no GPU
    2. This can be done on Macbook Pro 2013 13", 2013 15", 2019 15", Windows w/ GTX1160, Jetson (and other platform by Eric), Digital Ocean droplet, AWS Pytorch

# Todo

- Run models with evaluation data from Cityscapes
- Run models with evaluation data from ApolloScape
- Revisit methodology for further experimentation
- Review literature (and document) again for details regarding dataset
- Review literature and documentation of pytorch (or any other) regarding GPU usage during evaluation

# Dataset

## Cityscapes

- [Scripts to understand data](https://github.com/mcordts/cityscapesScripts)
    - How does this feed into the training parts?
    - How is this different to the training steps in the previous projects?

# Unknowns

- Which part of the dataset needs to be used?
    - Which part of Cityscapes? Coarse? Fine? 8-bit something?
    - Which part of ApolloScape? Ins or Seg?
- Do the methods use the GPU for evaluation? Or is not for training purposes?
