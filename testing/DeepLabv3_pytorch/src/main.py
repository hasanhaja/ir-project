# from utilities.labels import CityscapesLabels

# city_labels = CityscapesLabels()
# labels = city_labels.labels()
# # DEBUG
# print(len(labels))
# print(city_labels.groups())


import torch
# Download an example image from the pytorch website
import urllib
# sample execution (requires torchvision)
from PIL import Image
import PIL
from torchvision import transforms
import matplotlib.pyplot as plt
import time
import glob
from os import path
import cv2
import numpy as np
import os
from os.path import isfile, join

from modeling import *
import torch.nn as nn

# model = torch.hub.load('pytorch/vision:v0.6.0',
#    'deeplabv3_resnet101', pretrained=True)

model = deeplabv3plus_mobilenet(num_classes=19, output_stride=16)
# checkpoint = torch.load(
#     "../models/best_deeplabv3plus_mobilenet_cityscapes_os16.pth", map_location=torch.device('cpu'))

model.load_state_dict(torch.load(
    "../models/best_deeplabv3plus_mobilenet_cityscapes_os16.pth", map_location=torch.device('cpu'))["model_state"])
model = nn.DataParallel(model)

# model loaded state


model.eval()

# TODO a list of images to test. Perhaps a directory.

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

filename = "../preliminary_test_results/images/170927_064639931_Camera_6.jpg"


def process_frame(filename, rescale_size=(800, 700), compose=preprocess):
    """
    This function processes a single frame and returns the resulting segmented frame.
    """
    input_image = Image.open(filename)
    # input_image = input_image.resize((1692, 1355))
    input_image = input_image.resize(rescale_size)

    input_tensor = compose(input_image)
    # create a mini-batch as expected by the model
    input_batch = input_tensor.unsqueeze(0)

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        # print("------------Model------------")
        # print(model(input_batch))
        # print("-------------End-------------")

        # output = model(input_batch)['out'][0]
        output = model(input_batch)[0]

    output_predictions = output.argmax(0)

    # create a color pallette, selecting a color for each class
    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")

    # plot the semantic segmentation predictions of 21 classes in each color
    r = Image.fromarray(output_predictions.byte().cpu().numpy()
                        ).resize(input_image.size)
    r.putpalette(colors)

    # TODO save to another folder

    # plt.imshow(r)
    # plt.savefig("result.jpg")
    # plt.show()

    return r


def process_sequence(src="images", dest="result"):
    """
    This function processes the sequences of input images to segmented images
    """
    counter = 0
    for file in glob.glob(f"{src}/*.jpg"):

        trimmed = path.basename(file)
        trimmed = f"result_{trimmed}"

        result = process_frame(file)
        plt.imshow(result)
        plt.savefig(f"{dest}/{trimmed}")

        counter += 1
        print(f"Completed image #{counter}")


def convert_sequence_to_video(src="result", dest="result"):

    pathOut = "result.mp4"
    fps = 10  # 30fps
    frame_array = []
    files = [f for f in os.listdir(src) if isfile(join(f"./{src}", f))]

    files = list(filter(lambda path: ".DS_Store" not in path, files))

    # print(files)

    # for sorting the file names properly
    # files.sort(key = lambda x: x[5:-4])
    # files.sort()
    # frame_array = []
    # files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]#for sorting the file names properly
    # files.sort(key = lambda x: x[5:-4])

    for i in range(len(files)):
        filename = f"./{src}/{files[i]}"
        # print(filename)
        # reading each files
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)

        # inserting the frames into an image array
        frame_array.append(img)

    out = cv2.VideoWriter(f"./{dest}/{pathOut}",
                          cv2.VideoWriter_fourcc(*"MP4V"), fps, size)

    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()


start = time.process_time()

# result = process_frame(filename)
# plt.imshow(result)
# plt.savefig("test_result.jpg")

process_sequence(src="../preliminary_test_results/images")
# convert_sequence_to_video(src="result")

end = time.process_time()

elapsed_time = end - start
print(f"Time taken: {elapsed_time}s")
# TODO image sequence to video
