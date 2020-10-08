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

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import Cityscapes as PyCS
from training.apolloscape import Apolloscape
from training.cityscapes import Cityscapes as CS
from tqdm import tqdm
from training.stream_metrics import StreamSegMetrics
from training.visualizer import Visualizer
from utils import Denormalize


class MyDtypeOps(object):
    def __call__(self, image):
        # print(f"Types: {image.dtype}; Image: {image}")
        if not torch.is_tensor(image):
            image = transforms.functional.to_tensor(image)

        # print(image.dtype)
        return transforms.functional.convert_image_dtype(image, dtype=torch.float64)


class Debug(object):
    def __call__(self, image):
        print(f"DEBUG: {image.size()}")
        return image


class ImageOps(object):
    def __call__(self, image):
        # print(f"Old: {image.mode}")
        image.mode = "F"
        # print(f"New: {image.mode}")
        return image


class Cityscapes(PyCS):

    train_id_to_color = [c.color for c in PyCS.classes if (
        c.train_id != -1 and c.train_id != 255)]
    train_id_to_color.append([0, 0, 0])
    train_id_to_color = np.array(train_id_to_color)
    id_to_train_id = np.array([c.train_id for c in PyCS.classes])

    def __init__(self, root, split='train', mode='fine', target_type='instance',
                 transform=None, target_transform=None, transforms=None):
        super(Cityscapes, self).__init__(root, split, mode,
                                         target_type, transform, target_transform, transforms)

    @classmethod
    def encode_target(cls, target):
        return cls.id_to_train_id[np.array(target)]

    @classmethod
    def decode_target(cls, target):
        target[target == 255] = 19
        #target = target.astype('uint8') + 1
        return cls.train_id_to_color[target]


def process_frame(filename, compose, rescale_size=(800, 700)):
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


def validate(model, loader, device, metrics, ret_samples_ids=None):
    """Do validation and return specified samples"""
    metrics.reset()
    ret_samples = []
    # if opts.save_val_results:
    if not os.path.exists('results'):
        os.mkdir('results')
    denorm = Denormalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    img_id = 0

    with torch.no_grad():
        loader_iter = iter(loader)

        while True:
            loader_val = next(loader_iter, "done")
            if loader_val == "done":
                break
            else:
                if loader_val is not None:

                    images, labels = loader_val

                    images = images.to(device, dtype=torch.float32)
                    labels = labels.to(device, dtype=torch.long)

                    outputs = model(images)
                    preds = outputs.detach().max(dim=1)[1].cpu().numpy()
                    targets = labels.cpu().numpy()

                    metrics.update(targets, preds)

                    for i in range(len(images)):
                        image = images[i].detach().cpu().numpy()
                        target = targets[i]
                        pred = preds[i]

                        image = (denorm(image) * 255).transpose(1,
                                                                2, 0).astype(np.uint8)
                        # TODO I'm not sure what decode really does for me, so I'm gonna comment it out and see what happens. Might need to reshape it, but don't know into what.
                        target = loader.dataset.decode_target(
                            target).astype(np.uint8)
                        pred = loader.dataset.decode_target(
                            pred).astype(np.uint8)

                        # print(f"Target type: {type(target)}")
                        # print(f"Pred type: {type(pred)}")
                        # print("Before")
                        # target = target.astype(np.uint8)
                        # pred = pred.astype(np.uint8)

                        # print(f"Target type: {type(target)}")
                        # print(f"Pred type: {type(pred)}")

                        # continue

                        # TODO Can't figure this out yet
                        # Image.fromarray(image).save(
                        #     'results/%d_image.png' % img_id)
                        # Image.fromarray(target).save(
                        #     'results/%d_target.png' % img_id)
                        # Image.fromarray(pred).save(
                        #     'results/%d_pred.png' % img_id)

                        # fig = plt.figure()
                        # plt.imshow(image)
                        # plt.axis('off')
                        # plt.imshow(pred, alpha=0.7)
                        # ax = plt.gca()
                        # ax.xaxis.set_major_locator(
                        #     matplotlib.ticker.NullLocator())
                        # ax.yaxis.set_major_locator(
                        #     matplotlib.ticker.NullLocator())
                        # plt.savefig('results/%d_overlay.png' %
                        #             img_id, bbox_inches='tight', pad_inches=0)
                        # plt.close()
                        img_id += 1

        # for i, (images, labels) in tqdm(enumerate(loader)):

        #     images = images.to(device, dtype=torch.float32)
        #     labels = labels.to(device, dtype=torch.long)

        #     outputs = model(images)
        #     preds = outputs.detach().max(dim=1)[1].cpu().numpy()
        #     targets = labels.cpu().numpy()

        #     metrics.update(targets, preds)
        #     # if ret_samples_ids is not None and i in ret_samples_ids:  # get vis samples
        #     #     ret_samples.append(
        #     #         (images[0].detach().cpu().numpy(), targets[0], preds[0]))

        #     for i in range(len(images)):
        #         image = images[i].detach().cpu().numpy()
        #         target = targets[i]
        #         pred = preds[i]

        #         image = (denorm(image) * 255).transpose(1,
        #                                                 2, 0).astype(np.uint8)
        #         target = loader.dataset.decode_target(
        #             target).astype(np.uint8)
        #         pred = loader.dataset.decode_target(pred).astype(np.uint8)

        #         Image.fromarray(image).save(
        #             'results/%d_image.png' % img_id)
        #         Image.fromarray(target).save(
        #             'results/%d_target.png' % img_id)
        #         Image.fromarray(pred).save('results/%d_pred.png' % img_id)

        #         fig = plt.figure()
        #         plt.imshow(image)
        #         plt.axis('off')
        #         plt.imshow(pred, alpha=0.7)
        #         ax = plt.gca()
        #         ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
        #         ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
        #         plt.savefig('results/%d_overlay.png' %
        #                     img_id, bbox_inches='tight', pad_inches=0)
        #         plt.close()
        #         img_id += 1

        score = metrics.get_results()
    return score, ret_samples


def func(x):
    return x.repeat(1, 1, 1)


def get_dataset(dataset, data_root, crop_size):
    """ Dataset And Augmentation
    """
    root_full_path = os.path.join(data_root)

    # train_transform = et.ExtCompose([
    #     # et.ExtResize( 512 ),
    #     # et.ExtRandomCrop(size=(crop_size, crop_size)),
    #     # et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
    #     # et.ExtRandomHorizontalFlip(),
    #     et.ExtToTensor(),
    #     et.ExtNormalize(mean=[0.485, 0.456, 0.406],
    #                     std=[0.229, 0.224, 0.225]),
    # ])

    # val_transform = et.ExtCompose([
    #     # et.ExtResize( 512 ),
    #     et.ExtToTensor(),
    #     et.ExtNormalize(mean=[0.485, 0.456, 0.406],
    #                     std=[0.229, 0.224, 0.225]),
    # ])

    train_transform = transforms.Compose([
        # transforms.RandomResizedCrop(crop_size),
        transforms.Resize(256),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(512),
        # transforms.CenterCrop(224),
        # transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    target_transform = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor(),
    ])

    if dataset.lower() == "cityscapes":
        print(f"[INFO] Fetching Cityscapes dataset from: {root_full_path}")
        train_dst = Cityscapes(root=data_root,
                               split='train',
                               transform=train_transform,
                               #    target_transform=target_transform,
                               )
        val_dst = Cityscapes(root=data_root,
                             split='val',
                             transform=val_transform,
                             target_transform=target_transform,
                             target_type="semantic",
                             )
    else:
        print(f"[INFO] Fetching ApolloScape dataset from: {root_full_path}")
        train_dst = Apolloscape(root=root_full_path, road="road02_seg", transform=train_transform,
                                normalize_poses=True, pose_format='quat', train=True, cache_transform=True, stereo=False)

        val_dst = Apolloscape(root=root_full_path, road="road02_seg",
                              transform=val_transform, normalize_poses=True, pose_format='quat', train=False, cache_transform=True, stereo=False)

    return train_dst, val_dst


def dataset_config(dataset):
    if dataset.lower() == "cityscapes":
        return ("Cityscapes", "../../datasets/Cityscapes")
    elif dataset.lower() == "apolloscape":
        return ("ApolloScape", "../../datasets/ApolloScape")
    else:
        raise NameError(
            f"[ERROR] {dataset} not recognized. Use either \"Cityscapes\" or \"ApolloScape\".")


# def custom_collate(batch):
    # print("Called")
    # print(len(batch))
    # [print(e) for e in batch]
    # print("DONE!!")


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # LOAD DATA
    # preprocess = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                         std=[0.229, 0.224, 0.225]),
    # ])

    dataset, dataset_dir = dataset_config("cityscapes")
    # dataset, dataset_dir = dataset_config("apolloscape")
    train_dst, val_dst = get_dataset(dataset, dataset_dir, 768)

    print(len(val_dst))

    batch_size = 16

    # train_loader = DataLoader(train_dst, batch_size=batch_size,
    #   shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dst, batch_size=batch_size,
                            shuffle=True, num_workers=2,
                            # collate_fn=custom_collate
                            )

    # DEBUG Checking if the iteration works
    # data_iter = iter(val_loader)

    # while True:
    #     val = next(data_iter, "end")
    #     if val == "end":
    #         print("Ending...")
    #         break
    #     else:
    #         if val == None:
    #             print("None found!")
    #         else:
    #             inputs, labels = val
    #             print(labels)

    # return

    metrics = StreamSegMetrics(19)

    model = deeplabv3plus_mobilenet(num_classes=19, output_stride=16)
    # checkpoint = torch.load(
    #     "../models/best_deeplabv3plus_mobilenet_cityscapes_os16.pth", map_location=torch.device('cpu'))

    model.load_state_dict(torch.load(
        "../models/best_deeplabv3plus_mobilenet_cityscapes_os16.pth", map_location=torch.device(device))["model_state"])
    model = nn.DataParallel(model)

    # TODO Select val data from apolloscape to evaluate
    # TODO Stream metrics for getting results.

    # filename = "../preliminary_test_results/images/170927_064639931_Camera_6.jpg"

    start = time.process_time()

    # model loaded state
    model.eval()

    val_score, _ = validate(model=model, loader=val_loader,
                            device=device, metrics=metrics)
    print(metrics.to_str(val_score))

    # result = process_frame(filename)
    # plt.imshow(result)
    # plt.savefig("test_result.jpg")

    # process_sequence(src="../preliminary_test_results/images")
    # convert_sequence_to_video(src="result")

    end = time.process_time()

    elapsed_time = end - start
    print(f"Time taken: {elapsed_time}s")
    # TODO image sequence to video


if __name__ == "__main__":
    main()
