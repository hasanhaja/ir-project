import json
import os
from collections import namedtuple
import zipfile

from .utils import extract_archive, verify_str_arg, iterable_to_str
from .vision import VisionDataset
from PIL import Image

     class ApolloScape(VisionDataset):
    """`Cityscapes <http://www.cityscapes-dataset.com/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory ``leftImg8bit``
            and ``gtFine`` or ``gtCoarse`` are located.
        split (string, optional): The image split to use, ``train``, ``test`` or ``val`` if mode="fine"
            otherwise ``train``, ``train_extra`` or ``val``
        mode (string, optional): The quality mode to use, ``fine`` or ``coarse``
        target_type (string or list, optional): Type of target to use, ``instance``, ``semantic``, ``polygon``
            or ``color``. Can also be a list to output a tuple with all specified target types.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.

    Examples:

        Get semantic segmentation target

        .. code-block:: python

            dataset = Cityscapes('./data/cityscapes', split='train', mode='fine',
                                 target_type='semantic')

            img, smnt = dataset[0]

        Get multiple targets

        .. code-block:: python

            dataset = Cityscapes('./data/cityscapes', split='train', mode='fine',
                                 target_type=['instance', 'color', 'polygon'])

            img, (inst, col, poly) = dataset[0]

        Validate on the "coarse" set

        .. code-block:: python

            dataset = Cityscapes('./data/cityscapes', split='val', mode='coarse',
                                 target_type='semantic')

            img, smnt = dataset[0]
    """

    # TODO change to ApolloscapeClass
    # TODO add function to map hex color to rbg (util function already provided in labels_apollo.py)
    # TODO make void the classes not considered by cityscapes
    # TODO How is this used in evaluation?
    # TODO change the __init__ function and how the dataset is split
    # TODO check and update the __get__ function to work properly with apolloscape according to it file structure.
    # TODO _load_json() might need to be replaced with another function.

    ApolloScapeClass = namedtuple('ApolloScapeClass', [
                                  'name', 'id', 'train_id', 'category', 'category_id', 'has_instances', 'ignore_in_eval', 'color'])

    # TODO verify what other classes to make void for cityscapes comparison.
    # classes = [
    #     CityscapesClass('unlabeled', 0, 255, 'void', 0, False, True, (0, 0, 0)),
    #     CityscapesClass('ego vehicle', 1, 255, 'void', 0, False, True, (0, 0, 0)),
    #     CityscapesClass('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
    #     CityscapesClass('out of roi', 3, 255, 'void', 0, False, True, (0, 0, 0)),
    #     CityscapesClass('static', 4, 255, 'void', 0, False, True, (0, 0, 0)),
    #     CityscapesClass('dynamic', 5, 255, 'void', 0, False, True, (111, 74, 0)),
    #     CityscapesClass('ground', 6, 255, 'void', 0, False, True, (81, 0, 81)),
    #     CityscapesClass('road', 7, 0, 'flat', 1, False, False, (128, 64, 128)),
    #     CityscapesClass('sidewalk', 8, 1, 'flat', 1, False, False, (244, 35, 232)),
    #     CityscapesClass('parking', 9, 255, 'flat', 1, False, True, (250, 170, 160)),
    #     CityscapesClass('rail track', 10, 255, 'flat', 1, False, True, (230, 150, 140)),
    #     CityscapesClass('building', 11, 2, 'construction', 2, False, False, (70, 70, 70)),
    #     CityscapesClass('wall', 12, 3, 'construction', 2, False, False, (102, 102, 156)),
    #     CityscapesClass('fence', 13, 4, 'construction', 2, False, False, (190, 153, 153)),
    #     CityscapesClass('guard rail', 14, 255, 'construction', 2, False, True, (180, 165, 180)),
    #     CityscapesClass('bridge', 15, 255, 'construction', 2, False, True, (150, 100, 100)),
    #     CityscapesClass('tunnel', 16, 255, 'construction', 2, False, True, (150, 120, 90)),
    #     CityscapesClass('pole', 17, 5, 'object', 3, False, False, (153, 153, 153)),
    #     CityscapesClass('polegroup', 18, 255, 'object', 3, False, True, (153, 153, 153)),
    #     CityscapesClass('traffic light', 19, 6, 'object', 3, False, False, (250, 170, 30)),
    #     CityscapesClass('traffic sign', 20, 7, 'object', 3, False, False, (220, 220, 0)),
    #     CityscapesClass('vegetation', 21, 8, 'nature', 4, False, False, (107, 142, 35)),
    #     CityscapesClass('terrain', 22, 9, 'nature', 4, False, False, (152, 251, 152)),
    #     CityscapesClass('sky', 23, 10, 'sky', 5, False, False, (70, 130, 180)),
    #     CityscapesClass('person', 24, 11, 'human', 6, True, False, (220, 20, 60)),
    #     CityscapesClass('rider', 25, 12, 'human', 6, True, False, (255, 0, 0)),
    #     CityscapesClass('car', 26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
    #     CityscapesClass('truck', 27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
    #     CityscapesClass('bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
    #     CityscapesClass('caravan', 29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
    #     CityscapesClass('trailer', 30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
    #     CityscapesClass('train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
    #     CityscapesClass('motorcycle', 32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
    #     CityscapesClass('bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
    #     CityscapesClass('license plate', -1, -1, 'vehicle', 7, False, True, (0, 0, 142)),
    # ]

    classes = [
        #     name id trainId category  catId  hasInstanceignoreInEval   color
        ApolloScapeClass('others', 0, 255, '其他', 0, False, True, 0x000000),
        ApolloScapeClass('rover', 1, 255, '其他', 0, False, True, 0X000000),
        ApolloScapeClass('sky', 17, 0, '天空', 1, False, False, 0x4682B4),
        ApolloScapeClass('car', 33, 1, '移动物体', 2, True, False, 0x00008E),
        ApolloScapeClass('car_groups', 161, 1, '移动物体', 2, True, False, 0x00008E),
        ApolloScapeClass('motorbicycle', 34, 2, '移动物体', 2, True, False, 0x0000E6),
        ApolloScapeClass('motorbicycle_group', 162, 2, '移动物体', 2, True, False, 0x0000E6),
        ApolloScapeClass('bicycle', 35, 3, '移动物体', 2, True, False, 0x770B20),
        ApolloScapeClass('bicycle_group', 163, 3, '移动物体', 2, True, False, 0x770B20),
        ApolloScapeClass('person', 36, 4, '移动物体', 2, True, False, 0x0080c0),
        ApolloScapeClass('person_group', 164, 4, '移动物体', 2, True, False, 0x0080c0),
        ApolloScapeClass('rider', 37, 5, '移动物体', 2, True, False, 0x804080),
        ApolloScapeClass('rider_group', 165, 5, '移动物体', 2, True, False, 0x804080),
        ApolloScapeClass('truck', 38, 6, '移动物体', 2, True, False, 0x8000c0),
        ApolloScapeClass('truck_group', 166, 6, '移动物体', 2, True, False, 0x8000c0),
        ApolloScapeClass('bus', 39, 7, '移动物体', 2, True, False, 0xc00040),
        ApolloScapeClass('bus_group', 167, 7, '移动物体', 2, True, False, 0xc00040),
        ApolloScapeClass('tricycle', 40, 8, '移动物体', 2, True, False, 0x8080c0),
        ApolloScapeClass('tricycle_group', 168, 8, '移动物体', 2, True, False, 0x8080c0),
        ApolloScapeClass('road', 49, 9, '平面', 3, False, False, 0xc080c0),
        ApolloScapeClass('siderwalk', 50, 10, '平面', 3, False, False, 0xc08040),
        ApolloScapeClass('traffic_cone', 65, 11, '路间障碍', 4, False, False, 0x000040),
        ApolloScapeClass('road_pile', 66, 12, '路间障碍', 4, False, False, 0x0000c0),
        ApolloScapeClass('fence', 67, 13, '路间障碍', 4, False, False, 0x404080),
        ApolloScapeClass('traffic_light', 81, 14, '路边物体', 5, False, False, 0xc04080),
        ApolloScapeClass('pole', 82, 15, '路边物体', 5, False, False, 0xc08080),
        ApolloScapeClass('traffic_sign', 83, 16, '路边物体', 5, False, False, 0x004040),
        ApolloScapeClass('wall', 84, 17, '路边物体', 5, False, False, 0xc0c080),
        ApolloScapeClass('dustbin', 85, 18, '路边物体', 5, False, False, 0x4000c0),
        ApolloScapeClass('billboard', 86, 19, '路边物体', 5, False, False, 0xc000c0),
        ApolloScapeClass('building', 97, 20, '建筑', 6, False, False, 0xc00080),
        ApolloScapeClass('bridge', 98, 255, '建筑', 6, False, True, 0x808000),
        ApolloScapeClass('tunnel', 99, 255, '建筑', 6, False, True, 0x800000),
        ApolloScapeClass('overpass', 100, 255, '建筑', 6, False, True, 0x408040),
        ApolloScapeClass('vegatation', 113, 21, '自然', 7, False, False, 0x808040),
        ApolloScapeClass('unlabeled', 255, 255, '未标注', 8, False, True, 0xFFFFFF),
    ]

    def __init__(self, root, split='train', mode='fine', target_type='instance',
                 transform=None, target_transform=None, transforms=None):
        super(Cityscapes, self).__init__(
            root, transforms, transform, target_transform)
        self.mode = 'gtFine' if mode == 'fine' else 'gtCoarse'
        self.images_dir = os.path.join(self.root, 'leftImg8bit', split)
        self.targets_dir = os.path.join(self.root, self.mode, split)
        self.target_type = target_type
        self.split = split
        self.images = []
        self.targets = []

        verify_str_arg(mode, "mode", ("fine", "coarse"))
        if mode == "fine":
            valid_modes = ("train", "test", "val")
        else:
            valid_modes = ("train", "train_extra", "val")
        msg = ("Unknown value '{}' for argument split if mode is '{}'. "
               "Valid values are {{{}}}.")
        msg = msg.format(split, mode, iterable_to_str(valid_modes))
        verify_str_arg(split, "split", valid_modes, msg)

        if not isinstance(target_type, list):
            self.target_type = [target_type]
        [verify_str_arg(value, "target_type",
                        ("instance", "semantic", "polygon", "color"))
         for value in self.target_type]

        if not os.path.isdir(self.images_dir) or not os.path.isdir(self.targets_dir):

            if split == 'train_extra':
                image_dir_zip = os.path.join(
                    self.root, 'leftImg8bit{}'.format('_trainextra.zip'))
            else:
                image_dir_zip = os.path.join(
                    self.root, 'leftImg8bit{}'.format('_trainvaltest.zip'))

            if self.mode == 'gtFine':
                target_dir_zip = os.path.join(
                    self.root, '{}{}'.format(self.mode, '_trainvaltest.zip'))
            elif self.mode == 'gtCoarse':
                target_dir_zip = os.path.join(
                    self.root, '{}{}'.format(self.mode, '.zip'))

            if os.path.isfile(image_dir_zip) and os.path.isfile(target_dir_zip):
                extract_archive(from_path=image_dir_zip, to_path=self.root)
                extract_archive(from_path=target_dir_zip, to_path=self.root)
            else:
                raise RuntimeError('Dataset not found or incomplete. Please make sure all required folders for the'
                                   ' specified "split" and "mode" are inside the "root" directory')

        for city in os.listdir(self.images_dir):
            img_dir = os.path.join(self.images_dir, city)
            target_dir = os.path.join(self.targets_dir, city)
            for file_name in os.listdir(img_dir):
                target_types = []
                for t in self.target_type:
                    target_name = '{}_{}'.format(file_name.split('_leftImg8bit')[0],
                                                 self._get_target_suffix(self.mode, t))
                    target_types.append(os.path.join(target_dir, target_name))

                self.images.append(os.path.join(img_dir, file_name))
                self.targets.append(target_types)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise target is a json object if target_type="polygon", else the image segmentation.
        """

        image = Image.open(self.images[index]).convert('RGB')

        targets = []
        for i, t in enumerate(self.target_type):
            if t == 'polygon':
                target = self._load_json(self.targets[index][i])
            else:
                target = Image.open(self.targets[index][i])

            targets.append(target)

        target = tuple(targets) if len(targets) > 1 else targets[0]

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self):
        return len(self.images)

    def extra_repr(self):
        lines = ["Split: {split}", "Mode: {mode}", "Type: {target_type}"]
        return '\n'.join(lines).format(**self.__dict__)

    def _load_json(self, path):
        with open(path, 'r') as file:
            data = json.load(file)
        return data

    def _get_target_suffix(self, mode, target_type):
        if target_type == 'instance':
            return '{}_instanceIds.png'.format(mode)
        elif target_type == 'semantic':
            return '{}_labelIds.png'.format(mode)
        elif target_type == 'color':
            return '{}_color.png'.format(mode)
        else:
            return '{}_polygons.json'.format(mode)


###################

#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Zpark labels"""


# --------------------------------------------------------------------------------
# Definitions
# --------------------------------------------------------------------------------

# a label and all meta information
Label = namedtuple('Label', [

    'name',  # The identifier of this label, e.g. 'car', 'person', ... .
    # We use them to uniquely name a class
    'clsId',

    'id',  # An integer ID that is associated with this label.
    # The IDs are used to represent the label in ground truth images
    # An ID of -1 means that this label does not have an ID and thus
    # is ignored when creating ground truth images (e.g. license plate).
    # Do not modify these IDs, since exactly these IDs are expected by the
    # evaluation server.

    'trainId',  # Feel free to modify these IDs as suitable for your method. Then create
    # ground truth images with train IDs, using the tools provided in the
    # 'preparation' folder. However, make sure to validate or submit results
    # to our evaluation server using the regular IDs above!
    # For trainIds, multiple labels might have the same ID. Then, these labels
    # are mapped to the same class in the ground truth images. For the inverse
    # mapping, we use the label that is defined first in the list below.
    # For example, mapping all void-type classes to the same ID in training,
    # might make sense for some approaches.
    # Max value is 255!

    'category',  # The name of the category that this label belongs to

    'categoryId',  # The ID of this category. Used to create ground truth images
    # on category level.

    'hasInstances',  # Whether this label distinguishes between single instances or not

    'ignoreInEval',  # Whether pixels having this class as ground truth label are ignored
    # during evaluations or not

    'color',  # The color of this label
])


# --------------------------------------------------------------------------------
# A list of all labels
# --------------------------------------------------------------------------------

# Please adapt the train IDs as appropriate for you approach.
# Note that you might want to ignore labels with ID 255 during training.
# Further note that the current train IDs are only a suggestion. You can use whatever you like.
# Make sure to provide your results using the original IDs and not the training IDs.
# Note that many IDs are ignored in evaluation and thus you never need to predict these!

labels = [
    #     name                    clsId    id   trainId   category  catId  hasInstanceignoreInEval   color
    Label('others',    0,    0,   255, '其他',   0, False, True, 0x000000),
    Label('rover', 0x01,    1,   255, '其他',   0, False, True, 0X000000),
    Label('sky', 0x11,   17,    0, '天空',   1, False, False, 0x4682B4),
    Label('car', 0x21,   33,    1, '移动物体',   2, True, False, 0x00008E),
    Label('car_groups', 0xA1,  161,    1, '移动物体',   2, True, False, 0x00008E),
    Label('motorbicycle', 0x22,   34,    2,
          '移动物体',   2, True, False, 0x0000E6),
    Label('motorbicycle_group', 0xA2,  162,    2,
          '移动物体',   2, True, False, 0x0000E6),
    Label('bicycle', 0x23,   35,    3, '移动物体',   2, True, False, 0x770B20),
    Label('bicycle_group', 0xA3,  163,    3,
          '移动物体',   2, True, False, 0x770B20),
    Label('person', 0x24,   36,    4, '移动物体',   2, True, False, 0x0080c0),
    Label('person_group', 0xA4,  164,    4,
          '移动物体',   2, True, False, 0x0080c0),
    Label('rider', 0x25,   37,    5, '移动物体',   2, True, False, 0x804080),
    Label('rider_group', 0xA5,  165,    5, '移动物体',   2, True, False, 0x804080),
    Label('truck', 0x26,   38,    6, '移动物体',   2, True, False, 0x8000c0),
    Label('truck_group', 0xA6,  166,    6, '移动物体',   2, True, False, 0x8000c0),
    Label('bus', 0x27,   39,    7, '移动物体',   2, True, False, 0xc00040),
    Label('bus_group', 0xA7,  167,    7, '移动物体',   2, True, False, 0xc00040),
    Label('tricycle', 0x28,   40,    8, '移动物体',   2, True, False, 0x8080c0),
    Label('tricycle_group', 0xA8,  168,    8,
          '移动物体',   2, True, False, 0x8080c0),
    Label('road', 0x31,   49,    9, '平面',   3, False, False, 0xc080c0),
    Label('siderwalk', 0x32,   50,    10, '平面',   3, False, False, 0xc08040),
    Label('traffic_cone', 0x41,   65,    11,
          '路间障碍',   4, False, False, 0x000040),
    Label('road_pile', 0x42,   66,    12, '路间障碍',   4, False, False, 0x0000c0),
    Label('fence', 0x43,   67,    13, '路间障碍',   4, False, False, 0x404080),
    Label('traffic_light', 0x51,   81,    14,
          '路边物体',   5, False, False, 0xc04080),
    Label('pole', 0x52,   82,    15, '路边物体',   5, False, False, 0xc08080),
    Label('traffic_sign', 0x53,   83,    16,
          '路边物体',   5, False, False, 0x004040),
    Label('wall', 0x54,   84,    17, '路边物体',   5, False, False, 0xc0c080),
    Label('dustbin', 0x55,   85,    18, '路边物体',   5, False, False, 0x4000c0),
    Label('billboard', 0x56,   86,    19, '路边物体',   5, False, False, 0xc000c0),
    Label('building', 0x61,   97,    20, '建筑',   6, False, False, 0xc00080),
    Label('bridge', 0x62,   98,    255, '建筑',   6, False, True, 0x808000),
    Label('tunnel', 0x63,   99,    255, '建筑',   6, False, True, 0x800000),
    Label('overpass', 0x64,  100,    255, '建筑',   6, False, True, 0x408040),
    Label('vegatation', 0x71,  113,    21, '自然',   7, False, False, 0x808040),
    Label('unlabeled', 0xFF,  255,    255, '未标注',   8, False, True, 0xFFFFFF),
]


# --------------------------------------------------------------------------------
# Create dictionaries for a fast lookup
# --------------------------------------------------------------------------------

# Please refer to the main method below for example usages!

# name to label object
name2label = {label.name: label for label in labels}
# id to label object
id2label = {label.id: label for label in labels}
# trainId to label object
trainId2label = {label.trainId: label for label in reversed(labels)}
# category to list of label objects
category2labels = {}
for label in labels:
    category = label.category
    if category in category2labels:
        category2labels[category].append(label)
    else:
        category2labels[category] = [label]
color2label = {}
for label in labels:
    #color = (int(label.color[2:4],16),int(label.color[4:6],16),int(label.color[6:8],16))
    color = label.color
    r = color // (256*256)
    g = (color-256*256*r) // 256
    b = (color-256*256*r-256*g)
    color2label[(r, g, b)] = [label]
# --------------------------------------------------------------------------------
# Assure single instance name
# --------------------------------------------------------------------------------

""" returns the label name that describes a single instance (if possible)
 e.g.     input     |   output
        ----------------------
          car       |   car
          cargroup  |   car
          foo       |   None
          foogroup  |   None
          skygroup  |   None
"""


def assureSingleInstanceName(name):
    # if the name is known, it is not a group
    if name in name2label:
        return name
    # test if the name actually denotes a group
    if not name.endswith("group"):
        return None
    # remove group
    name = name[:-len("group")]
    # test if the new name exists
    if not name in name2label:
        return None
    # test if the new name denotes a label that actually has instances
    if not name2label[name].hasInstances:
        return None
    # all good then
    return name

# --------------------------------------------------------------------------------
# Main for testing
# --------------------------------------------------------------------------------


# just a dummy main
if __name__ == "__main__":
    # Print all the labels
    print("List of cityscapes labels:")
    print("")
    print("    {:>21} | {:>3} | {:>7} | {:>14} |".format('name', 'id', 'trainId', 'category')
          + "{:>10} | {:>12} | {:>12}".format('categoryId', 'hasInstances', 'ignoreInEval'))
    print("    " + ('-' * 98))
    for label in labels:
        print("    {:>21} | {:>3} | {:>7} |".format(label.name, label.id, label.trainId)
              + "  {:>14} |{:>10} ".format(label.category, label.categoryId)
              + "| {:>12} | {:>12}".format(label.hasInstances, label.ignoreInEval))
    print("")

    print("Example usages:")

    # Map from name to label
    name = '机动车'
    id = name2label[name].id
    print("ID of label '{name}': {id}".format(name=name, id=id))

    # Map from ID to label
    category = id2label[id].category
    print("Category of label with ID '{id}': {category}".format(
        id=id, category=category))

    # Map from trainID to label
    trainId = 0
    name = trainId2label[trainId].name
    print("Name of label with trainID '{id}': {name}".format(
        id=trainId, name=name))
