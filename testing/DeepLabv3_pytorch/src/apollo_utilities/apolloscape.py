import json
import os
from collections import namedtuple
from recordtype import recordtype
import zipfile

from torchvision.datasets.utils import extract_archive, verify_str_arg, iterable_to_str
from torchvision.datasets.vision import VisionDataset
from PIL import Image

import glob
import numpy as np

# TODO refactor to read_labels_for_camera


def read_poses_for_camera(record_path, camera_name):
    """Finds and reads poses file for camera_name."""

    # Resolve pose.txt file path for camera
    poses_path = os.path.join(record_path, camera_name, 'pose.txt')
    if os.path.exists(poses_path):
        poses = read_poses_dict(poses_path)
    else:
        # Sample type dataset (aka zpark-sample)
        poses_path = os.path.join(record_path, camera_name + '.txt')
        poses = read_poses_dict_6(poses_path)
    return poses


def read_all_data(image_dir, pose_dir, records_list, cameras_list, apollo_original_order=False, stereo=False):
    # iterate over all records and store it in internal data
    #     data = []
    d_images = []
    d_poses = np.empty((0, 4, 4))
    d_records = []
    skipped_inc = 0
    skipped_other = 0
    for i, r in enumerate(records_list):
        cam1s = sorted(glob.glob(os.path.join(image_dir, r, cameras_list[0], '*.jpg')),
                       reverse=not apollo_original_order)
        cam2s = sorted(glob.glob(os.path.join(image_dir, r, cameras_list[1], '*.jpg')),
                       reverse=not apollo_original_order)

        # Read poses for first camera
        pose1s = read_poses_for_camera(
            os.path.join(pose_dir, r), cameras_list[0])

        # Read poses for second camera
        pose2s = read_poses_for_camera(
            os.path.join(pose_dir, r), cameras_list[1])

        c1_idx = 0
        c2_idx = 0
        while c1_idx < len(cam1s) and c2_idx < len(cam2s):
            c1 = cam1s[c1_idx]
            c2 = cam2s[c2_idx]

            # Check stereo image path consistency
            im1 = os.path.basename(c1).split('_')
            im2 = os.path.basename(c2).split('_')
            im1_part = '_'.join(im1[:2])
            im2_part = '_'.join(im2[:2])

            if stereo and im1_part != im2_part:
                # Non-consistent images, drop with the lowest time unit
                # and repeat with the next idx
                skipped_inc += 1
                if im1_part < im2_part:
                    c1_idx += 1
                else:
                    c2_idx += 1
            else:

                # Images has equal timing (filename prefix) so add them to data.

                # First image
                d_images.append(c1)
                d_poses = np.vstack((d_poses, np.expand_dims(
                    pose1s[os.path.basename(c1)], axis=0)))
                d_records.append(r)

                # Second image
                d_images.append(c2)
                d_poses = np.vstack((d_poses, np.expand_dims(
                    pose2s[os.path.basename(c2)], axis=0)))
                d_records.append(r)

                # Continue with the next pair of images
                c1_idx += 1
                c2_idx += 1
    return np.array(d_images), d_poses, np.array(d_records)

# TODO atm, the entire dataset is being read and it is not being split at all.


def hex_to_rgb(color):
    r = color // (256*256)
    g = (color-256*256*r) // 256
    b = (color-256*256*r-256*g)
    return (r, g, b)


class Apolloscape(VisionDataset):
    # TODO make void the classes not considered by cityscapes

    ApolloScapeClass = recordtype('ApolloScapeClass', [
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
        ApolloScapeClass('others', 2, 255, '其他', 0, False,
                         True, hex_to_rgb(0x000000)),
        ApolloScapeClass('rover', 1, 255, '其他', 0, False,
                         True, hex_to_rgb(0X000000)),
        ApolloScapeClass('sky', 23, 10, '天空', 1, False,
                         False, hex_to_rgb(0x4682B4)),
        ApolloScapeClass('car', 26, 13, '移动物体', 2, True,
                         False, hex_to_rgb(0x00008E)),
        ApolloScapeClass('car_groups', 18, 255, '移动物体',
                         2, True, False, hex_to_rgb(0x00008E)),
        ApolloScapeClass('motorcycle', 32, 17, '移动物体',
                         2, True, False, hex_to_rgb(0x0000E6)),
        ApolloScapeClass('motorcycle_group', 22, 255,
                         '移动物体', 2, True, False, hex_to_rgb(0x0000E6)),
        ApolloScapeClass('bicycle', 33, 18, '移动物体', 2, True,
                         False, hex_to_rgb(0x770B20)),
        ApolloScapeClass('bicycle_group', 29, 255, '移动物体',
                         2, True, False, hex_to_rgb(0x770B20)),
        ApolloScapeClass('person', 24, 11, '移动物体', 2, True,
                         False, (220, 20, 60)),
        ApolloScapeClass('person_group', 30, 255, '移动物体',
                         2, True, False, (220, 20, 60)),
        ApolloScapeClass('rider', 25, 12, '移动物体', 2, True,
                         False, (255, 0, 0)),
        ApolloScapeClass('rider_group', 31, 255, '移动物体',
                         2, True, False, (255, 0, 0)),
        ApolloScapeClass('truck', 27, 14, '移动物体', 2, True,
                         False, (0, 0, 70)),
        ApolloScapeClass('truck_group', 34, -1, '移动物体',
                         2, True, False, (0, 0, 70)),
        ApolloScapeClass('bus', 28, 15, '移动物体', 2, True,
                         False, (0, 60, 100)),
        ApolloScapeClass('bus_group', 35, 15, '移动物体',
                         2, True, False, (0, 60, 100)),
        ApolloScapeClass('tricycle', 3, 255, '移动物体', 2,
                         True, False, hex_to_rgb(0x8080c0)),
        ApolloScapeClass('tricycle_group', 14, 255, '移动物体',
                         2, True, False, hex_to_rgb(0x8080c0)),
        ApolloScapeClass('road', 7, 0, '平面', 3, False,
                         False, (128, 64, 148)),
        ApolloScapeClass('sidewalk', 8, 1, '平面', 3,
                         False, False, (244, 35, 232)),
        ApolloScapeClass('traffic_cone', 4, 255, '路间障碍',
                         4, False, False, hex_to_rgb(0x000040)),
        ApolloScapeClass('road_pile', 5, 255, '路间障碍',
                         4, False, False, hex_to_rgb(0x0000c0)),
        ApolloScapeClass('fence', 13, 4, '路间障碍', 4, False,
                         False, (190, 153, 153)),
        ApolloScapeClass('traffic_light', 19, 6, '路边物体',
                         5, False, False, (250, 170, 30)),
        ApolloScapeClass('pole', 17, 5, '路边物体', 5, False,
                         False, (153, 153, 153)),
        ApolloScapeClass('traffic_sign', 20, 7, '路边物体',
                         5, False, False, (220, 220, 0)),
        ApolloScapeClass('wall', 12, 3, '路边物体', 5, False,
                         False, (102, 102, 156)),
        ApolloScapeClass('dustbin', 6, 255, '路边物体', 5,
                         False, False, hex_to_rgb(0x4000c0)),
        ApolloScapeClass('billboard', 9, 255, '路边物体',
                         5, False, False, hex_to_rgb(0xc000c0)),
        ApolloScapeClass('building', 11, 2, '建筑', 6, False,
                         False, (70, 70, 70)),
        ApolloScapeClass('bridge', 15, 255, '建筑', 6, False,
                         True, (150, 100, 100)),
        ApolloScapeClass('tunnel', 16, 255, '建筑', 6, False,
                         True, (150, 120, 90)),
        ApolloScapeClass('overpass', 10, 255, '建筑', 6,
                         False, True, hex_to_rgb(0x408040)),
        ApolloScapeClass('vegetation', 21, 8, '自然',
                         7, False, False, (107, 142, 35)),
        ApolloScapeClass('unlabeled', 0, 255, '未标注',
                         8, False, True, (0, 0, 0)),
    ]

    classes = sorted(classes, key=lambda clazz: clazz.id)

    # Cityscape to apolloscape mapping to undo what I've done:
    Mapper = recordtype("Mapper", ["name", "cs_id", "as_id"])
    ids = [
        Mapper('others', 2,    0),
        Mapper('rover', 1,    1),
        Mapper('sky', 23,   17),
        Mapper('car', 26,   33),
        Mapper('car_groups', 18,  161),
        Mapper('motorcycle', 32,   34),
        Mapper('motorcycle_group', 22,  162),
        Mapper('bicycle', 33,   35),
        Mapper('bicycle_group', 29,  163),
        Mapper('person', 24,   36),
        Mapper('person_group', 30,  164),
        Mapper('rider', 25,   37),
        Mapper('rider_group', 31,  165),
        Mapper('truck', 27,   38),
        Mapper('truck_group', -1,  166),
        Mapper('bus', 28,   39),
        Mapper('bus_group', 35,  167),
        Mapper('tricycle', 3,   40),
        Mapper('tricycle_group', 14,  168),
        Mapper('road', 7,   49),
        Mapper('sidewalk', 8,   50),
        Mapper('traffic_cone', 4,   65),
        Mapper('road_pile', 5,   66),
        Mapper('fence', 13,   67),
        Mapper('traffic_light', 19,   81),
        Mapper('pole', 17,   82),
        Mapper('traffic_sign', 20,   83),
        Mapper('wall', 12,   84),
        Mapper('dustbin', 6,   85),
        Mapper('billboard', 9,   86),
        Mapper('building', 11,   97),
        Mapper('bridge', 15,   98),
        Mapper('tunnel', 16,   99),
        Mapper('overpass', 10,  100),
        Mapper('vegetation', 21,  113),
        Mapper('unlabeled', 0,  255),
    ]

    def apollo_id_to_cs_id(id, ids=ids, Mapper=Mapper):
        # cs_id = 255 if id == 254 else id
        cs_id = id
        # TODO select nearest known id
        # TODO keep a list of (elem, score)
        if id == 17:
            print("Sky...")

        nearest = (300, 300)

        for mapper in ids:
            if mapper.as_id == id:
                cs_id = mapper.cs_id
                # print(f"Mapping {mapper}; CS ID: {cs_id}")
                return cs_id

            else:
                # TODO might have to switch around
                # print(f"INFO Estimating nearest ID for [{id}]...")
                if nearest[1] > (mapper.as_id - id):
                    nearest = (mapper.as_id, id - mapper.as_id)
                else:
                    continue

        for mapper in ids:
            if mapper.as_id == nearest[0]:
                cs_id = mapper.cs_id
                print(
                    f"Mapping {mapper}; CS ID: {cs_id}; Original AS ID: {id}")
        # print(f"CS ID: {cs_id}")
        return cs_id

    def apollo_classes_with_original_ids(classes=classes, ids=ids, Mapper=Mapper):
        updated_classes = classes

        for clazz in updated_classes:
            correct = ids[Mapper.cs_id == clazz.id]

            for mapper in ids:
                if mapper.cs_id == clazz.id:
                    correct = mapper.as_id

            # print(correct)
            clazz.id = correct

        return updated_classes

    # classes = apollo_classes_with_original_ids(classes)

    train_id_to_color = [c.color for c in classes if (
        c.train_id != -1 and c.train_id != 255)]
    # train_id_to_color = [c.color for c in classes if c.train_id != 255]
    # [print(c.name) for c in classes if (
    #     c.train_id != 255)]
    train_id_to_color.append([0, 0, 0])
    train_id_to_color = np.array(train_id_to_color)

    classes = apollo_classes_with_original_ids(classes)

    id_to_train_id = np.array([c.train_id for c in classes])
    # [print(f"DEBUG: AS ID {e}") for e in id_to_train_id]
    # print(f"DEBUG: AS max {max(id_to_train_id)}")

    def _construct_path_to_file(self, record, camera, image_name, img_type="image"):
        img_dir, ext = ("ColorImage", ".jpg") if img_type == "image" else (
            "Label", "_bin.png")
        path = os.path.join(self.root, self.road, img_dir, record, camera)
        file_path = f"{path}{image_name}{ext}"

        print(f"File path: {file_path}")

        return file_path

    def _construct_path(self, img_type="image"):
        return os.path.join(self.root, self.road, "ColorImage") if img_type == "image" else os.path.join(self.root, self.road, "Label")

    def __init__(self, root, road="road02_seg", split='train', target_type='semantic',
                 transform=None, target_transform=None, transforms=None):
        super(Apolloscape, self).__init__(
            root, transforms, transform, target_transform)

        self.road = road
        self.apollo_original_order = True
        self.target_type = target_type
        # TODO map the "train" param functionality to this, but not a priority
        self.split = split

        # This gets you to "ApolloScape/road02_seg/ColorImage"
        self.images_dir = self._construct_path()
        # This gets you to "ApolloScape/road02_seg/Label"
        self.targets_dir = self._construct_path("label")

        # Record: TODO might not need this
        self.records_list = [f for f in os.listdir(
            self.images_dir) if f not in [".DS_Store"]]
        self.records_list = sorted(self.records_list)
        # Camera
        self._cameras_list = sorted(os.listdir(os.path.join(self.images_dir, self.records_list[0])),
                                    reverse=not self.apollo_original_order)
        self.camera = self._cameras_list[0]

        # image
        # Populate these arrays with the images and the labels
        self.images = []
        self.targets = []

        # TODO this is for loop where all the images and the corresponding labels files come in
        # This gets you to "ApolloScape/road02_seg/ColorImage"
        for record in os.listdir(self.images_dir):

            # select based on record and the default camera

            img_dir = os.path.join(self.images_dir, record, self.camera)
            target_dir = os.path.join(self.targets_dir, record, self.camera)
            # ApolloScape works like this
            # root/{img_dir}/
            # general: road02_seg/{}/Record022/{}_Camera_{}{}
            # road02_seg/ColorImage/Record022/123412_234123_Camera_5.jpg
            # road02_seg/Label/Record022/123412_234123_Camera_5_bin.png
            # _bin.png can be the file extension.
            # TODO this last loop.
            for file_name in os.listdir(img_dir):
                # target_types = []
                # for t in self.target_type:
                #     target_name = '{}_{}'.format(file_name.split('_leftImg8bit')[0],
                #                                  self._get_target_suffix(self.mode, t))
                #     target_types.append(os.path.join(target_dir, target_name))

                file_root = os.path.basename(file_name).split(".jpg")[0]
                image_file = f"{file_root}.jpg"
                label_file = f"{file_root}_bin.png"

                self.images.append(os.path.join(img_dir, image_file))
                self.targets.append(os.path.join(target_dir, label_file))

        # TODO properly implement data splitting. Hard coding to 500 images.
        self.images = self.images[1000:1500]
        self.targets = self.targets[1000:1500]

    @classmethod
    def encode_target(cls, target):
        # print(f"DEBUG: AS Encode id {len(cls.id_to_train_id)}")
        target = np.array(target)
        # print(f"DEBUG: Target {target.shape}")

        transformer = np.vectorize(cls.apollo_id_to_cs_id)
        target = transformer(target)
        # target = np.array(list(map(lambda e: cls.apollo_id_to_cs_id(e), target)))
        # print(f"DEBUG: Target {target}")
        result = cls.id_to_train_id[target]
        # print(f"DEBUG: AS Encode target {result}")
        return result

    @classmethod
    def decode_target(cls, target):
        # print(f"What is this: {target[target == 255]}")
        # target[target == 255] = 19
        target[target == 255] = 18
        # print(f"What is this: {target[target == 255]}")
        #target = target.astype('uint8') + 1
        return cls.train_id_to_color[target]

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert('RGB')
        # NOTE This line is different for Cityscapes
        target = Image.open(self.targets[index])
        # print(f"DEBUG: Before transform Target - {target.mode}")
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        # print(f"DEBUG: Target - {type(target)}")
        target = self.encode_target(target)
        return image, target
    # def __getitem__(self, index):
    #     """
    #     Args:
    #         index (int): Index
    #     Returns:
    #         tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
    #         than one item. Otherwise target is a json object if target_type="polygon", else the image segmentation.
    #     """

    #     image = Image.open(self.images[index]).convert('RGB')

    #     # targets = []
    #     # for i, t in enumerate(self.target_type):
    #     #     if t == 'polygon':
    #     #         target = self._load_json(self.targets[index][i])
    #     #     else:
    #     #         target = Image.open(self.targets[index][i])

    #     #     targets.append(target)
    #     target = Image.open(self.targets[index])
    #     # target = tuple(targets) if len(targets) > 1 else targets[0]

    #     if self.transforms is not None:
    #         image, target = self.transforms(image, target)

    #     return image, target

    def __len__(self):
        return len(self.images)

    # TODO Update for apolloscape
    def extra_repr(self):
        lines = ["Split: {split}", "Mode: {mode}", "Type: {target_type}"]
        return '\n'.join(lines).format(**self.__dict__)

    # # TODO Update for apolloscape
    # def _load_json(self, path):
    #     with open(path, 'r') as file:
    #         data = json.load(file)
    #     return data

    # # TODO Update for apolloscape
    # def _get_target_suffix(self, mode, target_type):
    #     # if target_type == 'instance':
    #     #     # return '{}_instanceIds.png'.format(mode)
    #     # elif target_type == 'semantic':
    #     #     return '{}_labelIds.png'.format(mode)
    #     # elif target_type == 'color':
    #     #     return '{}_color.png'.format(mode)
    #     # else:
    #     #     return '{}_polygons.json'.format(mode)
    #     # TODO maybe instead of mode, it should be image file name
    #     if target_type == "semantic":
    #         return f"{mode}_bin.png"
    #     else:
    #         raise ValueError(
    #             "ERROR: Target type not supported. Try 'semantic'.")


###################


# --------------------------------------------------------------------------------
# Create dictionaries for a fast lookup
# --------------------------------------------------------------------------------

# Please refer to the main method below for example usages!

# # name to label object
# name2label = {label.name: label for label in labels}
# # id to label object
# id2label = {label.id: label for label in labels}
# # trainId to label object
# train_id2label = {label.trainId: label for label in reversed(labels)}
# # category to list of label objects
# category2labels = {}
# for label in labels:
#     category = label.category
#     if category in category2labels:
#         category2labels[category].append(label)
#     else:
#         category2labels[category] = [label]
# color2label = {}
# for label in labels:
#     #color = (int(label.color[2:4],16),int(label.color[4:6],16),int(label.color[6:8],16))
#     color = label.color
#     r = color // (256*256)
#     g = (color-256*256*r) // 256
#     b = (color-256*256*r-256*g)
#     color2label[(r, g, b)] = [label]
# # --------------------------------------------------------------------------------
# # Assure single instance name
# # --------------------------------------------------------------------------------

# """ returns the label name that describes a single instance (if possible)
#  e.g.     input     |   output
#         ----------------------
#           car       |   car
#           cargroup  |   car
#           foo       |   None
#           foogroup  |   None
#           skygroup  |   None
# """


# def assureSingleInstanceName(name):
#     # if the name is known, it is not a group
#     if name in name2label:
#         return name
#     # test if the name actually denotes a group
#     if not name.endswith("group"):
#         return None
#     # remove group
#     name = name[:-len("group")]
#     # test if the new name exists
#     if not name in name2label:
#         return None
#     # test if the new name denotes a label that actually has instances
#     if not name2label[name].has_instances:
#         return None
#     # all good then
#     return name

# # --------------------------------------------------------------------------------
# # Main for testing
# # --------------------------------------------------------------------------------


# # just a dummy main
# if __name__ == "__main__":
#     # Print all the labels
#     print("List of cityscapes labels:")
#     print("")
#     print("    {:>21} | {:>3} | {:>7} | {:>14} |".format('name', 'id', 'trainId', 'category')
#           + "{:>10} | {:>12} | {:>12}".format('categoryId', 'hasInstances', 'ignoreInEval'))
#     print("    " + ('-' * 98))
#     for label in labels:
#         print("    {:>21} | {:>3} | {:>7} |".format(label.name, label.id, label.train_id)
#               + "  {:>14} |{:>10} ".format(label.category, label.category_id)
#               + "| {:>12} | {:>12}".format(label.has_instances, label.ignore_in_eval))
#     print("")

#     print("Example usages:")

#     # Map from name to label
#     name = '机动车'
#     id = name2label[name].id
#     print("ID of label '{name}': {id}".format(name=name, id=id))

#     # Map from ID to label
#     category = id2label[id].category
#     print("Category of label with ID '{id}': {category}".format(
#         id=id, category=category))

#     # Map from trainID to label
#     train_id = 0
#     name = trainId2label[train_id].name
#     print("Name of label with trainID '{id}': {name}".format(
#         id=trainId, name=name))
