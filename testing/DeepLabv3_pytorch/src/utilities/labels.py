from utilities.util import flatten

__all__ = ["CityscapesLabels", "ApolloScapeLabels"]


class CityscapesLabels():
    """
    The purpose of this class is to return all the relevant labels that will be used for training. The labels are hardcoded based on the labels provide on the [Cityscapes website](https://www.cityscapes-dataset.com/dataset-overview/#features).
    """

    def __init__(self):
        self.flat = ["road", "sidewalk", "parking", "rail track"]
        self.human = ["person", "rider"]
        self.vehicle = ["car", "truck", "bus", "on rails",
                        "motorcycle", "bicycle", "caravan", "trailer"]
        self.construction = ["building", "wall",
                             "fence", "guard rail", "bridge", "tunnel"]
        self.objects = ["pole", "pole group", "traffic sign", "traffic light"]
        self.nature = ["vegetation", "terrain"]
        self.sky = ["sky"]
        self.void = ["ground", "dynamic", "static"]

        self.labels_ignored_for_evaluation = ["dynamic", "static", "pole group", "guard rail",
                                              "bridge", "tunnel", "caravan", "trailer", "rider", "parking", "rail track"]

    def groups(self):
        return ["flat", "human", "vehicle", "construction", "objects", "nature", "sky", "void"]

    def labels(self, all_labels=False):
        labels = [self.flat, self.human, self.vehicle, self.construction,
                  self.objects, self.nature, self.sky, self.void]
        labels = flatten(labels)

        return labels if all_labels else list(filter(lambda label: label not in self.labels_ignored_for_evaluation, labels))

class ApolloScapeLabels():

    def __init__(self):
        self.name = "Something"