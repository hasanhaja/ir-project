from training.stream_metrics import StreamSegMetrics
from training.visualizer import Visualizer
import torch.nn as nn
from os import listdir
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import Cityscapes
from training.apolloscape import Apolloscape
import torch
from tqdm import tqdm
from modeling import *
from training.scheduler import *
import ext_transforms as et
import copy
import time
# import sys
# sys.path.append('../')


def set_bn_momentum(model, momentum=0.1):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = momentum


def save_ckpt(path):
    """ save current model
    """
    torch.save({
        "cur_itrs": cur_itrs,
        "model_state": model.module.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "best_score": best_score,
    }, path)
    print("Model saved as %s" % path)


def get_dataset(dataset, data_root, crop_size):
    """ Dataset And Augmentation
    """
    root_full_path = os.path.join(data_root)

    train_transform = et.ExtCompose([
        # et.ExtResize( 512 ),
        et.ExtRandomCrop(size=(crop_size, crop_size)),
        et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        et.ExtRandomHorizontalFlip(),
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])

    val_transform = et.ExtCompose([
        # et.ExtResize( 512 ),
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])

    if dataset.lower() == "cityscapes":
        print(f"[INFO] Fetching Cityscapes dataset from: {root_full_path}")
        train_dst = Cityscapes(root=data_root,
                               split='train', transform=train_transform)
        val_dst = Cityscapes(root=data_root,
                             split='val', transform=val_transform)
    else:
        print(f"[INFO] Fetching ApolloScape dataset from: {root_full_path}")
        train_dst = Apolloscape(root=root_full_path, road="road02_seg", transform=train_transform,
                                normalize_poses=True, pose_format='quat', train=True, cache_transform=True, stereo=False)

        val_dst = Apolloscape(root=root_full_path, road="road02_seg",
                              transform=val_transform, normalize_poses=True, pose_format='quat', train=False, cache_transform=True, stereo=False)

    return train_dst, val_dst


def validate(model, loader, device, metrics, ret_samples_ids=None):
    """Do validation and return specified samples"""
    metrics.reset()
    ret_samples = []

    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(loader)):

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(images)
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()

            metrics.update(targets, preds)
            if ret_samples_ids is not None and i in ret_samples_ids:  # get vis samples
                ret_samples.append(
                    (images[0].detach().cpu().numpy(), targets[0], preds[0]))

        score = metrics.get_results()
    return score, ret_samples


def dataset_config(dataset):
    if dataset.lower() == "cityscapes":
        return ("Cityscapes", "../../datasets/Cityscapes")
    elif dataset.lower() == "apolloscape":
        return ("ApolloScape", "../../datasets/ApolloScape")
    else:
        raise NameError(
            f"[ERROR] {dataset} not recognized. Use either \"Cityscapes\" or \"ApolloScape\".")


def main():
    # Setting up metrics and visualization
    enable_vis = False
    vis_port = 28333
    vis_env = "main"
    vis = Visualizer(port=vis_port,
                     env=vis_env) if enable_vis else None
    metrics = StreamSegMetrics(19)

    # vis_sample_id = np.random.randint(0, len(val_loader), opts.vis_num_samples,
    #   np.int32) if opts.enable_vis else None
    vis_sample_id = None

    # Loading data
    dataset, dataset_dir = dataset_config("cityscapes")

    train_dst, val_dst = get_dataset(dataset, dataset_dir, 768)

    # Debug
    # test_dst = Cityscapes(cityscapes_dir, split="val", mode="fine", target_type="semantic")
    batch_size = 16

    train_loader = DataLoader(train_dst, batch_size=batch_size,
                              shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dst, batch_size=batch_size,
                            shuffle=True, num_workers=2)

    print(train_loader)

    return

    # TODO verify if the dimensions of the dataloaders are the same.
    # TODO verify if the labeling is correct

    dataloaders = {"train": train_loader, "val": val_loader}

    dataset_sizes = {"train": len(train_dst), "val": len(val_dst)}
    class_names = train_dst.classes

    # print(dataset_sizes)
    # print(class_names)

    # Debug
    # print(f"Total images in fine: {len(train_dst.images) + len(val_dst.images) + len(test_dst.images)}")      # Output 3975

    print(
        f"[INFO]\nDataset: {dataset}\nTrain set: {len(train_dst)}\nVal set: {len(val_dst)}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"[INFO]\nDevice: {device}")

    # Get model
    output_stride = 16
    model = deeplabv3plus_resnet101(
        num_classes=21, output_stride=output_stride)

    model.load_state_dict(torch.load("training/pretrained_weights/best_deeplabv3plus_resnet101_voc_os16.pth",
                                     map_location=torch.device('cpu'))["model_state"])

    set_bn_momentum(model.backbone, momentum=0.01)

    # print(model)

    """
    (classifier): Sequential(
      (0): Conv2d(304, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU(inplace=True)
      (3): Conv2d(256, 21, kernel_size=(1, 1), stride=(1, 1))
    )
    """

    # num_ftrs = model.classifier.in_features

    # generalize this
    # model.classifier[3] = nn.Conv2d(256, 19, kernel_size = (1, 1), stride=(1, 1))

    # TODO output the cityscapes model like this and compared

    print(model)

    return

    # TODO find the corresponding paper
    # TODO check with v3 and other implementations

    model.classifier = nn.Sequential(
        nn.Conv2d(304, 256, kernel_size=(3, 3), stride=(
            1, 1), padding=(1, 1), bias=False),
        nn.BatchNorm2d(256, eps=1e-05, momentum=0.1,
                       affine=True, track_running_stats=True),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 19, kernel_size=(1, 1), stride=(1, 1))
    )

    model = model.to(device)

    # Set up optimizer
    lr = 0.1
    weight_decay = 1e-4
    total_itrs = 30e3
    val_interval = 100

    best_score = 0.0
    cur_itrs = 0
    cur_epochs = 0

    optimizer = torch.optim.SGD(params=[
        {'params': model.backbone.parameters(), 'lr': 0.1*lr},
        {'params': model.classifier.parameters(), 'lr': lr},
    ], lr=lr, momentum=0.9, weight_decay=weight_decay)

    # Set up scheduler and criterion: Using the defaults

    scheduler = PolyLR(optimizer, total_itrs, power=0.9)

    criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')

    # Train eval metrics
    model = train_model(device, dataloaders, model, criterion, optimizer, scheduler,
                        num_epochs=1)

    print(model)

    # images, labels = train_loader

    # print(images)

    # while True:  # cur_itrs < opts.total_itrs:
    #     # =====  Train  =====
    #     model.train()
    #     cur_epochs += 1

    #     print(model)

    #     return
    #     # TODO Something is wrong with the for loop line with train_loader, missing arg lbl or something

    #     for (images, labels) in train_loader:
    #         cur_itrs += 1

    #         images = images.to(device, dtype=torch.float32)
    #         labels = labels.to(device, dtype=torch.long)

    #         optimizer.zero_grad()
    #         outputs = model(images)
    #         loss = criterion(outputs, labels)
    #         loss.backward()
    #         optimizer.step()

    #         np_loss = loss.detach().cpu().numpy()
    #         interval_loss += np_loss

    #         # if vis is not None:
    #         #         vis.vis_scalar('Loss', cur_itrs, np_loss)

    #         if (cur_itrs) % 10 == 0:
    #             interval_loss = interval_loss/10
    #             print("Epoch %d, Itrs %d/%d, Loss=%f" %
    #                   (cur_epochs, cur_itrs, total_itrs, interval_loss))
    #             interval_loss = 0.0

    #         if (cur_itrs) % val_interval == 0:
    #             save_ckpt('training/checkpoints/latest_%s_%s_os%d.pth' %
    #                       (model, dataset, output_stride))
    #             print("validation...")
    #             model.eval()
    #             val_score, ret_samples = validate(
    #                 model=model, loader=val_loader, device=device, metrics=metrics, ret_samples_ids=vis_sample_id)
    #             print(metrics.to_str(val_score))
    #             if val_score['Mean IoU'] > best_score:  # save best model
    #                 best_score = val_score['Mean IoU']
    #                 save_ckpt('training/checkpoints/best_%s_%s_os%d.pth' %
    #                           (model, dataset, output_stride))

    #             # if vis is not None:  # visualize validation score and samples
    #             #     vis.vis_scalar("[Val] Overall Acc", cur_itrs,
    #             #                    val_score['Overall Acc'])
    #             #     vis.vis_scalar("[Val] Mean IoU", cur_itrs,
    #             #                    val_score['Mean IoU'])
    #             #     vis.vis_table("[Val] Class IoU", val_score['Class IoU'])

    #             #     for k, (img, target, lbl) in enumerate(ret_samples):
    #             #         img = (denorm(img) * 255).astype(np.uint8)
    #             #         target = train_dst.decode_target(
    #             #             target).transpose(2, 0, 1).astype(np.uint8)
    #             #         lbl = train_dst.decode_target(
    #             #             lbl).transpose(2, 0, 1).astype(np.uint8)
    #             #         concat_img = np.concatenate(
    #             #             (img, target, lbl), axis=2)  # concat along width
    #             #         vis.vis_image('Sample %d' % k, concat_img)
    #             model.train()

    #         scheduler.step()

    #         if cur_itrs >= total_itrs:
    #             return


def train_model(device, dataloaders, model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                # inputs = inputs.to(device, dtype=torch.float32)
                # labels = labels.to(device, dtype=torch.long)

                # # zero the parameter gradients
                # optimizer.zero_grad()

                # # forward
                # # track history if only in train
                # with torch.set_grad_enabled(phase == 'train'):
                #     outputs = model(inputs)
                #     _, preds = torch.max(outputs, 1)
                #     loss = criterion(outputs, labels)

                #     # backward + optimize only if in training phase
                #     if phase == 'train':
                #         loss.backward()
                #         optimizer.step()

                # # statistics
                # running_loss += loss.item() * inputs.size(0)
                # running_corrects += torch.sum(preds == labels.data)
                print("In for loop")

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


if __name__ == "__main__":
    main()
