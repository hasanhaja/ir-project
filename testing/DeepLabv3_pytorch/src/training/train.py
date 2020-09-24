from stream_metrics import StreamSegMetrics
from visualizer import Visualizer
import torch.nn as nn
from os import listdir
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import Cityscapes
import torch
from tqdm import tqdm
from modeling import *
from scheduler import *
import sys
sys.path.append('../')


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


dataset = "Cityscapes"

# Data augmentation and normalization for training
# Just normalization for validation
# TODO might need something for "test" as well
# TODO Consult this function `def get_dataset(opts)` at https://github.com/VainF/DeepLabV3Plus-Pytorch/blob/2ab9bfdafabfcd951ef02062c0d0aafe01aabd8b/main.py#L97 for more guidance
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

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
# Cityscapes

cityscapes_dir = "../../../datasets/Cityscapes"

train_dst = Cityscapes(cityscapes_dir, split="train", mode="fine",
                       target_type="semantic", transform=data_transforms['train'])
val_dst = Cityscapes(cityscapes_dir, split="val", mode="fine",
                     target_type="semantic", transform=data_transforms['val'])

# Debug
# test_dst = Cityscapes(cityscapes_dir, split="val", mode="fine", target_type="semantic")

train_loader = DataLoader(train_dst, batch_size=4, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dst, batch_size=4, shuffle=True, num_workers=4)

# Debug
# print(f"Total images in fine: {len(train_dst.images) + len(val_dst.images) + len(test_dst.images)}")      # Output 3975

print(
    f"[INFO]\nDataset: {dataset}\nTrain set: {len(train_dst)}\nVal set: {len(val_dst)}")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f"[INFO]\nDevice: {device}")

# Get model
output_stride = 16
model = deeplabv3plus_resnet101(num_classes=19, output_stride=output_stride)

set_bn_momentum(model.backbone, momentum=0.01)

# Set up optimizer
optimizer = torch.optim.SGD(params=[
    {'params': model.backbone.parameters(), 'lr': 0.1*opts.lr},
    {'params': model.classifier.parameters(), 'lr': opts.lr},
], lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)

# Set up scheduler and criterion: Using the defaults

scheduler = PolyLR(optimizer, opts.total_itrs, power=0.9)

criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')

# Train eval metrics
total_itrs = 30e3
val_interval = 100

best_score = 0.0
cur_itrs = 0
cur_epochs = 0

while True:  # cur_itrs < opts.total_itrs:
    # =====  Train  =====
    model.train()
    cur_epochs += 1
    for (images, labels) in train_loader:
        cur_itrs += 1

        images = images.to(device, dtype=torch.float32)
        labels = labels.to(device, dtype=torch.long)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        np_loss = loss.detach().cpu().numpy()
        interval_loss += np_loss

        # if vis is not None:
        #         vis.vis_scalar('Loss', cur_itrs, np_loss)

        if (cur_itrs) % 10 == 0:
            interval_loss = interval_loss/10
            print("Epoch %d, Itrs %d/%d, Loss=%f" %
                  (cur_epochs, cur_itrs, total_itrs, interval_loss))
            interval_loss = 0.0

        if (cur_itrs) % val_interval == 0:
            save_ckpt('checkpoints/latest_%s_%s_os%d.pth' %
                      (model, dataset, output_stride))
            print("validation...")
            model.eval()
            val_score, ret_samples = validate(
                model=model, loader=val_loader, device=device, metrics=metrics, ret_samples_ids=vis_sample_id)
            print(metrics.to_str(val_score))
            if val_score['Mean IoU'] > best_score:  # save best model
                best_score = val_score['Mean IoU']
                save_ckpt('checkpoints/best_%s_%s_os%d.pth' %
                          (model, dataset, output_stride))

            # if vis is not None:  # visualize validation score and samples
            #     vis.vis_scalar("[Val] Overall Acc", cur_itrs,
            #                    val_score['Overall Acc'])
            #     vis.vis_scalar("[Val] Mean IoU", cur_itrs,
            #                    val_score['Mean IoU'])
            #     vis.vis_table("[Val] Class IoU", val_score['Class IoU'])

            #     for k, (img, target, lbl) in enumerate(ret_samples):
            #         img = (denorm(img) * 255).astype(np.uint8)
            #         target = train_dst.decode_target(
            #             target).transpose(2, 0, 1).astype(np.uint8)
            #         lbl = train_dst.decode_target(
            #             lbl).transpose(2, 0, 1).astype(np.uint8)
            #         concat_img = np.concatenate(
            #             (img, target, lbl), axis=2)  # concat along width
            #         vis.vis_image('Sample %d' % k, concat_img)
            model.train()

        scheduler.step()

        if cur_itrs >= total_itrs:
            break