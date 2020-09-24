import torch
from torchvision.datasets import Cityscapes
from torchvision import transforms
from os import listdir

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

# Loading data 
# Cityscapes

cityscapes_dir = "../../../datasets/Cityscapes"

# dataset = Cityscapes(cityscapes_dir, split="train", mode="fine", target_type="semantic", transform=data_transforms)

# print(dataset)

# Debug [PASSED]
# DONE What does this return? What is being destructured here? _Just the dataset, and this needs to be passed into the
# img, smnt = dataset[0]

# print(len(dataset.classes))
# test = [clazz for clazz in dataset.classes if clazz.ignore_in_eval == False]
# print(len(test))  # Output is 19
# print(len(dataset.images))    # Output is 2975
# print(img)
# print("Done.")

# TODO You need to call the function Cityscapes twice. Once for train and another for val.
image_datasets = {x: Cityscapes(cityscapes_dir, split="train", mode="fine", target_type="semantic", transform=data_transforms[x]) for x in ['train', 'val']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4) for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

# print(dataset_sizes)  # dataset duplicated and not split lol
# print(class_names)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
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
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
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