#20.38 start
#20.48 epoch 1 control
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
from tempfile import TemporaryDirectory
import torch.utils.data as ptdata

torch.set_default_dtype(torch.float32)

cudnn.benchmark = True
plt.ion()   # interactive mode to plot images during training

#* Data augmentation and normalization for training
#* Just normalization for validation
#* HSV
#* we will use mean and std of our dataset on normal training
#TODO kendi normalization ını deneyebilirsin
#TODO veri boyutu küçük imagenet devam
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256), #* resize the image to 256x256
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

#* blackwhite to rgb by tripling shape

data_dir = './data/torchvision_data_3D'

image_datasets = {}
for x in ['train', 'val']:
    image_datasets[x] = datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])

#batch size büyükten başla düşür
dataloaders = {}
for x in ['train', 'val']:
    dataloaders[x] = torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=0)

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

#* Use GPU if available else MPS else CPU
#* Implement Metal Performance Shaders (MPS) for faster training on Apple devices
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

print(f"Class names: {class_names}")
print(f"Dataset sizes: {dataset_sizes}")
print(f"Using device: {device}")

trainlosslist = []
vallosslist = []
trainacclist = []
valacclist = []

def train(model, criterion, optimizer, scheduler, num_epochs=25):


    since = time.time()

    #* Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        #* Save the best model weights in a temporary file
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

        #* Initialize the model weights
        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0

        for epoch in range(num_epochs):
            current_time = time.strftime("%H:%M", time.localtime())
            print(current_time)
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            #* Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  #* Set model to training mode
                else:
                    model.eval()   #* Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

#TODO Log lanacak değerler summarywriter ile
#TODO training loss
#TODO eval loss
#TODO learning rate
#TODO eval f1 score
#TODO tensorboard kullan
#TODO step e göre yada epoch a göre logla (x dimension)
#* tensorboard --logdir=log/

                #* Iterate over data (batches)
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    #* zero the parameter gradients
                    optimizer.zero_grad()

                    #* forward pass
                    #* track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        #* backward pass + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    #* statistics for loss and accuracy for each phase (train/val)
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.float() / dataset_sizes[phase]


                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                if phase == 'train':
                    trainlosslist.append(epoch_loss)
                    trainacclist.append(epoch_acc)
                elif phase == 'val':
                    vallosslist.append(epoch_loss)
                    valacclist.append(epoch_acc)

                #! create a checkpoint to save the model weights
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc

                #* deep copy the model
                torch.save(model.state_dict(), best_model_params_path)

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        #* load best model weights
        model.load_state_dict(torch.load(best_model_params_path, weights_only=True))
    return model

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))  # Convert tensor to a numpy array and rearrange dimensions for display
    mean = np.array([0.485, 0.456, 0.406])  # The mean used during normalization (for RGB images)
    std = np.array([0.229, 0.224, 0.225])   # The std deviation used during normalization (for RGB images)
    inp = std * inp + mean                  # De-normalize the image
    inp = np.clip(inp, 0, 1)                # Clip values to ensure they are between 0 and 1
    plt.imshow(inp)                         # Display the image
    if title is not None:
        plt.title(title)                    # Set the title if provided
    plt.pause(0.001)                        # Pause to ensure the plot updates

def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j]]}')
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

#* Load a pre-trained model and reset final fully connected layer
model_ft = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

#TODO freezing the weights of the model and only training the final layer is done next 4 lines
#TODO Play with that model to train more layers

#* Freeze all layers except the final layer
#TODO dondurmadan devam et
"""for param in model_ft.parameters():
    param.requires_grad = False"""

num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, len(class_names))

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

#* Observe that all parameters are being optimized
#TODO lr yi göz ayarı binary search deneyerek bul

optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.1, momentum=0.9) #0.004 dene

#* Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)

best_model = train(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=50)
print("Training completed")

print("Train loss list:")
print(trainlosslist)
print("Train accuracy list:")
print(trainacclist)

print("Validation loss list:")
print(vallosslist)
print("Validation accuracy list:")
print(valacclist)


#* Save the model
torch.save(best_model.state_dict(), 'best_model.pth')
print("Model saved as best_model.pth")

visualize_model(model_ft)

plt.ioff()
plt.show()




