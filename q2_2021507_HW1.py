import numpy as np
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
import torchvision.transforms.v2 as transforms 
import torch.optim as optim
import wandb

torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.cuda.manual_seed_all(42)
np.random.seed(42)

wandb.login()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = dict(
    epochs=10,
    batch_size=32,
    learning_rate=0.001,
    dataset="WildlifeClassification",
    num_classes=10
)

run = wandb.init(project="wildlife-classification", config=config)

class WildlifeDataset(Dataset):
    def __init__(self, dir, transform=transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])):
        super().__init__()
        self.transform = transform
        self.data = datasets.ImageFolder(dir, transform=self.transform)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        y = self._adjust_class_ids(y)
        return x, y

dataset = WildlifeDataset('/kaggle/input/cv-hw1/Wildlife_dataset/Cropped_final/')

train_ratio, val_ratio, test_ratio = 0.7, 0.1, 0.2
train_size = int(len(dataset) * train_ratio)
val_size = int(len(dataset)  * val_ratio)
test_size = int(len(dataset) - train_size - val_size)

train_data, val_data, test_data = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True)
val_loader = DataLoader(val_data, batch_size=config['batch_size'], shuffle=True)
test_loader = DataLoader(test_data, batch_size=config['batch_size'], shuffle=True)

train_labels = [label for _, label in train_data]
val_labels = [label for _, label in val_data]
test_labels = [label for _, label in test_data]

plt.hist(train_labels, bins=10, alpha=0.5, label='train')
plt.hist(val_labels, bins=10, alpha=0.5, label='validation')
plt.hist(test_labels, bins=10, alpha=0.5, label='test')

plt.legend(loc='upper right')
plt.show()

class CNN(nn.Module):
    def __init__(self,num_classes:int) -> None:
        super(CNN, self).__init__()
        self.conv_layer_1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.max_pool_1 = nn.MaxPool2d(kernel_size=4, stride=4)

        self.conv_layer_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1,padding=1)
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_layer_3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.max_pool_3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(128*16*16, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        out1 = self.conv_layer_1(x)
        out1 = self.relu(out1)
        out1 = self.max_pool_1(out1)

        out2 = self.conv_layer_2(out1)
        out2 = self.relu(out2)
        out2 = self.max_pool_2(out2)

        out3 = self.conv_layer_3(out2)
        out3 = self.relu(out3)
        out3 = self.max_pool_3(out3)

        out4 = out3.view(-1, 128*16*16)
        out4 = self.fc1(out4)

        return out4
    
model = CNN(num_classes=config['num_classes'])
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

for _ in range(config['epochs']):
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    for (images, labels) in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total_train += labels.size(0)
        correct_train += predicted.eq(labels.data).sum().item()
    
    training_accuracy = 100 * correct_train / total_train
    training_loss = running_loss / len(train_loader)

    model.eval()
    running_loss = 0.0
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for (images, labels) in val_loader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            loss = criterion(output, labels)

            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total_val += labels.size(0)
            correct_val += predicted.eq(labels.data).sum().item()

    validation_accuracy = 100 * correct_val / total_val
    validation_loss = running_loss / len(val_loader)

    wandb.log({"Training Loss": training_loss, "Training Accuracy": training_accuracy, "Validation Loss": validation_loss, "Validation Accuracy": validation_accuracy})

from sklearn.metrics import f1_score, confusion_matrix

# Test the model
model.eval()
correct_test = 0
total_test = 0

# log the f1 score
y_true = []
y_pred = []


with torch.no_grad():
    for (images, labels) in test_loader:
        images, labels = images.to(device), labels.to(device)
        output = model(images)
        _, predicted = torch.max(output.data, 1)
        total_test += labels.size(0)
        correct_test += predicted.eq(labels.data).sum().item()
        y_true += labels.cpu().numpy().tolist()
        y_pred += predicted.cpu().numpy().tolist()


test_accuracy = 100 * correct_test / total_test
wandb.log({"Test Accuracy": test_accuracy})


f1 = f1_score(y_true, y_pred, average='weighted')
wandb.log({"f1_score": f1})

# log the confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)
wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(probs=None,
                                y_true=y_true, preds=y_pred,
                                class_names=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])})

from sklearn.metrics import f1_score, confusion_matrix

model_resnet.eval()

correct_test = 0
total_test = 0
y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model_resnet(images)
        _, predicted = torch.max(outputs.data, 1)
        total_test += labels.size(0)
        correct_test += (predicted == labels).sum().item()
        y_true += labels.cpu().numpy().tolist()
        y_pred += predicted.cpu().numpy().tolist()


test_accuracy = 100 * correct_test / total_test
wandb.log({"Test Accuracy Resnet18": test_accuracy})

f1 = f1_score(y_true, y_pred, average='weighted')
wandb.log({"f1_score Resnet18": f1})

conf_matrix = confusion_matrix(y_true, y_pred)
wandb.log({"confusion_matrix Resnet18": wandb.plot.confusion_matrix(probs=None,
                                y_true=y_true, preds=y_pred,
                                class_names=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])})

# Define the data augmentation transformations
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomVerticalFlip(),
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])


augmented_dataset = WildlifeDataset('/kaggle/input/cv-hw1/Wildlife_dataset/Cropped_final/', transform=transform)
normal_dataset = WildlifeDataset('/kaggle/input/cv-hw1/Wildlife_dataset/Cropped_final/')

# sample 20% from the augmented dataset and add it to the normal dataset
augmented_data_size = int(len(augmented_dataset) * 0.1)
augmented_data, _ = torch.utils.data.random_split(augmented_dataset, [augmented_data_size, len(augmented_dataset) - augmented_data_size])

combined_dataset = torch.utils.data.ConcatDataset([normal_dataset, augmented_data])

run = wandb.init(project="wildlife-classification", config=config)

for _ in range(config['epochs']):
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    for (images, labels) in train_loader_augmented:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model_augmented(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total_train += labels.size(0)
        correct_train += predicted.eq(labels.data).sum().item()
    
    training_accuracy = 100 * correct_train / total_train
    training_loss = running_loss / len(train_loader_augmented)

    model_augmented.eval()
    running_loss = 0.0
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for (images, labels) in val_loader_augmented:
            images, labels = images.to(device), labels.to(device)
            output = model_augmented(images)
            loss = criterion(output, labels)

            running_loss += loss.item() 
            _, predicted = torch.max(output.data, 1)
            total_val += labels.size(0)
            correct_val += predicted.eq(labels.data).sum().item()

    validation_accuracy = 100 * correct_val / total_val
    validation_loss = running_loss / len(val_loader_augmented)

    wandb.log({"Training Loss Augmented": training_loss, "Training Accuracy Augmented": training_accuracy, "Validation Loss Augmented": validation_loss, "Validation Accuracy Augmented": validation_accuracy})

    # visualize the features of the model using t-SNE in 3D
tsne_3d = TSNE(n_components=3, random_state=42)

train_tsne_3d = tsne_3d.fit_transform(train_features)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(train_tsne_3d[:, 0], train_tsne_3d[:, 1], train_tsne_3d[:, 2], c=train_labels, cmap='viridis')
plt.title('t-SNE visualization of ResNet18 features in 3D')
plt.show()

train_features = torch.tensor(train_features)
train_labels = torch.tensor(train_labels)

tsne = TSNE(n_components=2, random_state=42)
train_tsne = tsne.fit_transform(train_features)

run = wandb.init(project="wildlife-classification", config=config)

for _ in range(config['epochs']):
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    for (images, labels) in train_loader_augmented:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model_augmented(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total_train += labels.size(0)
        correct_train += predicted.eq(labels.data).sum().item()
    
    training_accuracy = 100 * correct_train / total_train
    training_loss = running_loss / len(train_loader_augmented)

    model_augmented.eval()
    running_loss = 0.0
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for (images, labels) in val_loader_augmented:
            images, labels = images.to(device), labels.to(device)
            output = model_augmented(images)
            loss = criterion(output, labels)

            running_loss += loss.item() 
            _, predicted = torch.max(output.data, 1)
            total_val += labels.size(0)
            correct_val += predicted.eq(labels.data).sum().item()

    validation_accuracy = 100 * correct_val / total_val
    validation_loss = running_loss / len(val_loader_augmented)

    wandb.log({"Training Loss Augmented": training_loss, "Training Accuracy Augmented": training_accuracy, "Validation Loss Augmented": validation_loss, "Validation Accuracy Augmented": validation_accuracy})

plt.scatter(train_tsne[:, 0], train_tsne[:, 1], c=train_labels, cmap='viridis')
plt.title('t-SNE visualization of ResNet18 features')
plt.show()

