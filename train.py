### @author Ram Saran Vuppuluri.

import argparse
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
import os


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_directory',
                        action='store',
                        default='flowers',
                        dest='data_directory',
                        type=str,
                        help='training data directory')
    parser.add_argument('--arch',
                        action='store',
                        dest='arch',
                        choices=['alexnet', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
                                 'vgg19', 'vgg19_bn', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
                                 'resnet152', 'densenet121', 'densenet169',
                                 'densenet161', 'densenet201', 'inception_v3'],
                        type=str,
                        help='Base pre trained network')
    parser.add_argument('--drop_out',
                        action='store',
                        default=0.0,
                        dest='drop_out',
                        type=float,
                        help='Proportion of drop out in new classifier layers to address overfitting')
    parser.add_argument('--learning_rate',
                        action='store',
                        default=0.001,
                        dest='learning_rate',
                        type=float,
                        help='Learning rate for gradient descent')
    parser.add_argument('--num_labels',
                        action='store',
                        dest='num_labels',
                        default=102,
                        type=int,
                        help='Learning rate for gradient descent')
    parser.add_argument('--hidden_units',
                        action='store',
                        dest='hidden_units',
                        default=512,
                        type=int,
                        help='Number of starting hidden units')
    parser.add_argument('--epochs',
                        action='store',
                        dest='epochs',
                        default=5,
                        type=int,
                        help='Number of iterations')
    parser.add_argument('--gpu',
                        action='store',
                        dest='gpu',
                        choices=['cpu', 'cuda'],
                        type=str,
                        help='Run mode ')
    return parser


def generate_transformers():
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    test_validation_transforms = transforms.Compose([transforms.Resize(255),
                                                     transforms.CenterCrop(224),
                                                     transforms.ToTensor(),
                                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                                          [0.229, 0.224, 0.225])])

    return train_transforms, test_validation_transforms


def generate_datasets(train_dir, valid_dir, test_dir, train_transforms, test_validation_transforms):
    train_datasets = datasets.ImageFolder(train_dir, train_transforms)
    validation_datasets = datasets.ImageFolder(valid_dir, test_validation_transforms)
    test_datasets = datasets.ImageFolder(test_dir, test_validation_transforms)

    return train_datasets, validation_datasets, test_datasets


def generate_dataloaders(train_datasets, validation_datasets, test_datasets):
    train_dataloaders = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True)
    validation_dataloaders = torch.utils.data.DataLoader(validation_datasets, batch_size=64, shuffle=True)
    test_dataloaders = torch.utils.data.DataLoader(test_datasets, batch_size=64, shuffle=True)

    return train_dataloaders, validation_dataloaders, test_dataloaders


def create_model(arch, drop_out, learning_rate, num_labels, hidden_units, device):
    if arch == 'alexnet':
        model = models.alexnet(pretrained=True)
    elif arch == 'vgg11':
        model = models.vgg11(pretrained=True)
    elif arch == 'vgg11_bn':
        model = models.vgg11_bn(pretrained=True)
    elif arch == 'vgg13':
        model = models.vgg13(pretrained=True)
    elif arch == 'vgg13_bn':
        model = models.vgg13_bn(pretrained=True)
    elif arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif arch == 'vgg16_bn':
        model = models.vgg16_bn(pretrained=True)
    elif arch == 'vgg19':
        model = models.vgg19(pretrained=True)
    elif arch == 'vgg19_bn':
        model = models.vgg19_bn(pretrained=True)
    elif arch == 'resnet18':
        model = models.resnet18(pretrained=True)
    elif arch == 'resnet34':
        model = models.resnet34(pretrained=True)
    elif arch == 'resnet50':
        model = models.resnet50(pretrained=True)
    elif arch == 'resnet101':
        model = models.resnet101(pretrained=True)
    elif arch == 'resnet152':
        model = models.resnet152(pretrained=True)
    elif arch == 'squeezenet1_0':
        model = models.squeezenet1_0(pretrained=True)
    elif arch == 'squeezenet1_1':
        model = models.squeezenet1_1(pretrained=True)
    elif arch == 'densenet121':
        model = models.densenet121(pretrained=True)
    elif arch == 'densenet169':
        model = models.densenet169(pretrained=True)
    elif arch == 'densenet161':
        model = models.densenet161(pretrained=True)
    elif arch == 'densenet201':
        model = models.densenet201(pretrained=True)
    elif arch == 'inception_v3':
        model = models.inception_v3(pretrained=True)
    else:
        raise ValueError('Unexpected network architecture', arch)

    for param in model.parameters():
        param.requires_grad = False

    if arch == 'resnet18' or arch == 'resnet34' or arch == 'resnet50' or arch == 'resnet101' or arch == 'resnet152' or arch == 'inception_v3':
        num_filters = model.fc.in_features
        fc = nn.Sequential(nn.Linear(num_filters, 512),
                           nn.ReLU(),
                           nn.Dropout(0.2),
                           nn.Linear(512, 256),
                           nn.ReLU(),
                           nn.Dropout(0.2),
                           nn.Linear(256, 102),
                           nn.LogSoftmax(dim=1))
        model.fc = fc

        optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)
    else:
        if type(model.classifier) is nn.modules.Sequential:
            for classif in model.classifier:
                try:
                    num_filters = classif.in_features
                    break
                except AttributeError:
                    continue
        elif type(model.classifier) is nn.modules.Linear:
            num_filters = model.classifier.in_features

        classifier = nn.Sequential(nn.Linear(num_filters, hidden_units),
                                   nn.ReLU(),
                                   nn.Dropout(0.2),
                                   nn.Linear(hidden_units, 256),
                                   nn.ReLU(),
                                   nn.Dropout(0.2),
                                   nn.Linear(256, 102),
                                   nn.LogSoftmax(dim=1))
        model.classifier = classifier

        optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    criterion = nn.NLLLoss()

    model.to(device)

    return model, optimizer, criterion


def train_model(model, train_dataloaders, validation_dataloaders, optimizer, criterion, epochs):
    train_losses, validtion_losses, validation_accuracies = [], [], []

    for e in range(epochs):
        running_loss = 0

        for inputs, labels in train_dataloaders:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logOutput = model.forward(inputs)

            loss = criterion(logOutput, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()
        else:
            validation_loss = 0
            validation_accuracy = 0
            model.eval()
            with torch.no_grad():
                for valid_inputs, valid_labels in validation_dataloaders:
                    valid_inputs, valid_labels = valid_inputs.to(device), valid_labels.to(device)

                    valid_output = model.forward(valid_inputs)

                    valid_loss = criterion(valid_output, valid_labels)

                    validation_loss += valid_loss.item()

                    ps = torch.exp(valid_output)

                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == valid_labels.view(*top_class.shape)

                    validation_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            train_losses.append(running_loss / len(train_dataloaders))
            validtion_losses.append(validation_loss / len(validation_dataloaders))
            validation_accuracies.append(validation_accuracy / len(validation_dataloaders))
            model.train()

    return model, train_losses, validtion_losses, validation_accuracies


def test_accuracy(model, test_dataloaders):
    test_accuracy = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_dataloaders:
            inputs, labels = inputs.to(device), labels.to(device)

            output = model.forward(inputs)

            ps = torch.exp(output)
            top_p, top_class = ps.topk(1, dim=1)

            equals = top_class == labels.view(*top_class.shape)

            test_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print("Test Accuracy: {:.3f}".format(test_accuracy / len(test_dataloaders)))


def save_model(arch, model, train_datasets, hidden_units):
    model.class_to_idx = train_datasets.class_to_idx

    if arch == 'resnet18' or arch == 'resnet34' or arch == 'resnet50' or arch == 'resnet101' or arch == 'resnet152' or arch == 'inception_v3':
        checkpoint = {'structure': arch,
                      'hidden_layer1': hidden_units,
                      'fc': model.fc,
                      'state_dict': model.state_dict(),
                      'class_to_idx': model.class_to_idx}
    else:
        checkpoint = {'structure': arch,
                      'hidden_layer1': hidden_units,
                      'classifier': model.classifier,
                      'state_dict': model.state_dict(),
                      'class_to_idx': model.class_to_idx}

    if not os.path.exists('/home/workspace/saved_model'):
        os.makedirs('/home/workspace/saved_model')

    torch.save(checkpoint, '/home/workspace/saved_model/flower_species_classifier_' + arch + '.pth')


parms = get_arguments().parse_args()

device = torch.device("cuda" if parms.gpu == 'cuda' and torch.cuda.is_available() else "cpu")

train_dir = parms.data_directory + '/train'
valid_dir = parms.data_directory + '/valid'
test_dir = parms.data_directory + '/test'

train_transforms, test_validation_transforms = generate_transformers()

print('Transformers generated')

train_datasets, validation_datasets, test_datasets = generate_datasets(train_dir, valid_dir, test_dir, train_transforms,
                                                                       test_validation_transforms)

print('Datasets generated')

train_dataloaders, validation_dataloaders, test_dataloaders = generate_dataloaders(train_datasets, validation_datasets,
                                                                                   test_datasets)

print('Data loaders generated')

model, optimizer, criterion = create_model(parms.arch, parms.drop_out, parms.learning_rate, parms.num_labels,
                                           parms.hidden_units, device)

print('Model created')

model, train_losses, validtion_losses, validation_accuracies = train_model(model, train_dataloaders,
                                                                           validation_dataloaders, optimizer, criterion,
                                                                           parms.epochs)

print('Model trained')

print('Train loss per epoch')
print(train_losses)

print('Validation loss per epoch')
print(validtion_losses)

print('Validation accuracy per epoch')
print(validation_accuracies)

test_accuracy(model, test_dataloaders)

save_model(parms.arch, model, train_datasets, parms.hidden_units)
