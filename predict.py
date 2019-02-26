### @author Ram Saran Vuppuluri.

# run command python3 predict.py --path_to_image ../saved_model/flower_species_classifier_densenet121.pth --top_k 5 --gpu cuda

import argparse
import torch
from torchvision import transforms, models
from PIL import Image
import json
import random, os
from random import randint


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_directory',
                        action='store',
                        default='flowers',
                        dest='data_directory',
                        type=str,
                        help='training data directory')
    parser.add_argument('--path_to_image',
                        action='store',
                        default='/home/workspace/saved_model/',
                        dest='path_to_image',
                        type=str,
                        help='Path to saved model')
    parser.add_argument('--top_k',
                        action='store',
                        default=1,
                        dest='top_k',
                        type=int,
                        help='Number of highest values')
    parser.add_argument('--category_names',
                        action='store',
                        default='cat_to_name.json',
                        dest='category_names',
                        type=str,
                        help='Category name map')
    parser.add_argument('--gpu',
                        action='store',
                        dest='gpu',
                        choices=['cpu', 'cuda'],
                        type=str,
                        help='Run mode ')
    return parser


def load_model(arch, device):
    filepath = '/home/workspace/saved_model/flower_species_classifier_' + arch + '.pth'

    checkpoint = torch.load(filepath)

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

    model.structure = checkpoint['structure']
    model.hidden_layer1 = checkpoint['hidden_layer1']
    model.class_to_idx = checkpoint['class_to_idx']

    if arch == 'resnet18' or arch == 'resnet34' or arch == 'resnet50' or arch == 'resnet101' or arch == 'resnet152' or arch == 'inception_v3':
        model.fc = checkpoint['fc']
    else:
        model.classifier = checkpoint['classifier']

    model.load_state_dict(checkpoint['state_dict'])

    if device == 'cuda':
        return model.cuda()
    else:
        return model.cpu()


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)

    model = getattr(models, checkpoint['structure'])(pretrained=True)

    model.structure = checkpoint['structure']
    model.hidden_layer1 = checkpoint['hidden_layer1']

    if 'classifier' in checkpoint:
        model.classifier = checkpoint['classifier']
    elif 'fc' in checkpoint:
        model.fc = checkpoint['fc']

    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])

    if torch.cuda.is_available():
        return model.cuda()
    else:
        return model.cpu()


def process_image(image):
    img_pil = Image.open(image)

    test_validation_transforms = transforms.Compose([transforms.Resize(255),
                                                     transforms.CenterCrop(224),
                                                     transforms.ToTensor(),
                                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                                          [0.229, 0.224, 0.225])])

    img_tensor = test_validation_transforms(img_pil)

    return img_tensor


def predict(image_path, model, topk, device):
    model.to(device)
    model.eval()

    inputs = process_image(image_path)
    inputs = inputs.unsqueeze_(0)
    inputs = inputs.float()
    inputs = inputs.to(device)

    output = model.forward(inputs)

    ps = torch.exp(output)

    return ps.topk(topk, dim=1)


def sanity_check(test_dir, model, topk, device):
    ri = randint(1, 102)

    image = test_dir + "/" + str(ri) + "/" + random.choice(os.listdir(test_dir + "/" + str(ri) + "/"))

    probs, classes = predict(image, model, topk, device)

    probs = probs.to("cpu").data.numpy()

    classes = classes.to("cpu").data.numpy()

    top_classes = [index_to_class[each] for each in classes[0]]

    print("Flower Species: " + cat_to_name.get(str(ri)))

    print("Predicted flower species: ")

    for i in range(len(probs[0])):
        print("Probability: " + str(probs[0][i]) + " Species: " + cat_to_name.get(str(top_classes[i])))


parms = get_arguments().parse_args()

test_dir = parms.data_directory + '/test'

filepath = parms.path_to_image

cat_to_name = parms.category_names

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

device = torch.device("cuda" if parms.gpu == 'cuda' and torch.cuda.is_available() else "cpu")

# model = load_model(parms.arch, device)

model = load_checkpoint(filepath)

index_to_class = {val: key for key, val in model.class_to_idx.items()}

sanity_check(test_dir, model, parms.top_k, device)
