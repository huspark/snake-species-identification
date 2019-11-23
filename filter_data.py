import numpy as np
import torch
import torchvision
from torchvision import transforms
import os
from PIL import Image
import regex as re
import torch.utils.data as data
from torch.autograd import Variable

data_path = 'dataset'
test_img_filename = 'agkistrodon-contortrix/0a0c57ca18.jpg'

ImageNetLabels = {}
with open("ImageNetLabels.txt") as f:
    for line in f:
        (k, v) = line.split(': ')
        k = re.sub('[^0-9]+', '', k)
        v = re.sub('[^A-Za-z ]+', '', v)
        ImageNetLabels[int(k)] = v


def get_frequent_image_classes(data_path):
    freq = np.zeros(1000)

    model = torchvision.models.resnext101_32x8d(pretrained=True)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )])

    if torch.cuda.is_available():
        print("using GPU")
        model.cuda()
    model.eval()
    # count = 0
    dataset = torchvision.datasets.ImageFolder(root=data_path, transform=transform)

    full_data = data.DataLoader(dataset, batch_size=1, shuffle=False)

    with torch.no_grad():
        for i, (img, label) in enumerate(full_data):
            if i > 10:
                break
            # if i % 100 == 0:
            #     print(str(i) + ' images evaluated.')
            #img = torch.unsqueeze(img, 0)
            out = model(img)
            out = torch.argmax(out, 1)
            freq[out] += 1


    print(freq)
    most_freq_idx = np.argsort(freq)[::-1]

    with open('freq_labels.txt', 'w') as f:
        for idx in range(len(most_freq_idx)):
            label_idx = most_freq_idx[idx]
            if freq[most_freq_idx[idx]] != 0:
                f.write(ImageNetLabels.get(label_idx) + ' ' + str(int(freq[label_idx])) + '\n')
                print(ImageNetLabels.get(label_idx) + ' ' + str(int(freq[label_idx])))


if __name__ == '__main__':
    model = torchvision.models.resnext101_32x8d(pretrained=True)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )])

    img = Image.open(os.path.join(data_path, test_img_filename))
    img = transform(img)
    img = torch.unsqueeze(img, 0)
    model.eval()
    out = model(img)
    out = torch.Tensor.flatten(out)
    _, classes = torch.sort(out, descending=True)
    print(classes[:5])
    get_frequent_image_classes(data_path=data_path)
