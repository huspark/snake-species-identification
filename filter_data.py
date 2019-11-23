import numpy as np
import torch
import torchvision
from torchvision import transforms
import os
from PIL import Image
import regex as re
import torch.utils.data as tdata
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

print(ImageNetLabels)

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

    #     for batch_i, batch in enumerate(loader):
    #         data, target = batch
    #         if args.cuda:
    #             data, target = data.cuda(), target.cuda()
    #         data, target = Variable(data, volatile=True), Variable(target)
    #         output = model(data)
    #         loss += criterion(output, target, size_average=False).data
    #         # predict the argmax of the log-probabilities
    #         pred = output.data.max(1, keepdim=True)[1]
    #         correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    #         n_examples += pred.size(0)
    #         if n_batches and (batch_i >= n_batches):
    #             break
    #
    #     loss /= n_examples
    #     acc = 100. * correct / n_examples
    #
    #     if verbose:
    #         print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #             split, loss, correct, n_examples, acc))
    #     return loss, acc
    #
    # for batch_i, batch in enumerate(tdata.DataLoader(data_path, batch_size=10, shuffle=True)):
    #     data, target = batch
    #     if torch.cuda.is_available():
    #         data, target = data.cuda(), target.cuda()
    #     data, target = Variable(data, volatile=True), Variable(target)
    #     out = model(data)
    #     print(out.shape)
    #     break
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    count = 0
    with torch.no_grad():
        for species in os.listdir(data_path):
            if species == '.DS_Store':
                break
            for file in os.listdir(os.path.join(data_path, species)):
                if count > 100:
                    break
                filename = os.path.join(data_path, species, file)
                img = Image.open(filename)
                try:
                    img = transform(img)
                    img = torch.unsqueeze(img, 0)
                    model.eval()
                    out = model(img)
                    out = torch.Tensor.flatten(out)
                    _, classes = torch.sort(out, descending=True)
                    pred = classes[0]
                    freq[pred] += 1
                    count += 1
                    print(count)
                except RuntimeError:
                    pass

    print(freq)
    most_freq_idx = np.argsort(freq)[::-1]

    for idx in range(len(most_freq_idx)):
        if freq[idx] != 0:
            print(ImageNetLabels.get(idx))
    # idx = most_freq_idx[0]
    # while freq[idx] != 0:
    #     print(ImageNetLabels.get(idx))





# def evaluate(split, verbose=False, n_batches=None):
#     '''
#     Compute loss on val or test data. This function is taken from HW1.
#     '''
#     model.eval()
#     loss = 0
#     correct = 0
#     n_examples = 0
#     if split == 'val':
#         loader = val_loader
#     elif split == 'test':
#         loader = test_loader
#     for batch_i, batch in enumerate(loader):
#         data, target = batch
#         if args.cuda:
#             data, target = data.cuda(), target.cuda()
#         data, target = Variable(data, volatile=True), Variable(target)
#         output = model(data)
#         loss += criterion(output, target, size_average=False).data
#         # predict the argmax of the log-probabilities
#         pred = output.data.max(1, keepdim=True)[1]
#         correct += pred.eq(target.data.view_as(pred)).cpu().sum()
#         n_examples += pred.size(0)
#         if n_batches and (batch_i >= n_batches):
#             break
#
#     loss /= n_examples
#     acc = 100. * correct / n_examples
#
#     if verbose:
#         print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#             split, loss, correct, n_examples, acc))
#     return loss, acc

if __name__=='__main__':
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
    #print(out.max())
    #print(out.argmax())
    get_frequent_image_classes(data_path=data_path)
