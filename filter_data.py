import torch
import torchvision
from torchvision import transforms
import os
from PIL import Image

data_path = 'dataset'
test_img_filename = 'agkistrodon-contortrix/0a0c57ca18.jpg'

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
