import os
import argparse
import random
import numpy as np 
import time

import torch
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
from torchvision import transforms


def get_splits(dataset, train_split=0.8, test_split=0.15):
    '''
    Obtain train/test/validation splits from a given Torch dataset.

    Args:
        dataset (torch.utils.data.Dataset): Source dataset to be split
        train_split (float): Train split ratio
        test_split (float): Test split ratio
    
    Returns:
        train_dataset (torch.utils.data.Subset): Training subset
        test_dataset (torch.utils.data.Subset): Testing subset
        val_dataset (torch.utils.data.Subset): Validation subset of size 
            len(dataset) - len(dataset)*train_split - len(dataset)*test_split
    '''

    n = len(dataset)
    train_size, test_size = int(train_split * n), int(test_split * n)
    val_size = n - train_size - test_size

    train_dataset, test_dataset, val_dataset = data.random_split(dataset, [train_size, test_size, val_size])

    return train_dataset, test_dataset, val_dataset


def filter_data(dataset, batch_size=256):
    '''
    Filter non-snake or lizard images from the input dataset.

    Args:
        dataset (torch.utils.data.Dataset): dataset to apply the filter
        batch_size (int): batch size for evaluating images with resnext101_32x8d

    Returns:
        dataset (torch.utils.data.Dataset): filtered datset
    '''
    model = torchvision.models.resnext101_32x8d(pretrained=True)

    model.eval()
    if torch.cuda.is_available():
        print("using GPU to filter data")
        model.to(torch.device("cuda"))

    full_data = data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    idx = []

    with torch.no_grad():
        for i, (batch_img, _) in enumerate(full_data):
            if i % 10 == 0:
                print(str(batch_size*i) + ' images evaluated.' + ' ' + str(batch_size*i-len(idx)) + ' images filtered.')
            if torch.cuda.is_available():
                batch_img = batch_img.to(torch.device("cuda"))
            batch_out = model(batch_img)
            batch_out = torch.argsort(batch_out, dim=1, descending=True)[:, :5]
            for j in range(batch_out.shape[0]):
                if check_top_five(batch_out[j]):
                    idx.append(i*batch_size+j)

    dataset = torch.utils.data.Subset(dataset, idx)

    return dataset


def check_top_five(label_tensor):
    '''
    Checks if top 5 ImageNet predictions of an input image are snake or lizard-like.

    Args:
        label_tensor (torch.Tensor): the top five predictions of an input image

    Returns:
        True (Boolean): if the top five predictions contain a snake or lizard label
        False (Boolean): otherwise
    '''
    label_list = label_tensor.tolist()
    for i in range(len(label_list)):
        if 37 < label_list[i] < 51 or 51 < label_list[i] < 69:
            return True
    return False


def train(epoch):
    '''
    Train the model for one epoch. This function is taken from HW1.
    '''
    # Some models use slightly different forward passes and train and test
    # time (e.g., any model with Dropout). This puts the model in train mode
    # (as opposed to eval mode) so it knows which one to use.
    
    # train loop
    for batch_idx, batch in enumerate(train_loader):
        model.train()
        # prepare data
        images, targets = Variable(batch[0]), Variable(batch[1])
        if args.cuda:
            images, targets = images.cuda(), targets.cuda()
        
        output = model(images)
        loss = F.cross_entropy(output, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            val_loss, val_acc = evaluate('val', n_batches=4)
            train_loss = loss.data
            examples_this_epoch = batch_idx * len(images)
            epoch_progress = 100. * batch_idx / len(train_loader)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t'
                  'Train Loss: {:.6f}\tVal Loss: {:.6f}\tVal Acc: {}'.format(
                epoch, examples_this_epoch, len(train_loader.dataset),
                epoch_progress, train_loss, val_loss, val_acc))

    scheduler.step()

    return val_acc


def evaluate(split, verbose=False, n_batches=None):
    '''
    Compute loss on val or test data. This function is taken from HW1.
    '''
    model.eval()
    loss = 0
    correct = 0
    n_examples = 0
    if split == 'val':
        loader = val_loader
    elif split == 'test':
        loader = test_loader
    with torch.no_grad():
        for batch_i, batch in enumerate(loader):
            data, target = batch
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            output = model(data)
            loss += criterion(output, target, size_average=False).data
            # predict the argmax of the log-probabilities
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            n_examples += pred.size(0)
            if n_batches and (batch_i >= n_batches):
                break

    loss /= n_examples
    acc = 100. * correct / n_examples

    if verbose:
        print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            split, loss, correct, n_examples, acc))
    return loss, acc


if __name__ == '__main__':

    # Parse the arguments
    parser = argparse.ArgumentParser(description="Snakes")

    parser.add_argument('--path', type=str, default="dataset", help="Dataset folder path")
    parser.add_argument('--batch-size', type=int, default=10, help='Input batch size for training')
    parser.add_argument('--resize', type=bool, default=True, help='Resize width')
    parser.add_argument('--width', type=int, default=224, help='Resize width')
    parser.add_argument('--height', type=int, default=224, help='Resize height')

    # Val split is the remaining part
    parser.add_argument('--train_split', type=float, default=0.8, help='Train split ratio')
    parser.add_argument('--test_split', type=float, default=0.15, help='Test split ratio')

    parser.add_argument('--model', type=str, help="Model name")
    parser.add_argument('--weight-decay', type=float, default=0.0, help='Weight decay hyperparameter')
    parser.add_argument('--lr', type=float, default = 1e-3, metavar='LR', help='learning rate')
    parser.add_argument('--lr-step-size', type=int, default = 50, help='learning rate scheduler step size')
    parser.add_argument('--lr-step-gamma', type=float, default = 0.1, help='learning rate step gamma')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train')

    parser.add_argument('--log-interval', type=int, default=10, metavar='N', 
        help='number of batches between logging train status')

    parser.add_argument('--filter', type=bool, default=False, help='Set to True to enable filtering of data')

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()

    #device = torch.device("cuda" if args.cuda else "cpu")
    #kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    random_seed = 128

    if args.cuda:
        torch.cuda.empty_cache() 
        kwargs = {'num_workers': 1, 'pin_memory': True}
        device = "cuda"
        torch.cuda.manual_seed(random_seed)
    else:
        kwargs = {}
        device = "cpu"

    print('Device: ', device)

    # TODO: Do I need to call it twice here?
    np.random.seed(random_seed)

    # The number of classes is the number of folders in the processed dataset root folder
    n_classes = len(os.listdir(args.path))

    # Pre-calculated dataset mean and average standard deviation, broken images are omitted
    snakes_mean_color = [103.64519509 / 255.0, 118.35241679 / 255.0, 124.96846096/ 255.0] 
    snakes_std_color  = [50.93829403 / 255.0, 52.51745522 / 255.0, 54.89964224/ 255.0] 

    # Transforms
    # Resize() goes before ToTensor()
    # Normalize goes after ToTensor()
    transform_list = []

    if args.resize:
        transform_list.append(transforms.Resize((args.height, args.width)))

    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(snakes_mean_color, snakes_std_color))

    transform = transforms.Compose(transform_list) 

    # Load and split the dataset
    dataset = torchvision.datasets.ImageFolder(root=args.path, transform=transform)
    #print(dataset)

    train_dataset, test_dataset, val_dataset = get_splits(dataset, args.train_split, args.test_split)

    if args.filter:
        train_dataset = filter_data(train_dataset)
        torch.cuda.empty_cache()

    train_loader = data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    val_loader = data.DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

    # Model definition
    model = None
    optimizer = None
    scheduler = None

    #args.model = 'vgg19bn'
    #args.weight_decay = 1e-5

    # Select the model
    # TODO: Add option to switch between transfer learning and fine-tuning
    # TODO: Add option to switch between adam and SGD

    # VGG19 Batch normalized
    if args.model == 'vgg19bn':
        model = torchvision.models.vgg19_bn(pretrained = True)
        model.classifier[-1] = torch.nn.Linear(4096, n_classes)
        
    # VGG16
    elif args.model == 'vgg16':
        model = torchvision.models.vgg16(pretrained = True) 
        model.classifier[-1] = torch.nn.Linear(4096, n_classes)

    # RESNET18
    elif args.model == 'resnet18':
        model = torchvision.models.resnet18(pretrained = True) 
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, n_classes)

    # Squeezenet
    elif args.model == 'squeezenet':
        model = torchvision.models.squeezenet1_1(pretrained=True)
        model.classifier[1] = torch.nn.Conv2d(512, n_classes, kernel_size=(1,1), stride=(1,1))

    else:
        raise Exception('Unknown model {}'.format(args.model))

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_step_gamma)

    # TODO: Add args to change criterion
    criterion = F.cross_entropy

    model.to(device)

    acc_best = 0
    
    start = time.time()
    # train the model one epoch at a time
    for epoch in range(1, args.epochs + 1):
        val_acc = train(epoch)

        if val_acc > acc_best:
            acc_best = val_acc
            print('Saving better model ', val_acc.item())
            torch.save(model, args.model + '_best.pt')

    print ("Elapsed training {:.2f}".format(time.time() - start), 's')

    start = time.time()
    evaluate('test', verbose=True)
    print ("Elapsed evaluation {:.2f}".format(time.time() - start), 's')

    print('Best accuracy: ', acc_best.item())