import os, sys
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

import pickle

from sampler import ImbalancedDatasetSampler


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
                print(str(batch_size * i) + ' images evaluated.' + ' ' + str(
                    batch_size * i - len(idx)) + ' images filtered.')
            if torch.cuda.is_available():
                batch_img = batch_img.to(torch.device("cuda"))
            batch_out = model(batch_img)
            batch_out = torch.argsort(batch_out, dim=1, descending=True)[:, :5]
            for j in range(batch_out.shape[0]):
                if check_top_five(batch_out[j]):
                    idx.append(i * batch_size + j)

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


def get_weights(dataset):
    '''
    Obtain class-wise and element-wise weights for sampling.
    Source: https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703

    Works for Subset class

    TODO: Optimize (taken as is and slightly adapted)

    Args:
        dataset (torch.util.data.Subset): Source dataset

    Returns:
        w_classes (sequence): weights per class
        w_images (sequence): weights per image

    '''
    n_classes = 85
    n_images = len(dataset)

    w_classes = [0.] * n_classes
    w_images = [0.] * n_images
    weight_per_class = [0.] * n_classes

    # Count number of images per class
    for idx, val in enumerate(dataset):
        if idx % 1000 == 0:
            print(str(idx) + '/' + str(n_images) + ' processed for w_classes')
        w_classes[val[1]] += 1.0

    for i in range(n_classes):
        weight_per_class[i] = float(n_images) / float(w_classes[i])

    for idx, val in enumerate(dataset):
        if idx % 1000 == 0:
            print(str(idx) + '/' + str(n_images) + ' processed for w_images')
        w_images[idx] = weight_per_class[val[1]]

    return w_classes, w_images


def get_weights_fast(dataset):
    '''
    Obtain class-wise and element-wise weights for sampling.
    Source: https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703

    TODO: Optimize (taken as is and slightly adapted)

    Args:
        dataset (torchvision.datasets.ImageFolder): Source dataset

    Returns:
        w_classes (sequence): weights per class
        w_images (sequence): weights per image
    '''

    images = dataset.imgs
    n_classes = len(dataset.classes)
    n_images = len(images)

    w_classes = [0.] * n_classes
    w_images = [0.] * n_images
    weight_per_class = [0.] * n_classes

    # Count number of images per class
    for item in images:
        w_classes[item[1]] += 1.0

    for i in range(n_classes):
        weight_per_class[i] = float(n_images) / float(w_classes[i])

    for idx, val in enumerate(images):
        w_images[idx] = weight_per_class[val[1]]

    return w_classes, w_images


def get_lr(optimizer):
    '''
    Get current learning rate

    Args:
        optimizer:

    Returns:
        lr (): current learning rate of the optimizer
    '''
    for param_group in optimizer.param_groups:
        return param_group['lr']


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

        if args.imbalanced:
            for i in range(len(targets)):
                # To ensure the probability of images[i] being augmented is equal to normalized_w_classes[targets[i]],
                # (1-p) ** num_transforms must be equal to 1 - normalized_w_classes[targets[i]]
                # Here, p is the probability of each transformation in trasnform_list_imbalanced

                # Therefore, we set 1 - p = (1-normalized_w_classes[targets[i]]) ** (1/num_transforms)
                # which is equivalent to p = 1 - (1-normalized_w_classes[targets[i]]) ** (1/num_transforms)
                # So, we set p (the probability of applying a random transformation to images[i] to the specified value

                p = transform_prob[targets[i]]
                transform_list_imbalanced = []
                transform_list_imbalanced.append(transforms.ToPILImage())
                transform_list_imbalanced.append(transforms.RandomHorizontalFlip(p=p))
                transform_list_imbalanced.append(transforms.RandomApply(
                    [transforms.RandomResizedCrop(size=(args.height, args.width), scale=(0.6, 1.0))], p=p))
                transform_list_imbalanced.append(transforms.ToTensor())
                transform_imbalanced = transforms.Compose(transform_list_imbalanced)
                images[i] = transform_imbalanced(images[i])

        if args.cuda:
            images, targets = images.cuda(), targets.cuda()

        output = model(images)

        # Use weighted cross-entropy when imbalaced flag is true
        # loss = F.cross_entropy(output, targets)
        if args.imbalanced:
            loss = criterion(output, targets, weight=w_classes_train)
        else:
            loss = criterion(output, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            val_loss, val_acc = evaluate('val', n_batches=4)
            train_loss = loss.data
            examples_this_epoch = batch_idx * len(images)
            epoch_progress = 100. * batch_idx / len(train_loader)

            # Need this when weighted cross-entropy is used
            train_loss = train_loss / args.batch_size

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t'
                  'Train Loss: {:.6f}\tVal Loss: {:.6f}\tVal Acc: {}'.format(
                epoch, examples_this_epoch, len(train_loader.dataset),
                epoch_progress, train_loss, val_loss, val_acc))

    scheduler.step()

    return val_acc, loss


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
    for batch_i, batch in enumerate(loader):
        data, target = batch
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)

        if args.imbalanced:
            # loss += criterion(output, target, weight=w_classes).data
            loss += criterion(output, target, weight=w_classes_train).data
        else:
            loss += criterion(output, target).data

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
    parser.add_argument('--test_split', type=float, default=0.1, help='Test split ratio')

    parser.add_argument('--model', type=str, help="Model name")
    parser.add_argument('--weight-decay', type=float, default=0.0, help='Weight decay hyperparameter')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR', help='learning rate')
    parser.add_argument('--lr-step-size', type=int, default=50, help='learning rate scheduler step size')
    parser.add_argument('--lr-step-gamma', type=int, default=0.1, help='learning rate step gamma')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train')

    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='number of batches between logging train status')

    parser.add_argument('--filter', type=bool, default=False, help='Set to True to enable filtering of data')

    # New: arguments to continue training
    parser.add_argument('--resume', type=str, default=None, help='Filename of the model to continue training')

    # New: optimizer selection
    parser.add_argument('--optimizer', choices=['sgd', 'adam'], default='adam', help='Optimizer selection')
    parser.add_argument('--momentum', type=float, default=0.99, help='SGD momentum')

    # New: dealing with the imbalanced dataset
    parser.add_argument('--imbalanced', action='store_true', default=False, help='Handle imbalanced dataset')

    # New: multi-gpu
    parser.add_argument('--devices', nargs='+', type=int, help='CUDA Devices', default=None)

    # New: num_workers for loader
    parser.add_argument('--num-workers', type=int, default=1, help='num_workers for DataLoader')

    args = parser.parse_args()

    args.cuda = torch.cuda.is_available()

    # Workaround
    if args.devices is not None:
        print("CUDA Devices: ", args.devices)

    random_seed = 128

    if args.cuda:
        torch.cuda.empty_cache()
        kwargs = {'num_workers': args.num_workers, 'pin_memory': True}
        device = "cuda"
        torch.cuda.manual_seed(random_seed)

    else:
        kwargs = {}
        device = "cpu"

    print('Device: ', device)

    # TODO: Do I need to call it twice here?
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # The number of classes is the number of folders in the processed dataset root folder
    n_classes = len(os.listdir(args.path))

    # Pre-calculated dataset mean and average standard deviation, broken images are omitted
    snakes_mean_color = [103.64519509 / 255.0, 118.35241679 / 255.0, 124.96846096 / 255.0]
    snakes_std_color = [50.93829403 / 255.0, 52.51745522 / 255.0, 54.89964224 / 255.0]

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

    # TODO: Finish, optimize
    if args.imbalanced:
        pass

    w_classes, w_images = get_weights_fast(dataset)
    w_classes = torch.FloatTensor(w_classes)
    w_images = torch.DoubleTensor(w_images)

    if args.cuda:
        w_classes = w_classes.to(device)
        w_images = w_images.to(device)

    # New: sampler
    # TODO: Finish, fix sampler for the subset
    train_dataset, test_dataset, val_dataset = get_splits(dataset, args.train_split, args.test_split)

    weights_precalculated = True

    if not weights_precalculated:
        print("Calculating weights...")
        start = time.time()
        w_classes_train, w_images_train = get_weights(train_dataset)
        end = time.time()
        print("Elapsed train weights: ", (end - start) / 60.0)
        w_classes_test, w_images_test = get_weights(test_dataset)
        end = time.time()
        print("Elapsed test weights: ", (end - start) / 60.0)
        w_classes_val, w_images_val = get_weights(val_dataset)
        end = time.time()
        print("Elapsed val weights: ", (end - start) / 60.0)

        # Save pre-calculated weights
        with open('weights_train.pkl', 'wb') as f:
            pickle.dump([w_classes_train, w_images_train], f)
        with open('weights_test.pkl', 'wb') as f:
            pickle.dump([w_classes_test, w_images_test], f)
        with open('weights_val.pkl', 'wb') as f:
            pickle.dump([w_classes_val, w_images_val], f)

        print("Finished calculating weights...")
    else:
        with open('weights_train.pkl', 'rb') as f:
            w_classes_train, w_images_train = pickle.load(f)
        with open('weights_test.pkl', 'rb') as f:
            w_classes_test, w_images_test = pickle.load(f)
        with open('weights_val.pkl', 'rb') as f:
            w_classes_val, w_images_val = pickle.load(f)
        pass

    # TODO: Convert to numerical
    w_classes_train = torch.FloatTensor(w_classes_train)
    w_images_train = torch.DoubleTensor(w_images_train)
    w_classes_train_normalized = w_classes_train / w_classes_train.sum()

    if args.cuda:
        w_classes_train = w_classes_train.to(device)
        w_images_train = w_images_train.to(device)
        w_classes_train_normalized = w_classes_train_normalized.to(device)

    # input("Press Enter to continue...")

    if args.filter:
        train_dataset = filter_data(train_dataset)
        torch.cuda.empty_cache()

    train_sampler = torch.utils.data.sampler.WeightedRandomSampler(w_images_train, len(w_images_train))
    # print("Test new sampler start")
    # train_sampler = ImbalancedDatasetSampler(train_dataset)
    # print("Test new sampler end")

    if args.imbalanced:
        count = np.zeros(85)
        for label in dataset.targets:
            count[label] += 1
        # print(count)
        count = [val/max(count) for val in count]
        # print(count)
        count = [1-val for val in count]
        # print(count)
        num_transforms = 2
        # (1 - p) ** 2 = 1 - count
        # 1 - p = (1 - normalized) ** (1/2)
        transform_prob = [1 - (1 - val) ** (1 / num_transforms) for val in count]
        transform_prob = torch.FloatTensor(transform_prob)
        transform_prob = transform_prob.to(device)
        # print(transform_prob)

        # Temporary, doesn't work
        train_loader = data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, sampler=train_sampler,
                                       **kwargs)
    else:
        train_loader = data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

    # Don't use imbalanced features for train and validation loaders
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    val_loader = data.DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

    # Model definition
    model = None
    optimizer = None
    scheduler = None

    # Select the model
    # TODO: Add option to switch between transfer learning and fine-tuning

    # VGG19 Batch normalized
    if args.model == 'vgg19bn':
        model = torchvision.models.vgg19_bn(pretrained=True)
        model.classifier[-1] = torch.nn.Linear(4096, n_classes)

    # VGG16
    elif args.model == 'vgg16':
        model = torchvision.models.vgg16(pretrained=True)
        model.classifier[-1] = torch.nn.Linear(4096, n_classes)

    # RESNET18
    elif args.model == 'resnet18':
        model = torchvision.models.resnet18(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, n_classes)

    # Squeezenet
    elif args.model == 'squeezenet':
        model = torchvision.models.squeezenet1_1(pretrained=True)
        model.classifier[1] = torch.nn.Conv2d(512, n_classes, kernel_size=(1, 1), stride=(1, 1))
    else:
        raise Exception('Unknown model {}'.format(args.model))

    # New: Multi-GPU
    if args.devices is not None:
        # model = torch.nn.DataParallel(model, device_ids=args.devices)
        try:
            model = torch.nn.DataParallel(model, device_ids=args.devices)
        except:
            e = sys.exc_info()[0]
            print(e)

    acc_best = 0
    epoch_start = 1

    # Done: Add option to switch between adam and SGD
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                              weight_decay=args.weight_decay)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_step_gamma)

    # TODO: Make an if-else in order not to initialize everything if resume flag is specified
    # Resume training if specified
    if args.resume is not None:
        print('Loading model ... ')

        if os.path.isfile(args.resume):
            # Need to load on CPU first, otherwise CUDA out of memory error
            checkpoint = torch.load(args.resume, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)

            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            epoch_start = checkpoint['epoch'] + 1
            loss = checkpoint['loss']
            acc_best = checkpoint['acc_best']

            train_dataset = checkpoint['train_dataset']
            test_dataset = checkpoint['test_dataset']
            val_dataset = checkpoint['val_dataset']

            train_loader = checkpoint['train_loader']
            test_loader = checkpoint['test_loader']
            val_loader = checkpoint['val_loader']

            print('Starting epoch: ', checkpoint['epoch'], ' Loss: {:.8f}'.format(checkpoint['loss']),
                  'Best accuracy: {:.8f}'.format(checkpoint['acc_best']))
            # DEBUG:

            opt_dict = optimizer.state_dict()
            sch_dict = scheduler.state_dict()
            print('Debug: ')
            print('Epoch:', epoch_start, ' LR: {:.8f}'.format(get_lr(optimizer)), 'Sch Epoch', sch_dict['last_epoch'])
            print('Scheduler epoch:', scheduler.last_epoch)
        else:
            raise Exception('Checkpoint not found {}'.format(args.resume))

    # TODO: Add args to change criterion
    # Debug
    print('Imbalanced', args.imbalanced)
    criterion = F.cross_entropy

    model.to(device)

    start = time.time()
    # Train the model one epoch at a time
    for epoch in range(epoch_start, epoch_start + args.epochs):

        opt_dict = optimizer.state_dict()
        sch_dict = scheduler.state_dict()

        print('Epoch:', epoch, ' LR:', get_lr(optimizer), 'Sch Epoch', sch_dict['last_epoch'])

        val_acc, loss = train(epoch)

        # Saves the best model dictionary with "_best.pth"
        if val_acc > acc_best:
            acc_best = val_acc
            print('Saving better model ', val_acc.item())

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
                'acc_best': acc_best,

                # Not sure if need to save the datasets
                'train_dataset': train_dataset,
                'test_dataset': test_dataset,
                'val_dataset': val_dataset,

                'train_loader': train_loader,
                'test_loader': test_loader,
                'val_loader': val_loader,

                'loss': loss}, args.model + '_best.pth')

        # Saves the current model dictionary with "_last.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_acc': val_acc,
            'acc_best': acc_best,

            # Not sure if need to save the datasets
            'train_dataset': train_dataset,
            'test_dataset': test_dataset,
            'val_dataset': val_dataset,

            'train_loader': train_loader,
            'test_loader': test_loader,
            'val_loader': val_loader,

            'loss': loss}, args.model + '_last.pth')

    print("Elapsed training {:.2f}".format((time.time() - start) / 3600.0), 'hours')

    start = time.time()
    evaluate('test', verbose=True)
    print("Elapsed evaluation {:.2f}".format((time.time() - start)), 'seconds')

    print('Best accuracy: ', acc_best.item())