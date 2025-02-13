from datasets import transforms_shir as transforms
# from utils import *

def get_transform(args):
    Idim = int(args['Idim'])
    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        # transforms.Resize((Idim, Idim)),
        transforms.ColorJitter(brightness=0.4,
                               contrast=0.4,
                               saturation=0.4,
                               hue=0.1),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(int(args['rotate']), scale=(float(args['scale1']), float(args['scale2']))),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[142.07, 98.48, 132.96], std=[65.78, 57.05, 57.78])
    ])
    transform_test = transforms.Compose([
        transforms.ToPILImage(),
        # transforms.Resize((Idim, Idim)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[142.07, 98.48, 132.96], std=[65.78, 57.05, 57.78])
    ])
    return transform_train, transform_test

