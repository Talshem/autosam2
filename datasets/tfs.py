from datasets import transforms_shir as transforms
# from utils import *


def get_transform(args):
    if args == "polypgen":
        return get_polyp_transform()

def get_polyp_transform():
    transform_train = transforms.Compose([
        # transforms.Resize((352, 352)),
        transforms.ToPILImage(),
        transforms.ColorJitter(brightness=0.4,
                               contrast=0.4,
                               saturation=0.4,
                               hue=0.1),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(90, scale=(0.75, 1.25)),
        transforms.ToTensor(),
        # transforms.Normalize([105.61, 63.69, 45.67],
        #                      [83.08, 55.86, 42.59])
    ])
    transform_test = transforms.Compose([
        # transforms.Resize((352, 352)),
        transforms.ToPILImage(),
        transforms.ToTensor(),
        # transforms.Normalize([105.61, 63.69, 45.67],
        #                      [83.08, 55.86, 42.59])
    ])
    return transform_train, transform_test
