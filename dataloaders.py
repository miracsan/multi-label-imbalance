import torch
from torchvision import datasets, transforms

from customDatasets import NIHDataset
from customTransforms import RescaleBB
from utils import get_configs_from_dataset

def construct_dataloaders(dataset, batch_size, num_pool_ops=3, output_filenames=False):

    if dataset == "nih":
        transforms_train = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ])
        transforms_test = transforms_train
        
        transform_bbox = RescaleBB(224, 1024)
        
        path_to_folder = "/content/data"
        
        train_set = NIHDataset(path_to_folder, fold="train", transform=transforms_train)
        val_set = NIHDataset(path_to_folder, fold="val", transform=transforms_train)
        test_set = NIHDataset(path_to_folder, fold="test", transform=transforms_test)
        bbox_set = NIHDataset(path_to_folder, fold="bbox", transform=transforms_test, transform_bbox=transform_bbox)


    dataloaders = {}
    dataloaders['train'] = torch.utils.data.DataLoader(train_set, batch_size, shuffle=False, num_workers=2)
    dataloaders['val'] = torch.utils.data.DataLoader(val_set, batch_size, shuffle=False, num_workers=2)
    dataloaders['test'] = torch.utils.data.DataLoader(test_set, batch_size, shuffle=False, num_workers=2)
    dataloaders['bbox'] = torch.utils.data.DataLoader(bbox_set, 1, shuffle=False, num_workers=1)
    
    dataloaders['train_set'] = train_set
    dataloaders['val_set'] = val_set
    dataloaders['test_set'] = test_set
    dataloaders['bbox_set'] = bbox_set
    
    dataloaders['bbox'].Dataset = bbox_set
    
    dataloaders["batch_size"] = batch_size
        
    return dataloaders
    


def get_dataset_mean_and_std(dataset, num_channels=3):
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False)
    mean = torch.zeros(num_channels)
    std = torch.zeros(num_channels)
    for inputs, labels in dataloader:
        for i in range(num_channels):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std
    


    