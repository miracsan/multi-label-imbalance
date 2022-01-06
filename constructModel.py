import torch
import torch.nn as nn
from torchvision import models
from REGISTRY import double_class_methods

def construct_model(architecture, method, num_classes, in_channels=1):
    
    if architecture == "densenet121":
        model = models.densenet121(pretrained=True)
        num_ftrs = model.classifier.in_features
        if method in double_class_methods:
            model.classifier = nn.Linear(num_ftrs, 2 * num_classes)
        else:
            model.classifier = nn.Linear(num_ftrs, num_classes)
            
    elif architecture == "shortdensenet121":

        class ShortNet(nn.Module):
            def __init__(self, base_model):
              super(ShortNet, self).__init__()
              self.base_model = base_model
              self.global_pool = nn.AvgPool2d(28)
              self.classifier = nn.Linear(512, 14)
        
            def forward(self, x):
              x = self.base_model.features.conv0(x)
              x = self.base_model.features.norm0(x)
              x = self.base_model.features.relu0(x)
              x = self.base_model.features.pool0(x);
              x = self.base_model.features.denseblock1(x);
              x = self.base_model.features.transition1(x);
              x = self.base_model.features.denseblock2(x);
              x = self.base_model.features.transition2.norm(x);
              x = self.global_pool(x)
              x = x.squeeze(-1).squeeze(-1)
              x = self.classifier(x)
              return x
              
        model = models.densenet121(pretrained=True)
        model = ShortNet(model)
        num_ftrs = model.classifier.in_features
        if method in double_class_methods:
            model.classifier = nn.Linear(num_ftrs, 2 * num_classes)
        else:
            model.classifier = nn.Linear(num_ftrs, num_classes)
            
    elif architecture == "resnet50":
        model = models.resnet50(pretrained=True)
        num_ftrs = model.classifier.in_features
        if method in double_class_methods:
            model.classifier = nn.Linear(num_ftrs, 2 * num_classes)
        else:
            model.classifier = nn.Linear(num_ftrs, num_classes)
        

    else:
        print("Unknown architecture. Aborting...")
        return
    
    model.architecture = architecture
    model.num_classes = num_classes
    model.method = method
    model.in_channels = in_channels
    
    return model
    

def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path);
    architecture = checkpoint['architecture'];
    num_classes = checkpoint['num_classes'];
    train_method = checkpoint['train_method'];
    model_state_dict = checkpoint['model_state_dict'];
    optimizer_state_dict = checkpoint['optimizer_state_dict'];
    starting_epoch = checkpoint['epoch'] if 'epoch'in checkpoint else 0;
    best_acc = checkpoint['best_acc'] if 'best_acc' in checkpoint else 0.0;
    in_channels = checkpoint['in_channels'] if 'in_channels' in checkpoint else 1;
    
    model = construct_model(architecture, train_method, num_classes, in_channels);
    model.load_state_dict(model_state_dict);
    print("Existing model was trained using {0}".format(train_method))
    
    return model, starting_epoch, best_acc, train_method
    

def put_model_to_device(model):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model


def get_num_pool_ops(architecture):
    if architecture == 'vnet':
        num_pool_ops = 4
    else:
        num_pool_ops = 3
    return num_pool_ops