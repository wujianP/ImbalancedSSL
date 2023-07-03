from model.wideresnetwithABC import WideResNet
from model.resnetwithABC import ResNet
from model.resnet_for_imagenet127 import ResNet50


def create_model(model_name, num_class, ema=False, pretrained=False):
    if model_name == 'wideresnet':
        model = WideResNet(num_classes=num_class)
    elif model_name == 'resnet':
        model = ResNet(num_classes=num_class, encoder_name=model_name, pretrained=pretrained)
    elif model_name == 'resnet_img127':
        model = ResNet50(num_classes=num_class, rotation=True, classifier_bias=True)
    elif model_name == 'resnet_baseline':
        from model.resnetwithABC_baseline import ResNet as ResNet_baseline
        model = ResNet_baseline(num_classes=num_class, encoder_name='resnet', pretrained=False)
    else:
        raise KeyError

    model = model.cuda()
    params = list(model.parameters())
    
    if ema:
        for param in params:
            param.detach_()

    return model, params
