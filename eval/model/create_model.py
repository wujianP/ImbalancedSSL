from resnetwithABC import ResNet


def create_model(model_name, num_class, ema=False, pretrained=False):
    model = ResNet(num_classes=num_class, encoder_name='resnet', pretrained=False)
    model = model.cuda()
    params = list(model.parameters())
    
    if ema:
        for param in params:
            param.detach_()

    return model, params
