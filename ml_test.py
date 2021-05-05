import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import sys


def inference():
    assert(len(sys.argv) == 3)
    model_path = sys.arv[1]
    img_path = sys.argv[2]

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    img_t = transform(Image.open(img_path))
    batch_t = torch.unsqueeze(img_t, 0)

    alexnet = models.alexnet()
    alexnet.load_state_dict(torch.load(model_path))
    alexnet.eval()
    out = alexnet(batch_t)

    with open('imagenet_classes.txt') as f:
        classes = [line.strip() for line in f.readlines()]

    _, index = torch.max(out, 1)
    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
    # print('{}: {}'.format(classes[index[0]], percentage[index[0]].item()))
    return '{}: {}'.format(classes[index[0]], percentage[index[0]].item())