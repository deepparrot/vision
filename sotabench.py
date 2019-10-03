import PIL
import torch
import torchvision
import tqdm

from torchbench.image_classification import ImageNet
import torchvision.transforms as transforms
import PIL

import torchvision.models as models
"""
squeezenet = models.squeezenet1_0(pretrained=True)
densenet = models.densenet161(pretrained=True)
inception = models.inception_v3(pretrained=True)
googlenet = models.googlenet(pretrained=True)
shufflenet = models.shufflenet_v2_x1_0(pretrained=True)
mobilenet = models.mobilenet_v2(pretrained=True)
resnext50_32x4d = models.resnext50_32x4d(pretrained=True)
wide_resnet50_2 = models.wide_resnet50_2(pretrained=True)
mnasnet = models.mnasnet1_0(pretrained=True)
"""
# DEEP RESIDUAL LEARNING

# Define the transforms need to convert ImageNet data to expected model input
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
    std=[0.229, 0.224, 0.225])
input_transform = transforms.Compose([
    transforms.Resize(256, PIL.Image.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])

ImageNet.benchmark(
    model=models.resnet18(pretrained=True),
    paper_model_name='ResNet-18',
    paper_arxiv_id='1512.03385',
    input_transform=input_transform,
    batch_size=256,
    num_gpu=1,
    paper_results={'Top 1 Accuracy': 0.7212}
)

ImageNet.benchmark(
    model=models.resnet34(pretrained=True),
    paper_model_name='ResNet-34 A',
    paper_arxiv_id='1512.03385',
    input_transform=input_transform,
    batch_size=256,
    num_gpu=1,
    paper_results={'Top 1 Accuracy': 0.7497, 'Top 5 Accuracy': 0.9224}
)

ImageNet.benchmark(
    model=models.resnet50(pretrained=True),
    paper_model_name='ResNet-50',
    paper_arxiv_id='1512.03385',
    input_transform=input_transform,
    batch_size=256,
    num_gpu=1
)

ImageNet.benchmark(
    model=models.resnet101(pretrained=True),
    paper_model_name='ResNet-101',
    paper_arxiv_id='1512.03385',
    input_transform=input_transform,
    batch_size=256,
    num_gpu=1
)   

ImageNet.benchmark(
    model=models.resnet152(pretrained=True),
    paper_model_name='ResNet-152',
    paper_arxiv_id='1512.03385',
    input_transform=input_transform,
    batch_size=256,
    num_gpu=1
)   

# ALEXNET

ImageNet.benchmark(
    model=models.alexnet(pretrained=True),
    paper_model_name='AlexNet (single)',
    paper_arxiv_id='1404.5997',
    input_transform=input_transform,
    batch_size=256,
    num_gpu=1,
    paper_results={'Top 1 Accuracy': 0.5714}
)   

# VGG

ImageNet.benchmark(
    model=models.vgg11(pretrained=True),
    paper_model_name='VGG-11',
    paper_arxiv_id='1409.1556',
    input_transform=input_transform,
    batch_size=256,
    num_gpu=1,
    paper_results={'Top 1 Accuracy': 0.704, 'Top 5 Accuracy': 0.896}
)

ImageNet.benchmark(
    model=models.vgg13(pretrained=True),
    paper_model_name='VGG-13',
    paper_arxiv_id='1409.1556',
    input_transform=input_transform,
    batch_size=256,
    num_gpu=1,
    paper_results={'Top 1 Accuracy': 0.713, 'Top 5 Accuracy': 0.901}
)   

ImageNet.benchmark(
    model=models.vgg16(pretrained=True),
    paper_model_name='VGG-16',
    paper_arxiv_id='1409.1556',
    input_transform=input_transform,
    batch_size=256,
    num_gpu=1
)   

ImageNet.benchmark(
    model=models.vgg19(pretrained=True),
    paper_model_name='VGG-19',
    paper_arxiv_id='1409.1556',
    input_transform=input_transform,
    batch_size=256,
    num_gpu=1
)   

ImageNet.benchmark(
    model=models.vgg11_bn(pretrained=True),
    paper_model_name='VGG-11 (batch-norm)',
    paper_arxiv_id='1409.1556',
    input_transform=input_transform,
    batch_size=256,
    num_gpu=1,
    paper_results={'Top 1 Accuracy': 0.704, 'Top 5 Accuracy': 0.896}
)

ImageNet.benchmark(
    model=models.vgg13_bn(pretrained=True),
    paper_model_name='VGG-13 (batch-norm)',
    paper_arxiv_id='1409.1556',
    input_transform=input_transform,
    batch_size=256,
    num_gpu=1,
    paper_results={'Top 1 Accuracy': 0.713, 'Top 5 Accuracy': 0.901}
)   

ImageNet.benchmark(
    model=models.vgg16_bn(pretrained=True),
    paper_model_name='VGG-16 (batch-norm)',
    paper_arxiv_id='1409.1556',
    input_transform=input_transform,
    batch_size=256,
    num_gpu=1
)   

ImageNet.benchmark(
    model=models.vgg19_bn(pretrained=True),
    paper_model_name='VGG-19 (batch-norm)',
    paper_arxiv_id='1409.1556',
    input_transform=input_transform,
    batch_size=256,
    num_gpu=1
)   

