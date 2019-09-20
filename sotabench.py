import PIL
from sotabencheval.semantic_segmentation import PASCALVOCEvaluator
import torch
import torchvision
from torchvision.models.segmentation import fcn_resnet101
import torchvision.transforms as transforms
import tqdm

from sotabench_transforms import Compose, Resize, ToTensor

MODEL_NAME = 'fcn_resnet101'

def collate_fn(batch):
    images, targets = list(zip(*batch))
    batched_imgs = cat_list(images, fill_value=0)
    batched_targets = cat_list(targets, fill_value=255)
    return batched_imgs, batched_targets

device = torch.device('cuda')

normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
my_transforms = Compose([Resize((520, 480)), ToTensor(), normalize])

dataset_test = torchvision.datasets.VOCSegmentation(root='./data', year='2012', image_set="val", 
                                                    transforms=my_transforms)
test_sampler = torch.utils.data.SequentialSampler(dataset_test)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=32,
    sampler=test_sampler, num_workers=4,
    collate_fn=collate_fn)

model = torchvision.models.segmentation.__dict__['fcn_resnet101'](num_classes=21, pretrained=True)
model.to(device)
model.eval()

evaluator = PASCALVOCEvaluator(root='./data', dataset_year='2012', split='val', paper_model_name='FCN (ResNet-101)',
                              paper_arxiv_id='1605.06211')

with torch.no_grad():
    for image, target in tqdm.tqdm(data_loader_test):
        image, target = image.to('cuda'), target.to('cuda')
        output = model(image)
        output = output['out']
        evaluator.add(output.argmax(1).flatten().cpu().numpy(), target.flatten().cpu().numpy())

evaluator.save()


"""
from torchbench.object_detection import COCO
from torchbench.utils import send_model_to_device
from torchbench.object_detection.transforms import Compose, ConvertCocoPolysToMask, ToTensor
import torchvision
import PIL

def coco_data_to_device(input, target, device: str = "cuda", non_blocking: bool = True):
    input = list(inp.to(device=device, non_blocking=non_blocking) for inp in input)
    target = [{k: v.to(device=device, non_blocking=non_blocking) for k, v in t.items()} for t in target]
    return input, target

def coco_collate_fn(batch):
    return tuple(zip(*batch))

def coco_output_transform(output, target):
    output = [{k: v.to("cpu") for k, v in t.items()} for t in output]
    return output, target

transforms = Compose([ConvertCocoPolysToMask(), ToTensor()])
   
model = torchvision.models.detection.__dict__['maskrcnn_resnet50_fpn'](num_classes=91, pretrained=True)

# Run the benchmark
COCO.benchmark(
    model=model,
    paper_model_name='Mask R-CNN (ResNet-50-FPN)',
    paper_arxiv_id='1703.06870',
    transforms=transforms,
    model_output_transform=coco_output_transform,
    send_data_to_device=coco_data_to_device,
    collate_fn=coco_collate_fn,
    batch_size=8,
    num_gpu=1
)
"""
