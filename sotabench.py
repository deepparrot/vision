import PIL
from sotabencheval.semantic_segmentation import PASCALVOCEvaluator
import torch
import torchvision
from torchvision.models.segmentation import fcn_resnet101
import torchvision.transforms as transforms
import tqdm

from sotabench_transforms import Normalize, Compose, Resize, ToTensor

MODEL_NAME = 'fcn_resnet101'

def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., : img.shape[-2], : img.shape[-1]].copy_(img)
    return batched_imgs

def collate_fn(batch):
    images, targets = list(zip(*batch))
    batched_imgs = cat_list(images, fill_value=0)
    batched_targets = cat_list(targets, fill_value=255)
    return batched_imgs, batched_targets

device = torch.device('cuda')

normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
my_transforms = Compose([Resize((520, 480)), ToTensor(), normalize])

dataset_test = torchvision.datasets.VOCSegmentation(root='./.data/vision/voc2012', year='2012', image_set="val", 
                                                    transforms=my_transforms, download=True)
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
        if evaluator.cache_exists:
            break
        
evaluator.save()

### OBJECT DETECTION

from sotabencheval.object_detection import COCOEvaluator
from torch.utils.data import DataLoader
import torchbench
from torchbench.utils import send_model_to_device
from torchbench.object_detection.transforms import Compose, ConvertCocoPolysToMask, ToTensor
import os

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

model, device = send_model_to_device(
    model, device='cuda', num_gpu=1
)
model.eval()

model_output_transform = coco_output_transform
send_data_to_device = coco_data_to_device

test_dataset = torchbench.datasets.CocoDetection(
    root=os.path.join('./.data/vision/coco', "val%s" % '2017'),
    annFile=os.path.join(
        './.data/vision/coco', "annotations/instances_val%s.json" % '2017'
    ),
    transform=None,
    target_transform=None,
    transforms=transforms,
    download=True,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    collate_fn=coco_collate_fn,
)
test_loader.no_classes = 91  # Number of classes for COCO Detection

iterator = tqdm.tqdm(test_loader, desc="Evaluation", mininterval=5)

evaluator = COCOEvaluator(
    root='./.data',
    paper_model_name='ResNeXt-101-32x8d',
    paper_arxiv_id='1611.05431')

def prepare_for_coco_detection(predictions):
    coco_results = []
    for original_id, prediction in predictions.items():
        if len(prediction) == 0:
            continue

        boxes = prediction["boxes"]
        boxes = convert_to_xywh(boxes).tolist()
        scores = prediction["scores"].tolist()
        labels = prediction["labels"].tolist()

        coco_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": labels[k],
                    "bbox": box,
                    "score": scores[k],
                }
                for k, box in enumerate(boxes)
            ]
        )
    return coco_results

def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)

with torch.no_grad():
    for i, (input, target) in enumerate(iterator):
        input, target = send_data_to_device(input, target, device=device)
        original_output = model(input)
        output, target = model_output_transform(original_output, target)
        result = {
            tar["image_id"].item(): out for tar, out in zip(target, output)
        }
        result = prepare_for_coco_detection(result)

        evaluator.add(result)
        if evaluator.cache_exists:
            break
        
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
