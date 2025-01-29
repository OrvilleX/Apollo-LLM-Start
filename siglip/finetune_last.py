import pandas as pd
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
from multilabel_dataset import MultiLabelDataset
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm

def collate_fn(batch):
    # 过滤掉 None
    batch = [item for item in batch if item is not None]
    
    if len(batch) == 0:
        return None

    data = torch.stack([item[0] for item in batch])
    target = torch.stack([item[1] for item in batch])
    return data, target

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_id = "/root/autodl-tmp/siglip-so400m-patch14-384"
df = pd.read_csv("/root/autodl-tmp/multilabel_modified/multilabel_classification(2).csv")

labels = list(df.columns)[2:]
id2label = {id: label for id, label in enumerate(labels)}

processor = AutoImageProcessor.from_pretrained(model_id, device=device)
model = AutoModelForImageClassification.from_pretrained(model_id, problem_type="multi_label_classification", id2label=id2label)
model = model.to(device)

size = processor.size["height"]
mean = processor.image_mean
std = processor.image_std

transform = Compose([
    Resize((size, size)),
    ToTensor(),
    Normalize(mean=mean, std=std),
])

train_dataset = MultiLabelDataset(root="/root/autodl-tmp/multilabel_modified/images",
                                  df=df, transform=transform)

train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=2, shuffle=True)
batch = next(iter(train_dataloader))

outputs = model(pixel_values=batch[0].to(device), labels=batch[1].to(device))


optimizer = AdamW(model.parameters(), lr=5e-5)

losses = AverageMeter()

model.train()
for epoch in range(10):
    for idx, batch in enumerate(tqdm(train_dataloader)):
        # 跳过无效批次
        if batch is None:
            continue

        pixel_values, labels = batch

        optimizer.zero_grad()

        outputs = model(
            pixel_values=pixel_values.to(device),
            labels=labels.to(device),
        )

        loss = outputs.loss
        losses.update(loss.item(), pixel_values.size(0))
        loss.backward()

        optimizer.step()

        if idx % 2000 == 0:
            print('Epoch: [{0}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   epoch, loss=losses,))

model.save_pretrained("./saved_model/")
