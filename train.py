import torch

import pandas as pd

from imagen_pytorch.download import load_checkpoint
from imagen_pytorch.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)
from imagen_pytorch.resample import create_named_schedule_sampler

from imagen_pytorch import dist_util, logger
from dataset import get_loader
from train_utils import TrainLoop

import json

has_cuda = th.cuda.is_available()
device = th.device('cpu' if not has_cuda else 'cuda')


imagen_save_path = "/storage/imagen/imagen.pt"

coco_image_path = "/datasets/coco/train2014/"
coco_annotation_path = "/datasets/coco/annotations/captions_train2014.json"

options = model_and_diffusion_defaults()
options['use_fp16'] = False
options['t5_name'] = 't5-large'
model, diffusion = create_model_and_diffusion(**options)

model.eval()
#if has_cuda:
#    model.convert_to_fp16()
model.to(device)
#model.load_state_dict(load_checkpoint('base', device), strict=False)
# model.load_state_dict(th.load('/content/Imagen-pytorch/imagen-pytorch.pt'))



if os.path.exists(imagen_save_path):
    model.load_state_dict(torch.load(imagen_save_path))

print('total base parameters', sum(x.numel() for x in model.parameters()))


def get_images_id(images_list):
    images_dict = {}
    for i in images_list:
        images_dict[i['id']] = i['file_name']
    return images_dict



def create_dataset():
    with open(coco_annotation_path) as json_file:
        data = json.load(json_file)
        
    images_dict = get_images_id(data['images'])
    
    df = []
    for annotation in data['annotations']:
        df.append([images_dict[annotation['image_id']], annotation['caption']])
        
    
    df = pd.DataFrame(df)
    df.columns = ['path', 'text']
    
    data = get_loader(batch_size=4,
                  resolution=64,
                   image_dir=coco_image_path,
                   df=df,
                   tokenizer_name='t5-large', 
                   max_len=128,
                   zero_text_prob=0.1,
                   shuffle=True)
    
    return data


def configure_logger():
    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    

def train(data):
    
    schedule_sampler = create_named_schedule_sampler('uniform', diffusion)
    TrainLoop(
            model=model,
            diffusion=diffusion,
            data=data,
            batch_size=4,
            microbatch=-1,
            lr=1e-4,
            ema_rate="0.9999",
            log_interval=100,
            save_interval=10000,
            resume_checkpoint=False,
            use_fp16=False,
            fp16_scale_growth=1e-3,
            schedule_sampler=schedule_sampler,
            weight_decay=0.01,
            lr_anneal_steps=0,
            save_dir=imagen_save_path,
    ).run_loop()


    
if __name__ == '__main__':
    configure_logger()
    data = create_dataset()
    train(data)
    
    
