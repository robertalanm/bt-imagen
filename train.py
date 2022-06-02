import torch

import pandas as pd
import os

from bt_imagen.resample import create_named_schedule_sampler

from bt_imagen import dist_util, logger
from bt_imagen.dataset import get_loader
from bt_imagen.utils import TrainLoop

from imagen_pytorch import Unet, Imagen, ImagenTrainer

import json

has_cuda = torch.cuda.is_available()
device = torch.device('cpu' if not has_cuda else 'cuda')


imagen_save_path = "/storage/imagen/imagen.pt"

coco_annotation_path = "/storage/coco/annotations/captions_train2014.json"
coco_image_path = "/storage/coco/train2014"


def create_model():
    unet1 = Unet(
    dim = 32,
    cond_dim = 512,
    dim_mults = (1, 2, 4, 8),
    num_resnet_blocks = 3,
    layer_attns = (False, True, True, True),
    )

    unet2 = Unet(
        dim = 32,
        cond_dim = 512,
        dim_mults = (1, 2, 4, 8),
        num_resnet_blocks = (2, 4, 8, 8),
        layer_attns = (False, False, False, True),
        layer_cross_attns = (False, False, False, True)
    )

    # imagen, which contains the unets above (base unet and super resoluting ones)

    imagen = Imagen(
        unets = (unet1, unet2),
        text_encoder_name = 't5-large',
        image_sizes = (64, 256),
        beta_schedules = ('cosine', 'linear'),
        timesteps = 1000,
        cond_drop_prob = 0.5
    ).cuda()

    # wrap imagen with the trainer class

    trainer = ImagenTrainer(imagen)

    if os.path.exists(imagen_save_path):
        trainer.load(imagen_save_path)
        logger.log("loaded imagen from {}".format(imagen_save_path))

    return trainer


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
    

def train(trainer, data):
    
    for batch, cond in data:

        print(batch)
        print(cond)


    
if __name__ == '__main__':
    configure_logger()
    trainer = create_model()
    data = create_dataset()
    train(data)
    
    
