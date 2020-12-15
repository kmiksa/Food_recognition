import numpy as np
import pandas as np
import os

import fastai
from fastai.vision.all import *

path= 'train_set/'
def get_x(r): return path + r['img_name']
def get_y(r): return str(r['label'])

class Worker(object):
    def __init__(self, csv_path='train_labels.csv',  batch_size=64, image_size=128, unfreeze = False):
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.unfreeze = unfreeze
        self.df = pd.read_csv(self.csv_path)
        

    def get_dls(self, batch_size, image_size):
        dblock = DataBlock(blocks=(ImageBlock, CategoryBlock),
                       get_x=get_x, 
                       get_y=get_y,
                       item_tfms=Resize(460),
                       splitter=RandomSplitter(0.05),
                       batch_tfms=[*aug_transforms(size=self.image_size, min_scale=0.75),
                                   Normalize.from_stats(*imagenet_stats)])
        return dblock.dataloaders(self.df, bs=self.batch_size)


if __name__ == "__main__":
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train image recognition model.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'default' or 'input'")
    parser.add_argument('--run', required=True,
                        help='run number') 
    parser.add_argument('--csv_path', required=False,
                        help='path to labels') 
    parser.add_argument('--batch_size', required=False,
                        help='batch size')
    parser.add_argument('--image_size', required=False,
                        help='input image size ')
    parser.add_argument('--unfreeze', required=False,
                        help='Train whole model')
    args = parser.parse_args()
    
    if args.command == 'default':
        worker = Worker()
    else:
        worker = Worker(args.csv_path, args.batch_size, args.image_size, args.unfreeze)
        
    
  
    data_block = worker.get_dls(worker.batch_size, worker.image_size)

    model = xresnet101(n_out=data_block.c)
    learner = Learner(data_block, model, loss_func=LabelSmoothingCrossEntropy(), 
                metrics=accuracy, cbs=MixUp())

    #learner = learner.load('best_run5_medium')


    #Adding callback to Save best model. 
    #TODO: find if there is updated callback for Tensorboard. 
    #fine_tune - all layers
    #fit_one_cycle - one layer if not unfreezed
    
 
        
    if worker.unfreeze:
        learner.unfreeze()
        
    learner.fit_one_cycle(100 ,1e-4, cbs = SaveModelCallback(monitor='accuracy', 
                                                     comp=np.greater, 
                                                     fname='best_run' + str(run) + '_small'))


    #export and save after 100 epochs 
    try:
        learn.save('epoch_small_100_run'+str(args.run) +'.pkl') 
    except Exception as e:
        print('Coulndt save model')
        print(e)


    try:
        learn.export('epoch_small_100_run'+str(args.run) +'.pkl')
    except Exception as e:
        print('Coulndt export model')
        print(e)
 
    
    data_block = worker.get_dls(64, 256)

    #train on whole 
    learner.unfreeze()
    learner.fit_one_cycle(100, lr_max=slice(1e-5,1e-3), cbs = SaveModelCallback(monitor='accuracy', 
                                                     comp=np.greater, 
                                                    fname='best_run' + str(run) + '_medium'))

    #save and export after 100 epochs as well 
    try:
        learn.save('epoch_medium_100_run'+str(args.run) +'.pkl')
    except Exception as e:
        print('Coulndt save model')
        print(e)


    try:
        learn.export('epoch_medium_100_run'+str(args.run) +'.pkl')
    except Exception as e:
        print('Coulndt export model')
        print(e)
                     
