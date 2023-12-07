import os
import torch
from options import test_options
from dataloader import data_loader
from model import create_model
from util import visualizer
from itertools import islice

if __name__=='__main__':
    
    opt = test_options.TestOptions().parse()
    
    dataset = data_loader.dataloader(opt)
    dataset_size = len(dataset) * opt.batchSize
    print('testing images = %d' % dataset_size)
    
    model = create_model(opt)
    model.eval()
    
    visualizer = visualizer.Visualizer(opt)
    for i, data in enumerate(dataset):
        model.set_input(data)
        model.test()