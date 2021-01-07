import numpy as np
import torch
import os
from modules.trainer import Trainer
from utils.helper_funcs import read_data


def main():
    args = dict()
    args['dataset'] = '2_digits_per_client'
    args['model'] = 'logistic'
    args['wd'] = 0.001
    args['verbose'] = True
    args['num_iters'] = 1000
    args['local_iters'] = 100
    args['num_round'] = args['num_iters'] // args['local_iters']
    args['clients_per_round'] = 100 
    args['bs'] = 64
    args['lr'] = 0.1
    args['seed'] = 0
    args['input_shape'] = 784
    args['num_class'] = 10
    args['quantize'] = True
    args['quan_level'] = 10
    args['gpu'] = True
    args['gpu'] = args['gpu'] and torch.cuda.is_available()
    
    
    args['secure'] = True
    args['secure_epsilon'] = 1.0
    args['secure_delta'] = 10e-4
    
    args['clipping'] = True
    args['secure_clip'] = 1.0
    
    args['subsampling'] = True
    args['subsampling_gamma'] = .5

    
    if args['secure']:
        args['clipping'] = True

    # Set random seed
    np.random.seed((1 + args['seed']))
    torch.manual_seed(12 + args['seed'])
    if args['gpu']:
        torch.cuda.manual_seed_all(123 + args['seed'])

    train_path = os.path.join('./data/mnist/data/train', args['dataset'])
    test_path = os.path.join('./data/mnist/data/test', args['dataset'])

    dataset = read_data(train_path, test_path)

    trainer = Trainer(args, dataset)
    trainer.train()


if __name__ == '__main__':
    main()
