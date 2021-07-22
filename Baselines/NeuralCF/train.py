import argparse
import pandas as pd
import numpy as np
import time
import torch
import pickle
from NeuralCF.dataprocessing import train_sample, validate_test_sample
from NeuralCF.model import NeurCF_runner

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser('My MSDCR_model')

parser.add_argument('--epoch', type=int, default=5, help='number of epoches')
parser.add_argument('--batch_size', type=int, default=10, help='batch_size')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--latent_item_dim', type=int, default=32, help='latent embedding dimension')
parser.add_argument('--latent_dim', type=int, default=32, help='latent embedding dimension')
parser.add_argument('--attn_dropout', type=int, default=0.1, help='attention dropout rate')

args=parser.parse_args()

MATB_config = {
              'epoch': args.epoch,
              'batch_size': args.batch_size,
              'lr': args.lr,
              'latent_item_dim':args.latent_item_dim,
              'latent_dim': args.latent_dim,
              'attn_dropout': args.attn_dropout,
}

def SampleTrain():
    print("The model is in training！")
    config = MATB_config
    input_mov_col = 15
    input_bok_col = 4
    input_dim_user = 2111
    input_mov_item = 21948
    input_bok_item = 14298
    NeurCF = NeurCF_runner(input_mov_col, input_dim_user, input_mov_item, config)

    with open('../data/mov_bok/train(12).pkl', 'rb') as f:
        train = pickle.load(f)
        # train = train_sample(path='../data_graph/item/train.pkl')
        min_loss = 100000
        i = 0
        print('Start training：')
        for epoch in range(config['epoch']):
            print('Epoch {}  starts !!!'.format(epoch + 1))
            loss_epoch_mean = NeurCF.train_single_epoch(train, config, epoch_id=epoch)
            if loss_epoch_mean < min_loss:
                l = np.abs(loss_epoch_mean - min_loss)
                min_loss = loss_epoch_mean
                print("save model")
                torch.save(NeurCF.NeurCF.state_dict(), '../parameters/' + 'MTBA_best' + '.pth')
                if l < 2:
                    i += 1
                    if i == 5:
                        break
        f.close()

    NeurCF.NeurCF.load_state_dict(torch.load('../parameters/' + 'MTBA_best' + '.pth'))
    print('Start test !!!')
    with open('../data/mov_bok/test.pkl', 'rb') as f:
        validate = pickle.load(f)
        NeurCF.evaluate(data=validate)
        f.close()


SampleTrain()
