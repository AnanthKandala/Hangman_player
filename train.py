import torch
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader
import numpy as np
import enchant
from data_loader import tokenize, WordDataset
from scheduler import *
from transformer import Model
import time 
from datetime import datetime
from tqdm import tqdm
import os

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    training_file = '/orange/physics-dept/an.kandala/coding_projects/Deep_learning_projects/hang_man/words_250000_train.txt'
    training_dataset = WordDataset(training_file, 16, device)

    torch.set_float32_matmul_precision('high')
    model = Model().to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    train_dataloader = DataLoader(training_dataset, batch_size=2048)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)#, betas = (0.9, 0.98), eps = 1.0e-9)
    scheduler = Scheduler(optimizer, model.embedding_dim, 20)
    # scheduler = None
    loss_func = nn.KLDivLoss(reduction='batchmean')
    def train(num_epochs, scheduler=None):
        model.train()
        now = datetime.now(); dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        log_file = 'training_loss.txt'
        with open(log_file, 'w') as f:
            f.write(f'{dt_string}\n')
            f.write(f'dataset length = {training_dataset.__len__()}, num_model_params = {num_params}, num_epochs = {num_epochs} \n')
            f.write(f'num_batches = {len(train_dataloader)}\n')
            f.write('Training Loss\n')
        os.system(f'cat {log_file}')
        training_loss = np.empty(num_epochs)
        for epoch in range(num_epochs):
            start = time.time()
            epoch_loss = torch.empty(len(train_dataloader))
            for batch_idx, (inputs, labels) in tqdm(enumerate(train_dataloader)):
                optimizer.zero_grad()
                predictions = model(inputs)
                loss = loss_func(predictions, labels)
                if torch.any(torch.isnan(loss)):
                    print(f'epoch {epoch}, batch {batch_idx}, loss = {loss}')
                    print(f'inputs = {inputs}')
                    print(f'labels = {labels}, sum = {labels.sum(dim=1)}')
                    print(f'predictions = {torch.exp(predictions)}, sum = {torch.exp(predictions).sum(dim=1)}')
                    print(f'loss = {loss}')
                    raise ValueError('loss is nan')
                # print(batch_idx, loss)
                loss.backward()
                optimizer.step()
                epoch_loss[batch_idx] = loss.detach().item()
                # print(f'epoch {epoch}, batch {batch_idx}, loss = {loss}')
                if scheduler is not None:
                    scheduler.step()
            training_loss[epoch] = epoch_loss.mean()
            end = time.time()
            string = f'epoch {epoch} {training_loss[epoch]} [{end-start} s]\n'
            with open(log_file, 'a') as f:
                f.write(string)
            print(string)
            if epoch%20 == 0:
                    torch.save(model, f'model_{epoch}.pth')
    # print(scheduler)
    train(100, scheduler)
    torch.save(model, 'model.pth')
    