from transformer import Model
import torch
from torch import nn, Tensor
from data_loader import tokenize
import numpy as np

if __name__ == "__main__":
    device = 'cpu'
    model = Model().to(device)
    model.eval()
    max_len = model.max_len
    with torch.no_grad():
        word = 'hangman'
        
        unique_letters = list(set(word))
        num_letters_to_mask = np.random.randint(1, len(unique_letters)) #select the number of letters to be masked
        letters_to_mask = np.random.choice(unique_letters, num_letters_to_mask, replace=False)  #select the letters to be masked
        masked_word = ''.join(['#' if ltr in letters_to_mask else ltr for ltr in word]) #replace the chosen letters with # in the word
        print(f'word = {word}, masked_word = {masked_word}')
        
        tokenized_word = torch.tensor(tokenize(masked_word, max_len))
        
        onehot = np.zeros(26)
        for letter in letters_to_mask:
            onehot[ord(letter) - ord('a')] += 1
        onehot_label = torch.tensor([label if label > 0 else 1e-10 for label in onehot / len(letters_to_mask)]).unsqueeze(0)
        # onehot_label = torch.tensor(onehot / len(letters_to_mask)).unsqueeze(0)
        
        prediction = model(tokenized_word.unsqueeze(0))
        print(f'toeknized_work = {tokenized_word}')
        print(f'onehot_label = {onehot_label}, sum = {onehot_label.sum()}')
        print(f'prediction = {torch.exp(prediction.squeeze())}, sum = {torch.exp(prediction.squeeze()).sum()}')
        loss_func = nn.KLDivLoss(reduction='batchmean')
        loss = loss_func(prediction, onehot_label)
        print(f'loss = {loss}')
    # /orange/physics-dept/an.kandala/coding_projects/Deep_learning_projects/hang_man/words_250000_train.txt