import torch
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader
import numpy as np
import enchant

def tokenize(word: str , max_len: int) -> Tensor:
    '''converts a word to a tensor of integers'''
    assert len(word) <= max_len, f"word length [{len(word)}] is greater than max_len [{max_len}]"
    # -> empty letter slot
    ltr2idx = {ltr: idx+1 for idx, ltr in enumerate('#abcdefghijklmnopqrstuvwxyz')}
    tokenized_word = [ltr2idx[ltr] for ltr in word]+ [0]*(max_len - len(word))
    return tokenized_word 

class WordDataset(Dataset):
    def __init__(self, word_file, max_len, device):
        #select only valid english words with length less than or equal to max_len and with at least 2 unique letters
        words = [word for word in open(word_file).read().split('\n') if len(word) <= max_len and len(list(set(word)))>1]
        #randomly select 40% of the letters to be masked
        selected_indices = np.random.choice(np.arange(len(words)), len(words), replace=False)
        self.words = [words[i] for i in selected_indices]
        self.max_len = max_len
        self.device = device
    
    def __len__(self):
        return len(self.words)
    
    def __getitem__(self, idx):
        word = self.words[idx]
        unique_letters = list(set(word))
        #select the number of letters to be masked
        # num_letters_to_mask = np.random.randint(1, len(unique_letters))
        num_letters_to_mask = 1
        #select the letters to be masked
        letters_to_mask = np.random.choice(unique_letters, num_letters_to_mask, replace=False)
        #replace the chosen letters with # in the word
        masked_word = ''.join(['#' if ltr in letters_to_mask else ltr for ltr in word])
        tokenized_word = torch.tensor(tokenize(masked_word, self.max_len)).to(self.device)
        onehot = np.zeros(26)
        for letter in letters_to_mask:
            onehot[ord(letter) - ord('a')] += word.count(letter)
        onehot_label = torch.tensor([label if label > 0 else 1e-10 for label in onehot / len(letters_to_mask)]).to(self.device)
        return tokenized_word, onehot_label