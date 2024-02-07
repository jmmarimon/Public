import csv
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

# All possible tokens in our vocabulary (total 39).
TOKEN_LIST = ['<pad>', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',\
               'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',\
               'Y', 'Z', '"', '-', "'", '.', ',', '(', ')', '_', '+', ' ',\
               '<sos>', '<eos>']

# Create dictionaries that map index to token, and vice versa
idx_to_token = {i:c for i, c in enumerate(TOKEN_LIST)}
token_to_idx = {c:i for i, c in enumerate(TOKEN_LIST)}

def load_data(data_path, labels_path=None):
    """Load in data (and labels, if given)

    Args:
        data_path (str): Path to the *_data.npy file
        labels_path (str, optional): Path to the *_labels.npy file. Defaults to None.
                                     If not given, will only return data list.

    Returns:
        list(np.array), list(str): Data and list of labels (if given). 
    """
    data = [d for d in np.load(data_path, allow_pickle=True)]
    if labels_path is not None:
        labels = np.load(labels_path, allow_pickle=True)
        labels = [" ".join([word.decode('utf-8').upper() for word in utterance]) for utterance in labels]
        return data, labels
    else:
        return data

def convert_str_to_idxs(utterance):
    """Converts utterance str to list of indices, and adds <sos> and <eos> indices.
    
    Args:
        utterance (str): The utterance as a string, without <sos> or <eos> pre-added.
    
    Returns:
        list: a list of integers, corresponding to the index of each letter in the utterance
    """
    idxs = [token_to_idx['<sos>']] + [token_to_idx[t] for t in utterance] + [token_to_idx['<eos>']]
    return idxs

def convert_idxs_to_str(idxs, remove_special_tokens=True):
    """Converts list of indices to str.
    
    Args:
        idxs (list): List of valid indices of tokens to convert.
        remove_special_tokens (bool, optional): If True, removes the first <sos> and last <eos> tokens,
                                                and everything after the last <eos> token (generally padding).
                                                Note that if <sos>, <eos>, or <pad> are present between the first
                                                <sos> and last <eos>, they'll stay in. Default True.
    Returns:
        str: The indices mapped to a string.
    """
    if remove_special_tokens:
        if token_to_idx["<eos>"] in idxs:
            idxs = idxs[:idxs.index(token_to_idx["<eos>"])]
        if token_to_idx["<sos>"] in idxs:
            idxs = idxs[1:]
    return "".join([idx_to_token[i] for i in idxs])

def plot_attention(attention, save_image=True, display_plot=False):
    """Displays attention plot, optionally saves it

    Args:
        attention (torch.FloatTensor): (max_len, seq_len)
        save_image (boolean, optional): If True, save the attention plot to attention_plots/[timestamp].png
                                        Default is True.
        display_plot (boolean, optional): If True, displays the plot in the Jupyter notebook cell output.
                                          If you're running this in a training loop, you may want to set as False and instead
                                          manually open the image file if you want to see it, to reduce cluttering the cell output.
                                          Default is False.
    """
    if attention.is_cuda:
        attention = attention.cpu().detach()
    plt.clf() # Clear previous plot (if any)
    sns.heatmap(attention, cmap='GnBu', cbar_kws={'label': 'Attention Score'}) # Plot attentions as a heatmap
    plt.xlabel("seq_len") # Label axes
    plt.ylabel("max_len")
    
    if save_image:
        os.makedirs("attention_plots", exist_ok=True)
        plt.savefig(fname=f"attention_plots/{datetime.now().strftime('%Y_%m_%d-%I_%M_%S_%p')}", facecolor='w', bbox_inches='tight')
    
    if display_plot:
        plt.show()

def export_predictions_to_csv(predictions):
    """Export list of predictions to formatted `.csv` file.

    Args:
        predictions (list): List of string predictions to export
    """
    os.makedirs("submissions", exist_ok=True)
    file_name = f"submissions/submission_{datetime.now().strftime('%Y_%m_%d-%I_%M_%S_%p')}.csv"
    with open(file_name, 'w', newline = '') as f:
        writer = csv.writer(f)
        writer.writerow(['id','label'])
        # for i, l in enumerate(predictions):
        writer.writerows(enumerate(predictions))

    print(f"Wrote predictions to {file_name}")
    
# --------------------
# Test methods
# --------------------

@torch.no_grad()
def init_pblstm_for_testing(pblstm):
    rng = torch.Generator()
    rng.manual_seed(0)

    key = lambda x: x[0]
    parameters = sorted(list(pblstm.named_parameters()), key = key)
    assert len(parameters) == 8, "Expected LSTM to have 8 sets of parameters"
    for name, param in parameters:
        param[...] = torch.randint(5, size = param.shape, generator = rng)

@torch.no_grad()
def init_encoder_for_testing(pblstm):
    rng = torch.Generator()
    rng.manual_seed(0)

    key = lambda x: x[0]
    parameters = sorted(list(pblstm.named_parameters()), key = key)
    assert len(parameters) == 36, "Expected Encoder to have 36 sets of parameters (8 for each of 4 lstms, 2 for each linear layer)"
    for name, param in parameters:
        param[...] = torch.randint(5, size = param.shape, generator = rng)

@torch.no_grad()
def init_decoder_for_testing(pblstm):
    rng = torch.Generator()
    rng.manual_seed(0)

    key = lambda x: x[0]
    parameters = sorted(list(pblstm.named_parameters()), key = key)
    assert len(parameters) == 11, "Expected Decoder to have 11 sets of parameters (4 for each of 2 lstmcells, 2 for linear, 1 for embedding)"
    for name, param in parameters:
        param[...] = torch.randint(5, size = param.shape, generator = rng)

@torch.no_grad()
def init_seq2seq_for_testing(pblstm):
    rng = torch.Generator()
    rng.manual_seed(0)

    key = lambda x: x[0]
    parameters = sorted(list(pblstm.named_parameters()), key = key)
    assert len(parameters) == 47, "Expected Decoder to have 47 sets of parameters (36 for encoder, 11 for decoder)"
    for name, param in parameters:
        param[...] = torch.randint(5, size = param.shape, generator = rng)
