from torch.nn.functional import one_hot
from tqdm import tqdm
import torch.nn.functional as F
import torch

def label_mapping_base(logits, mapping_sequence):
    '''
    :param logits: output of the pretrained model
    :param mapping_sequence: one-to-one label mapping matrix
    :return: predicted results on the target label space
    '''
    modified_logits = logits[:, mapping_sequence]
    return modified_logits

def label_mapping_calculation(logits, mapping_matrix):
    '''
    :param logits: output of the pretrained model
    :param mapping_sequence: real reweight matrix $\omega$
    :return: predicted results on the target label space
    '''
    modified_logits = torch.mm(logits, mapping_matrix)
    return modified_logits

def get_freq_distribution(fx, y):
    '''
    :param fx: the logits output of the pretrained model
    :param y: the groud truth labels
    :return: frequency distribution matrix of [source predicted labels, target ground truth labels]
    '''
    fx = one_hot(torch.argmax(fx, dim=-1), num_classes=fx.size(-1))
    freq_matrix = [fx[y==i].sum(0).unsqueeze(1) for i in range(len(y.unique()))]
    freq_matrix = torch.cat(freq_matrix, dim=1)
    return freq_matrix

def greedy_mapping(freq_matrix):
    '''
    :param freq_matrix: frequency distribution matrix of [source predicted labels, target ground truth labels]
    :return: greedy one-to-one mapping results
    '''
    mapping_matrix = torch.zeros_like(freq_matrix, dtype=int)
    freq_matrix_flat = freq_matrix.flatten()
    for _ in range(freq_matrix.size(1)):
        loc = freq_matrix_flat.argmax().item()
        loc = [loc // freq_matrix.size(1), loc % freq_matrix.size(1)]
        mapping_matrix[loc[0], loc[1]] = 1
        freq_matrix[loc[0]] = -1
        if mapping_matrix[:, loc[1]].sum() == 1:
            freq_matrix[:, loc[1]] = -1
    return mapping_matrix

def one2one_mappnig_matrix(visual_prompt, network, data_loader):
    '''
    The optimal one-to-one label mapping (for FLM, ILM)
    :param visual_prompt: Current input VR
    :param network: Pretrained model
    :param data_loader: Dataloader for downstream tasks
    :return: optimal one-to-one label mapping
    '''
    device = next(visual_prompt.parameters()).device
    if hasattr(network, "eval"):
        network.eval()
    fx0s = []
    ys = []
    pbar = tqdm(data_loader, total=len(data_loader), desc=f"FLM", ncols=100) if len(data_loader) > 20 else data_loader
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            fx0 = network(visual_prompt(x))
        fx0s.append(fx0)
        ys.append(y)
    fx0s = torch.cat(fx0s).cpu().float()
    ys = torch.cat(ys).cpu().int()
    freq_matrix = get_freq_distribution(fx0s, ys)
    pairs = torch.nonzero(greedy_mapping(freq_matrix))
    mapping_sequence = pairs[:, 0][torch.sort(pairs[:, 1]).indices.tolist()]
    return mapping_sequence

def blm_reweight_matrix(visual_prompt, network, data_loader, lap=1):
    '''
    The optimal real reweight mapping matrix (Bayesian-Guided Label Mapping)
    :param visual_prompt: Current input VR
    :param network: Pretrained model
    :param data_loader: Dataloader for downstream tasks
    :param lap: laplace smooth factor - $\lambda$
    :return: optimal real reweight mapping matrix
    '''
    device = next(visual_prompt.parameters()).device
    if hasattr(network, "eval"):
        network.eval()
    fx0s = []
    ys = []
    pbar = tqdm(data_loader, total=len(data_loader), desc=f"BLM", ncols=100) if len(data_loader) > 20 else data_loader
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            fx0 = network(visual_prompt(x))
        fx0s.append(fx0)
        ys.append(y)
    fx0s = torch.cat(fx0s).cpu().float()
    ys = torch.cat(ys).cpu().int()
    freq_matrix = get_freq_distribution(fx0s, ys)

    # Begin calculating marginal distribution
    classes_sum = torch.sum(freq_matrix, dim=1, keepdim=True)

    # Laplace Smoothing
    classes_sum = classes_sum + lap * torch.ones_like(classes_sum)
    matrix = torch.div(freq_matrix, classes_sum)

    # Normalization
    target_sum = torch.sum(matrix, dim=0)
    norm_matrix = torch.div(matrix, target_sum)
    return norm_matrix.to(device)


def blmpp_reweight_matrix(visual_prompt, network, data_loader, lap=0, k=3):
    '''
    The optimal real reweight mapping matrix (Improved Bayesian-Guided Label Mapping)
    :param visual_prompt: Current input VR
    :param network: Pretrained model
    :param data_loader: Dataloader for downstream tasks
    :param lap: laplace smooth factor - $\lambda$
    :param k: Top k factor
    :return: optimal real reweight mapping matrix
    '''
    device = next(visual_prompt.parameters()).device
    if hasattr(network, "eval"):
        network.eval()
    probs_list = []
    ys = []
    pbar = tqdm(data_loader, total=len(data_loader), desc=f"BLM+", ncols=100) if len(data_loader) > 20 else data_loader
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            fx0 = network(visual_prompt(x))
            probabilities = F.softmax(fx0, dim=1)

            # Top-k Mechanism
            top_values, top_indices = torch.topk(probabilities, k, dim=1)

            # Probability Summation Module
            result = torch.zeros_like(probabilities)
            result.scatter_(1, top_indices, top_values)
        probs_list.append(result.cpu().float())
        ys.append(y)
    probs = torch.cat(probs_list, dim=0)
    ys = torch.cat(ys).cpu().int()

    matrix = torch.zeros((len(ys.unique()), probs.size(-1)))
    indices = ys.view(-1, 1).long().expand(-1, matrix.size(-1))
    matrix.scatter_add_(0, indices, probs)
    matrix = matrix.t()

    # Begin calculating marginal distribution
    classes_sum = torch.sum(matrix, dim=1, keepdim=True)

    # Laplace Smoothing
    classes_sum = classes_sum + lap * torch.ones_like(classes_sum)
    matrix = torch.div(matrix, classes_sum)

    # Normalization
    target_sum = torch.sum(matrix, dim=0)
    norm_matrix = torch.div(matrix, target_sum)
    return norm_matrix.to(device)
