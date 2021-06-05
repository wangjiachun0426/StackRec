import os
import numpy as np
import torch


def to_pickled_df(data_directory, **kwargs):
    for name, df in kwargs.items():
        df.to_pickle(os.path.join(data_directory, name + '.df'))


def pad_history(itemlist, length, pad_item):
    if len(itemlist)>=length:
        return itemlist[-length:]
    if len(itemlist)<length:
        temp = [pad_item] * (length-len(itemlist))
        itemlist.extend(temp)
        return itemlist


def torch_gather_nd(data, indices):
    """
    This function is taken from Haozhe Xie's reply
    on the page, implementing tf.gather_nd in pytorch
    https://discuss.pytorch.org/t/implement-tf-gather-nd-in-pytorch/37502
    """
    x = []
    for i in range(len(indices)):
        x.append(data[i,indices[i][-1],:].tolist())
    x = torch.Tensor(x)
    return x


def extract_axis_1_torch(data, ind):
    """
    Get specified elements along the first axis of tensor.
    :param data: Tensorflow tensor that will be subsetted.
    :param ind: Indices to take (one for each element along axis 0 of data).
    :return: Subsetted tensor.
    """

    batch_range = torch.range(start=0, end=data.size(0)-1)
    batch_range = batch_range.long()
    ind = ind.long()
    indices = torch.stack([batch_range, ind], dim=1)
    res = torch_gather_nd(data, indices)

    return res


def normalize(inputs,epsilon=1e-8):
    '''Applies layer normalization.

    Args:
      inputs: A tensor with 2 or more dimensions, where the first dimension has
        `batch_size`.
      epsilon: A floating number. A very small number for preventing ZeroDivision Error.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    inputs_shape = inputs.size()
    #print (inputs_shape)
    params_shape = inputs_shape[-1:]

    mean = inputs.mean(dim=-1,keepdim=True)
    #print (mean.size())
    variance = inputs.var(dim=-1,keepdim=True)
    beta = torch.zeros(params_shape)
    gamma = torch.ones(params_shape)
    num = inputs-mean
    deno = variance+epsilon
    normalized = (num) / ((deno) ** (.5))
    outputs = gamma * normalized + beta

    return outputs


def calculate_hit(sorted_list,topk,true_items,rewards,r_click,total_reward,hit_click,ndcg_click,hit_purchase,ndcg_purchase):
    for i in range(len(topk)):
        rec_list = sorted_list[:, -topk[i]:]
        for j in range(len(true_items)):
            if true_items[j] in rec_list[j]:
                rank = topk[i] - np.argwhere(rec_list[j] == true_items[j])
                total_reward[i] += rewards[j]
                if rewards[j] == r_click:
                    hit_click[i] += 1.0
                    ndcg_click[i] += 1.0 / np.log2(rank + 1)
                else:
                    hit_purchase[i] += 1.0
                    ndcg_purchase[i] += 1.0 / np.log2(rank + 1)


def set_device():
    is_cuda = torch.cuda.is_available()

    # If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
    if is_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device

