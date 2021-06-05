import torch
import torch.nn as nn
import utils
import shutil
import time
import math
import numpy as np
import argparse
import Data_loader_SSEPT as Data_loader
import os
import random
import SSEPT_Alpha
import ast

torch.manual_seed(10)


def sampler(batch, usernum, itemnum, maxlen):
    threshold_user = 0.2
    threshold_item = 0.99
    new_batch = []
    for seq in batch:

        if random.random() > threshold_user:
            user = np.random.randint(0, usernum)
            seq[0] = user

        idx = maxlen - 1
        for i in reversed(seq[1:-1]):
            if i != 0 and random.random() > threshold_item:
                i = np.random.randint(0, itemnum)
                seq[idx] = i
            idx -= 1
            if idx == 0: break
        new_batch.append(seq)
    new_batch = np.array(new_batch)
    return new_batch


def getBatch(data, batch_size):
    start_inx = 0
    end_inx = batch_size

    while end_inx < len(data):
        batch = data[start_inx:end_inx]
        start_inx = end_inx
        end_inx += batch_size
        yield batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--top_k', type=int, default=5,
                        help='Sample from top k predictions')
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='hyperpara-Adam')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--datapath', type=str, default='Data/movielen_20/movielen_20_context.csv',
                        help='data path')
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--save_dir', default='Models/ML20_baseline_24_emb64_bs128', type=str)
    parser.add_argument('--eval_iter', type=int, default=2000,
                        help='sample generator output evry x steps')
    parser.add_argument('--early_stop', type=int, default=20,
                        help='after x step early stop')
    parser.add_argument('--step', type=int, default=250000,
                        help='trainging step')
    parser.add_argument('--tt_percentage', type=float, default=0.2,
                        help='0.2 means 80% training 20% testing')
    parser.add_argument('--data_ratio', type=float, default=1,
                        help='real trainging data')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--L2', type=float, default=0,
                        help='L2 regularization')
    parser.add_argument('--hidden_factor', type=int, default=128,
                        help='Number of hidden factors, i.e., embedding size.')
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--num_blocks', default=24, type=int)
    parser.add_argument('--dropout', default=0.2, type=float)
    parser.add_argument('--method', type=str, default="from_scratch",
                        help='from_scratch, stack')
    parser.add_argument('--load_model', type=ast.literal_eval, default=False,
                        help='whether loading pretrain model')
    parser.add_argument('--model_path', type=str, default="Models/",
                        help='load model path')
    args = parser.parse_args()
    print(args)

    dl = Data_loader.Data_Loader({'model_type': 'generator', 'dir_name': args.datapath})
    all_samples = dl.items
    items_voc = dl.item2id
    print("shape: ", np.shape(all_samples))
    user_size = dl.user_size

    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(all_samples)))
    all_samples = all_samples[shuffle_indices]

    # Split train/test set
    dev_sample_index = -1 * int(args.tt_percentage * float(len(all_samples)))
    train_set, valid_set = all_samples[:dev_sample_index], all_samples[dev_sample_index:]

    random.seed(10)
    ratio = args.data_ratio
    train_set_len = len(train_set)
    train_index_set = set(list(range(train_set_len)))

    if ratio == 0.2:
        train_ratio = int(ratio * float(train_set_len))
        real_train_index_set = random.sample(list(train_index_set), train_ratio)
        real_train_set = train_set[real_train_index_set]
        train_set = np.array(real_train_set)
        print("real train len", len(train_set))
    elif ratio == 0.4:
        last_ratio = ratio - 0.2
        last_train_ratio = int(last_ratio * float(train_set_len))
        last_train_index_set = random.sample(list(train_index_set), last_train_ratio)
        last_train_set = train_set[last_train_index_set]

        remain_train_index_set = train_index_set - set(last_train_index_set)
        remain_len = len(remain_train_index_set)
        new_train_index_set = random.sample(list(remain_train_index_set), int(1.0 / 4.0 * float(remain_len)))
        new_train_set = train_set[new_train_index_set]
        real_train_set = np.concatenate((last_train_set, new_train_set), axis=0)
        train_set = np.array(real_train_set)
        print("real train len", len(train_set))
    elif ratio == 0.6:
        last_last_ratio = ratio - 0.2 - 0.2
        last_last_train_ratio = int(last_last_ratio * float(train_set_len))
        last_last_train_index_set = random.sample(list(train_index_set), last_last_train_ratio)
        last_last_train_set = train_set[last_last_train_index_set]

        remain_train_index_set = train_index_set - set(last_last_train_index_set)
        remain_len = len(remain_train_index_set)
        last_train_index_set = random.sample(list(remain_train_index_set), int(1.0 / 4.0 * float(remain_len)))
        last_train_set = train_set[last_train_index_set]
        real_train_set = np.concatenate((last_last_train_set, last_train_set), axis=0)

        remain_train_index_set = remain_train_index_set - set(last_train_index_set)
        remain_len = len(remain_train_index_set)
        new_train_index_set = random.sample(list(remain_train_index_set), int(1.0 / 3.0 * float(remain_len)))
        new_train_set = train_set[new_train_index_set]

        real_train_set = np.concatenate((real_train_set, new_train_set), axis=0)
        train_set = np.array(real_train_set)
        print("real train len", len(train_set))
    elif ratio == 0.8:
        last_last_ratio = ratio - 0.2 - 0.2 - 0.2
        last_last_train_ratio = int(last_last_ratio * float(train_set_len))
        last_last_train_index_set = random.sample(list(train_index_set), last_last_train_ratio)
        last_last_train_set = train_set[last_last_train_index_set]

        remain_train_index_set = train_index_set - set(last_last_train_index_set)
        remain_len = len(remain_train_index_set)
        last_train_index_set = random.sample(list(remain_train_index_set), int(1.0 / 4.0 * float(remain_len)))
        last_train_set = train_set[last_train_index_set]
        real_train_set = np.concatenate((last_last_train_set, last_train_set), axis=0)

        remain_train_index_set = remain_train_index_set - set(last_train_index_set)
        remain_len = len(remain_train_index_set)
        new_train_index_set = random.sample(list(remain_train_index_set), int(1.0 / 3.0 * float(remain_len)))
        new_train_set = train_set[new_train_index_set]
        real_train_set = np.concatenate((real_train_set, new_train_set), axis=0)

        remain_train_index_set = remain_train_index_set - set(new_train_index_set)
        remain_len = len(remain_train_index_set)
        new_train_index_set = random.sample(list(remain_train_index_set), int(1.0 / 2.0 * float(remain_len)))
        new_train_set = train_set[new_train_index_set]


        real_train_set = np.concatenate((real_train_set, new_train_set), axis=0)
        train_set = np.array(real_train_set)
        print("real train len", len(train_set))

    elif ratio == 1:

        train_set = np.array(train_set)
        print("real train len", len(train_set))
    else:
        train_ratio = int(ratio * float(train_set_len))
        real_train_index_set = random.sample(list(train_index_set), train_ratio)
        real_train_set = train_set[real_train_index_set]
        train_set = np.array(real_train_set)
        print("real train len", len(train_set))

    model_para = {
        'user_size': user_size,
        'item_size': len(items_voc),
        'emb_size': int(args.hidden_factor / 2),
        'hidden_factor': args.hidden_factor,
        'num_blocks': args.num_blocks,
        'num_heads': args.num_heads,
        'dropout': args.dropout,
        'batch_size': args.batch_size,
        'step': args.step,
        'seq_len': len(all_samples[0]) - 1,
        'learning_rate': args.learning_rate,
        'load_model': args.load_model,
        'model_path': args.model_path,
        'method': args.method
    }
    print(model_para)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = SSEPT_Alpha(model_para, device=args.device, ).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.L2)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    criterion = nn.CrossEntropyLoss()

    numIters = 1
    max_mrr = 0
    break_stick = 0
    early_stop = 0
    while (1):
        if break_stick == 1:
            break
        model.train()
        batch_no = 0
        batch_size = model_para['batch_size']

        for batch_idx, batch_sam in enumerate(getBatch(train_set, batch_size)):
            start = time.time()

            batch_sam = sampler(batch_sam, model_para['user_size'], model_para['item_size'], model_para['seq_len'])
            users, inputs, targets = torch.LongTensor(batch_sam[:, 0:1]).to(args.device), torch.LongTensor(
                batch_sam[:, 1:-1]).to(args.device), torch.LongTensor(batch_sam[:, 2:]).to(
                args.device).view([-1])
            optimizer.zero_grad()
            outputs = model(users, inputs, onecall=False)  # [batch_size*seq_len, item_size]
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            end = time.time()
            if numIters % args.eval_iter == 0:
                print("-------------------------------------------------------train")
                print("LOSS: {}\tBATCH_NO: {}\t STEP:{}\t total_batches:{}".format(
                    loss.item(), batch_no, numIters, train_set.shape[0] / batch_size))
                print("TIME FOR BATCH", end - start)
                print("TIME FOR EPOCH (mins)", (end - start) * (train_set.shape[0] / batch_size) / 60.0)

            batch_no += 1

            if numIters % args.eval_iter == 0:
                print("-------------------------------------------------------test")

                model.eval()
                batch_size_test = batch_size * 1
                curr_preds_5 = []
                rec_preds_5 = []
                ndcg_preds_5 = []
                curr_preds_10 = []
                rec_preds_10 = []
                ndcg_preds_10 = []
                with torch.no_grad():
                    for batch_idx, batch_sam in enumerate(getBatch(valid_set, batch_size)):
                        users, inputs, targets = torch.LongTensor(batch_sam[:, 0:1]).to(args.device), torch.LongTensor(
                            batch_sam[:, 1:-1]).to(args.device), torch.LongTensor(batch_sam[:, -1]).to(
                            args.device).view([-1])
                        outputs = model(users, inputs)  # [batch_size, item_size] only predict the last position

                        _, sort_idx_10 = torch.topk(outputs, k=args.top_k + 5, sorted=True)  # [batch_size, 10]
                        _, sort_idx_5 = torch.topk(outputs, k=args.top_k, sorted=True)  # [batch_size, 5]

                        pred_items_5, pred_items_10, target = sort_idx_5.data.cpu().numpy(), sort_idx_10.data.cpu().numpy(), targets.data.cpu().numpy()
                        for bi in range(pred_items_5.shape[0]):

                            true_item = target[bi]
                            predictmap_5 = {ch: i for i, ch in enumerate(pred_items_5[bi])}
                            predictmap_10 = {ch: i for i, ch in enumerate(pred_items_10[bi])}

                            rank_5 = predictmap_5.get(true_item)
                            rank_10 = predictmap_10.get(true_item)
                            if rank_5 == None:
                                curr_preds_5.append(0.0)
                                rec_preds_5.append(0.0)
                                ndcg_preds_5.append(0.0)
                            else:
                                MRR_5 = 1.0 / (rank_5 + 1)
                                Rec_5 = 1.0  # 3
                                ndcg_5 = 1.0 / math.log(rank_5 + 2, 2)  # 3
                                curr_preds_5.append(MRR_5)
                                rec_preds_5.append(Rec_5)  # 4
                                ndcg_preds_5.append(ndcg_5)  # 4
                            if rank_10 == None:
                                curr_preds_10.append(0.0)
                                rec_preds_10.append(0.0)  # 2
                                ndcg_preds_10.append(0.0)  # 2
                            else:
                                MRR_10 = 1.0 / (rank_10 + 1)
                                Rec_10 = 1.0  # 3
                                ndcg_10 = 1.0 / math.log(rank_10 + 2, 2)  # 3
                                curr_preds_10.append(MRR_10)
                                rec_preds_10.append(Rec_10)  # 4
                                ndcg_preds_10.append(ndcg_10)  # 4

                    mrr = sum(curr_preds_5) / float(len(curr_preds_5))
                    mrr_10 = sum(curr_preds_10) / float(len(curr_preds_10))
                    hit = sum(rec_preds_5) / float(len(rec_preds_5))
                    hit_10 = sum(rec_preds_10) / float(len(rec_preds_10))
                    ndcg = sum(ndcg_preds_5) / float(len(ndcg_preds_5))
                    ndcg_10 = sum(ndcg_preds_10) / float(len(ndcg_preds_10))

                    if mrr > max_mrr:
                        max_mrr = mrr
                        print("Save model!  mrr_5:", mrr)
                        print("Save model!  mrr_10:", mrr_10)
                        print("Save model!  hit_5:", hit)
                        print("Save model!  hit_10:", hit_10)
                        print("Save model!  ndcg_5:", ndcg)
                        print("Save model!  ndcg_10:", ndcg_10)
                        torch.save(model.state_dict(),
                                   args.save_dir + "/{}_{}_{}_{}.pkl".format(args.num_blocks, args.learning_rate,
                                                                             args.data_ratio, args.step))
                        early_stop = 0
                    else:
                        print("mrr_5:", mrr)
                        print("mrr_10:", mrr_10)
                        print("hit_5:", hit)
                        print("hit_10:", hit_10)
                        print("ndcg_5:", ndcg)
                        print("ndcg_10:", ndcg_10)
                        early_stop += 1
            if numIters >= model_para['step']:
                break_stick = 1
                break
            if early_stop >= args.early_stop:
                break_stick = 1
                print("early stop!")
                break

            numIters += 1


if __name__ == '__main__':
    main()


