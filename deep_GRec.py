import tensorflow as tf
import data_loader
import generator_deep_GRec as generator_deep
import shutil
import time
import math
import numpy as np
import argparse
import sys
import os
import random
import ast
import collections


tf.set_random_seed(10)


MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])


def create_masked_lm_predictions_frombatch(item_batch, masked_lm_prob,
                                           max_predictions_per_seq, items, rng, item_size):
    rng = random.Random()
    output_tokens_batch = []
    maskedpositions_batch = []
    maskedlabels_batch = []
    masked_lm_weights_batch = []
    item_batch_ = item_batch[:, 1:]  # remove start and end
    for line_list in range(item_batch_.shape[0]):

        output_tokens, masked_lm_positions, masked_lm_labels = create_masked_lm_predictions(item_batch_[line_list],
                                                                                            masked_lm_prob,
                                                                                            max_predictions_per_seq,
                                                                                            items, rng, item_size)
        # print output_tokens
        output_tokens.insert(0, item_batch[line_list][0])
        output_tokens_batch.append(output_tokens)
        maskedpositions_batch.append(masked_lm_positions)
        maskedlabels_batch.append(masked_lm_labels)
        masked_lm_weights = [1.0] * len(masked_lm_labels)
        # note you can not change here since it should be consistent with 'num_to_predict' in create_masked_lm_predictions
        num_to_predict = min(max_predictions_per_seq,
                             max(1, int(round(len(item_batch_[line_list]) * masked_lm_prob))))

        while len(masked_lm_weights) < num_to_predict:
            masked_lm_weights.append(0.0)
        masked_lm_weights_batch.append(masked_lm_weights)

    return output_tokens_batch, maskedpositions_batch, maskedlabels_batch, masked_lm_weights_batch


def create_masked_predictions_frombatch(item_batch):
    output_tokens_batch = []
    maskedpositions_batch = []
    maskedlabels_batch = []
    for line_list in range(item_batch.shape[0]):
        output_tokens, masked_lm_positions, masked_lm_labels = create_endmask(item_batch[line_list])
        output_tokens_batch.append(output_tokens)
        maskedpositions_batch.append(masked_lm_positions)
        maskedlabels_batch.append(masked_lm_labels)
    return output_tokens_batch, maskedpositions_batch, maskedlabels_batch


def create_endmask(tokens):
    masked_lm_positions = []
    masked_lm_labels = []
    lens = len(tokens)
    masked_token = 0
    dutokens = list(tokens)
    dutokens[-1] = masked_token

    masked_lm_positions.append(lens - 1)
    masked_lm_labels.append(tokens[-1])
    return dutokens, masked_lm_positions, masked_lm_labels


# from BERT
def create_masked_lm_predictions(tokens, masked_lm_prob,
                                 max_predictions_per_seq, vocab_words, rng, item_size):
    """Creates the predictions for the masked LM objective."""

    cand_indexes = []
    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[SEP]":
            continue
        cand_indexes.append(i)

    rng.shuffle(cand_indexes)

    output_tokens = list(tokens)

    num_to_predict = min(max_predictions_per_seq,
                         max(1, int(round(len(tokens) * masked_lm_prob))))

    masked_lms = []
    covered_indexes = set()
    for index in cand_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        if index in covered_indexes:
            continue
        covered_indexes.add(index)

        masked_token = None
        # 80% of the time, replace with [MASK]
        if rng.random() < 1.0:
            # masked_token = "[MASK]"
            masked_token = 0  # item_size is "[MASK]"   0 represents '<unk>'
        else:
            # 10% of the time, keep original
            if rng.random() < 0.5:
                masked_token = tokens[index]
            # 10% of the time, replace with random word
            else:
                masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]

        output_tokens[index] = masked_token

        masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))

    masked_lms = sorted(masked_lms, key=lambda x: x.index)

    masked_lm_positions = []
    masked_lm_labels = []
    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)

    return (output_tokens, masked_lm_positions, masked_lm_labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--top_k', type=int, default=5,
                        help='sample from top k predictions')
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='hyperpara-Adam')
    parser.add_argument('--datapath', type=str, default="Data/movielen_20/movielen_20.csv",
                        help='data path')
    parser.add_argument('--save_dir', type=str, default="Models/ml20",
                        help='save dir path')
    parser.add_argument('--eval_iter', type=int, default=1000,
                        help='sample generator output evry x steps')
    parser.add_argument('--early_stop', type=int, default=10,
                        help='after x step early stop')
    parser.add_argument('--step', type=int, default=400000,
                        help='trainging step')
    parser.add_argument('--tt_percentage', type=float, default=0.2,
                        help='0.2 means 80% training 20% testing')
    parser.add_argument('--data_ratio', type=float, default=1,
                        help='real trainging data')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--L2', type=float, default=0,
                        help='L2 regularization')
    parser.add_argument('--dilation_count', type=int, default=16,
                        help='dilation count number')
    parser.add_argument('--method', type=str, default="from_scratch",
                        help='from_scratch, StackR, stackC, stackA')
    parser.add_argument('--load_model', type=ast.literal_eval, default=False,
                        help='whether loading pretrain model')
    parser.add_argument('--model_path', type=str, default="Models/",
                        help='load model path')
    parser.add_argument('--padid', type=int, default=0,
                        help='pad id')

    parser.add_argument('--masked_lm_prob', type=float, default=0.2,
                        help='0.2 means 20% items are masked')
    parser.add_argument('--max_predictions_per_seq', type=int, default=50,
                        help='maximum number of masked tokens')
    parser.add_argument('--max_position', type=int, default=100,
                        help='maximum number of for positional embedding, it has to be larger than the sequence lens')
    parser.add_argument('--has_positionalembedding', type=bool, default=False,
                        help='whether contains positional embedding before performing cnnn')

    args = parser.parse_args()

    print(args)

    dl = data_loader.Data_Loader({'dir_name': args.datapath, 'padid': args.padid})
    all_samples = dl.item
    print(all_samples.shape)
    items = dl.item_dict
    itemlist = items.values()
    item_size = len(items)
    print("len(items)", item_size)

    max_predictions_per_seq = args.max_predictions_per_seq
    masked_lm_prob = args.masked_lm_prob

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
        'item_size': len(items),
        'dilated_channels': 64,
        'dilations': [1, 4] * args.dilation_count,
        'step': args.step,
        'kernel_size': 3,
        'learning_rate': args.learning_rate,
        'L2': args.L2,
        'batch_size': 1024,
        'load_model': args.load_model,
        'model_path': args.model_path,
        'method': args.method,
        'max_position': args.max_position,
        # maximum number of for positional embedding, it has to be larger than the sequence lens
        'has_positionalembedding': args.has_positionalembedding
    }

    print(model_para)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    itemrec = generator_deep.NextItNet_Decoder(model_para)
    itemrec.train_graph()
    optimizer = tf.train.AdamOptimizer(model_para['learning_rate'], beta1=args.beta1).minimize(itemrec.loss)
    itemrec.predict_graph(reuse=True)

    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    saver = tf.train.Saver(max_to_keep=1)

    init = tf.global_variables_initializer()
    sess.run(init)

    # writer=tf.summary.FileWriter('./stack_graph',sess.graph)

    numIters = 1
    max_mrr = 0
    break_stick = 0
    early_stop = 0
    while (1):
        if break_stick == 1:
            break

        batch_no = 0
        batch_size = model_para['batch_size']

        while (batch_no + 1) * batch_size < train_set.shape[0]:

            start = time.time()

            item_batch = train_set[batch_no * batch_size: (batch_no + 1) * batch_size, :]

            output_tokens_batch, maskedpositions_batch, maskedlabels_batch, masked_lm_weights_batch = create_masked_lm_predictions_frombatch(
                item_batch, masked_lm_prob, max_predictions_per_seq, items=itemlist, rng=None, item_size=item_size
            )

            _, loss = sess.run(
                [optimizer, itemrec.loss],
                feed_dict={
                    itemrec.itemseq_output: item_batch[:, 1:],  # 2 3 4 5 6 7 8 9
                    itemrec.itemseq_input: output_tokens_batch,  # 1 2 0 4 5 0 7 8 9
                    itemrec.masked_position: maskedpositions_batch,  # [1 4]
                    itemrec.masked_items: maskedlabels_batch,  # [3,6]
                    itemrec.label_weights: masked_lm_weights_batch  # [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0] #useless
                })
            end = time.time()
            if numIters % args.eval_iter == 0:
                print("-------------------------------------------------------train")
                print("LOSS: {}\tBATCH_NO: {}\t STEP:{}\t total_batches:{}".format(
                    loss, batch_no, numIters, train_set.shape[0] / batch_size))
                print("TIME FOR BATCH", end - start)
                print("TIME FOR EPOCH (mins)", (end - start) * (train_set.shape[0] / batch_size) / 60.0)

            batch_no += 1

            if numIters % args.eval_iter == 0:
                print("-------------------------------------------------------test")
                batch_no_test = 0
                batch_size_test = batch_size * 1
                curr_preds_5 = []
                rec_preds_5 = []
                ndcg_preds_5 = []
                curr_preds_10 = []
                rec_preds_10 = []
                ndcg_preds_10 = []
                while (batch_no_test + 1) * batch_size_test < valid_set.shape[0]:
                    item_batch = valid_set[batch_no_test * batch_size_test: (batch_no_test + 1) * batch_size_test, :]
                    [probs_10, probs_5] = sess.run(
                        [itemrec.top_10, itemrec.top_5],
                        feed_dict={
                            itemrec.itemseq_input: item_batch[:, 0:-1]
                        })
                    # print(probs_10[1].shape) #(256,1,10)
                    for bi in range(batch_size_test):
                        pred_items_10 = probs_10[1][bi][-1]
                        pred_items_5 = probs_5[1][bi][-1]

                        true_item = item_batch[bi][-1]
                        predictmap_5 = {ch: i for i, ch in enumerate(pred_items_5)}
                        pred_items_10 = {ch: i for i, ch in enumerate(pred_items_10)}

                        rank_5 = predictmap_5.get(true_item)
                        rank_10 = pred_items_10.get(true_item)
                        if rank_5 == None:
                            curr_preds_5.append(0.0)
                            rec_preds_5.append(0.0)
                            ndcg_preds_5.append(0.0)
                        else:
                            MRR_5 = 1.0 / (rank_5 + 1)
                            Rec_5 = 1.0
                            ndcg_5 = 1.0 / math.log(rank_5 + 2, 2)
                            curr_preds_5.append(MRR_5)
                            rec_preds_5.append(Rec_5)
                            ndcg_preds_5.append(ndcg_5)
                        if rank_10 == None:
                            curr_preds_10.append(0.0)
                            rec_preds_10.append(0.0)
                            ndcg_preds_10.append(0.0)
                        else:
                            MRR_10 = 1.0 / (rank_10 + 1)
                            Rec_10 = 1.0
                            ndcg_10 = 1.0 / math.log(rank_10 + 2, 2)
                            curr_preds_10.append(MRR_10)
                            rec_preds_10.append(Rec_10)
                            ndcg_preds_10.append(ndcg_10)

                    batch_no_test += 1

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
                    early_stop = 0
                    saver.save(sess, args.save_dir + "/{}_{}_{}_{}.ckpt".format(args.dilation_count, args.learning_rate,
                                                                                args.data_ratio, args.step))
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
