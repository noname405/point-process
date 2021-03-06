#from util_0 import *
from util_new_0 import *
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from model_0 import Net
from tqdm import tqdm
import numpy as np


def evaluate():
    model.eval()
    pred_times, pred_events = [], []
    gold_times, gold_events = [], []
    for i, batch in enumerate(tqdm(test_loader)):
        gold_times.append(batch[0][:, -1].numpy())
        gold_events.append(batch[1][:, -1].numpy())
        pred_time, pred_event = model.predict(batch)
        pred_times.append(pred_time)
        pred_events.append(pred_event)
        break
    pred_times = np.concatenate(pred_times).reshape(-1)
    gold_times = np.concatenate(gold_times).reshape(-1)
    pred_events = np.concatenate(pred_events).reshape(-1)
    gold_events = np.concatenate(gold_events).reshape(-1)
    print("gold_times",gold_times)
    print("pred_times",pred_times)
    time_error = abs_error(pred_times, gold_times)
    acc, recall, f1 = clf_metric(pred_events, gold_events, n_class=config.event_class)
    print("epoch" ,{epc})
    print("time_error:", {time_error}, "PRECISION:", {acc}, "RECALL:", {recall}, "F1:" ,{f1})


def evaluate_1(model_name):
    model.eval()
    pred_times, pred_events = [], []
    gold_times, gold_events = [], []
    for i, batch in enumerate(tqdm(test_loader)):
        _, gold_time,_, gold_event,_,gold_abs=batch
        gold_times.append(gold_abs.numpy())
        gold_events.append(gold_event.numpy())
        lamb=0
        if model_name == 'rmtpp':
            pred_time,_= model.predict_2(batch)
        else:
            pred_time,_,lamb = model.predict_3(batch)
        pred_times.append(pred_time)
        #pred_events.append(pred_event)
    pred_times = np.concatenate(pred_times).reshape(-1)
    gold_times = np.concatenate(gold_times).reshape(-1)
    #pred_events = np.concatenate(pred_events).reshape(-1)
    #gold_events = np.concatenate(gold_events).reshape(-1)
    print("gold_times",gold_times)
    print("pred_times",pred_times)
    time_error = abs_error(pred_times, gold_times)
    #acc, recall, f1 = clf_metric(pred_events, gold_events, n_class=config.event_class)
    print("epoch" ,{epc})
    print("time_error:", {time_error} ) #, "PRECISION:", {acc}, "RECALL:", {recall}, "F1:" ,{f1})
    print("lambda:", {lamb} )


if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument("--name", type=str, default="default")
    parser.add_argument("--model", type=str, default="erpp", help="erpp, rmtpp")
    parser.add_argument("--seq_len", type=int, default=10)
    parser.add_argument("--emb_dim", type=int, default=10)
    parser.add_argument("--hid_dim", type=int, default=32)
    parser.add_argument("--mlp_dim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--batch_size", type=int, default=4)
    #parser.add_argument("--event_class", type=int, default=7)
    parser.add_argument("--event_class", type=int, default=1)
    parser.add_argument("--verbose_step", type=int, default=30)
    parser.add_argument("--importance_weight", action="store_true")
    parser.add_argument("--lr", type=int, default=1e-3)
    parser.add_argument("--epochs", type=int, default=1000)
    config = parser.parse_args()

    # train_set = ATMDataset(config, subset='train')
    # test_set = ATMDataset(config, subset='test')
    # train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, collate_fn=ATMDataset.to_features)
    # test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False, collate_fn=ATMDataset.to_features)

    # train_set = EicuDataset(config, subset='train')
    # test_set = EicuDataset(config, subset='test')
    # train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, collate_fn=EicuDataset.to_features_1)
    # test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False, collate_fn=EicuDataset.to_features_1)

    eventTrainIn,timeTrainIn,eventTestIn,timeTestIn,maxTime,minTime=read_data(config.seq_len)
    #print("timeTrainIn",timeTrainIn)
    train_set = EicuDataset_new(eventTrainIn,timeTrainIn,config.seq_len)
    test_set = EicuDataset_new(eventTestIn,timeTestIn,config.seq_len)
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, collate_fn=EicuDataset_new.to_features_1)
    test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False, collate_fn=EicuDataset_new.to_features_1)

    weight = np.ones(config.event_class)
    # if config.importance_weight:
    #     weight = train_set.importance_weight()
    #     print("importance weight: ", weight)
    model = Net(config, lossweight=weight)
    model.set_optimizer(total_step=len(train_loader) * config.epochs, use_bert=False)
    #model.cuda()

    for epc in range(config.epochs):
        model.train()
        range_loss1 = range_loss2 = range_loss = 0
        for i, batch in enumerate(tqdm(train_loader)):
            #print("batch",batch)
            l1, l2, l = model.train_batch_2(batch)
            #l1, l2, l = model.train_batch(batch)
            range_loss1 += l1
            range_loss2 += l2
            range_loss += l

            if (i + 1) % config.verbose_step == 0:
                print("time loss: ", range_loss1 / config.verbose_step)
                #print("event loss:", range_loss2 / config.verbose_step)
                print("total loss:", range_loss / config.verbose_step)
                range_loss1 = range_loss2 = range_loss = 0

                evaluate_1(config.model)
                #evaluate()

        # if i%1==0:
        #     #evaluate_1()
        #     evaluate()
