import pandas
from tqdm import tqdm
import numpy as np
import torch
from collections import Counter
import math
import os,pickle
import itertools
from scipy.ndimage.interpolation import shift
import sys
sys.path.append('/Users/aishwaryaya/dir_1/eicu_data/data/tf_rmtpp/src/')
import keys_vocab

from util_eicu_0 import process_data_generate_onlyevents,patientinfo

feature_keys,vocab_sizes=keys_vocab.get_keys_vocab()


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def softmax(x):
    x = np.exp(x)
    return x / x.sum()

def generate_data(files,event_train_path,min_num_codes,max_num_codes,seq_len):
    time_seqs=[]
    event_seqs=[]

    for f in files:
            file_gcs=os.path.join(event_train_path,f)
            with open(file_gcs,"rb") as pklfile:
                train_data=pickle.load(pklfile) 
                #print("file_gcs",file_gcs)

                (train_event_list,train_time_list,train_out_event_list,train_out_time_list,
                 train_mask_list, train_in_event_list,
                 train_in_time_list)=process_data_generate_onlyevents(train_data, min_num_codes=min_num_codes,
                                                       max_num_codes=max_num_codes,
                                                   step_size=seq_len-1, feature_keys=feature_keys)
                # print("train_in_time_list",train_in_time_list)
                # print("train_in_event_list",train_in_event_list)
                time_seqs.extend(train_in_time_list)
                event_seqs.extend(train_in_event_list)

    return time_seqs,event_seqs

def read_data(seq_len):
    

    event_train_path='/Users/aishwaryaya/dir_1/eicu_data/data/patient_traindata_new_1'
    max_num_codes=10
    min_num_codes=1
    files=[f for f in os.listdir(event_train_path) if 'patient_traingcs' in f]
    fileno=len(files)
    print("fileno",fileno)
    n=2
    train_files=files[:n]
    #test_files=files[n:]
    # print("number of train_files ",len(train_files))
    # print("number of test_files ",len(test_files))
    test_files=[files[n+1]]
    train_time_seqs,train_event_seqs = generate_data(train_files,event_train_path,min_num_codes,max_num_codes,seq_len)
    test_time_seqs,test_event_seqs = generate_data(test_files,event_train_path,min_num_codes,max_num_codes,seq_len)
   
    maxTime = max(itertools.chain((max(x) for x in train_time_seqs), (max(x) for x in test_time_seqs)))
    minTime = min(itertools.chain((min(x) for x in train_time_seqs), (min(x) for x in test_time_seqs)))

    eventTrainIn = [x for x in train_event_seqs]
    #eventTrainOut = [x[1:] for x in train_event_seqs]
    timeTrainIn = [[(y - minTime) / (maxTime - minTime) if y!=0. else 0. for y in x] for x in train_time_seqs]
    #timeTrainOut = [[(y - minTime) / (maxTime - minTime) if y!=0. else 0. for y in x[1:]] for x in train_time_seqs]

    eventTestIn = [x for x in test_event_seqs]
    #eventTestOut = [x[1:] for x in test_event_seqs]
    timeTestIn = [[(y - minTime) / (maxTime - minTime) if y!=0. else 0. for y in x] for x in test_time_seqs]
    #timeTestOut = [[(y - minTime) / (maxTime - minTime) if y!=0. else 0. for y in x[1:]] for x in test_time_seqs]

    return eventTrainIn,timeTrainIn,eventTestIn,timeTestIn


class EicuDataset:
    def __init__(self, config, subset):
        self.subset = subset
        self.max_num_codes=10
        self.min_num_codes=1
        self.config = config
        self.seq_len = config.seq_len
        self.event_train_path='/Users/aishwaryaya/dir_1/eicu_data/data/patient_traindata_new_1'

        files=[f for f in os.listdir(self.event_train_path) if 'patient_traingcs' in f]
        fileno=len(files)
        n=2
        if subset=='train':
            self.train_files=files[:n]
        if subset=='test':
            self.train_files=[files[n+1]]

        self.time_seqs, self.event_seqs = self.generate_sequence()

    def generate_sequence(self):
        time_seqs=[]
        event_seqs=[]

        for f in self.train_files:
            file_gcs=os.path.join(self.event_train_path,f)
            with open(file_gcs,"rb") as pklfile:
                train_data=pickle.load(pklfile) 
                #print("file_gcs",file_gcs)

                (train_event_list,train_time_list,train_out_event_list,train_out_time_list,
                 train_mask_list, train_in_event_list,
                 train_in_time_list)=process_data_generate_onlyevents(train_data, min_num_codes=self.min_num_codes,
                                                       max_num_codes=self.max_num_codes,
                                                   step_size=self.seq_len-1, feature_keys=feature_keys)
                # print("train_in_time_list",train_in_time_list)
                # print("train_in_event_list",train_in_event_list)
                time_seqs.extend(train_in_time_list)
                event_seqs.extend(train_in_event_list)

        
        return time_seqs,event_seqs


    def __getitem__(self, item):
        return self.time_seqs[item], self.event_seqs[item]

    def __len__(self):
        return len(self.time_seqs)

    @staticmethod
    def to_features(batch):
        times, events = [], []
        for time, event in batch:
            # print("time1",len(time),time)
            # time = np.array([time[0]] + time)
            # time = np.diff(time)
            # print("time2",len(time),time)
            # print("event",len(event),event)
            times.append(time[:-1])
            events.append(event[:-1] *0)
        return torch.FloatTensor(times), torch.LongTensor(events)


    @staticmethod
    def to_features_1(batch):
        times, events = [], []
        times_tar, events_tar = [], []
        times_abs=[]
        for time, event in batch:
            # print("event",event)
            # print("time",time)
            time_new=np.copy(time)
            event_new=np.copy(event)

            ind=np.max(np.nonzero(time_new))
            time_nonzero=time_new[:ind+1]
            time_shift=time_nonzero-shift(time_nonzero,1)
            time_new[:ind+1]=time_shift

            tar_t=time_new[ind]-time_nonzero[-1]
            time_new[ind]=0.
            print("times",time_new[:-1],time_nonzero,tar_t)
            times.append(time_new[:-1])
            times_tar.append(tar_t)
            times_abs.append(np.copy(time)[:-1])
            
            events.append(event_new[:-1] *0)
            events_tar.append(event_new[-1]*0)

            # times_tar.append(time[-1])
            # times.append(time[:-1])
            
        return torch.FloatTensor(times), torch.FloatTensor(times_tar),torch.LongTensor(events),torch.LongTensor(events_tar),torch.FloatTensor(times_abs)

    @staticmethod
    def to_features_2(batch):
        times, events = [], []
        times_tar, events_tar = [], []
        for time, event in batch:
            # print("event",event)
            # print("time",time)
            time_new=np.copy(time)
            event_new=np.copy(event)

            ind=np.max(np.nonzero(time_new))
            time_nonzero=time_new[:ind+1]
            time_nonzero= np.array([time_nonzero[0]] + time_nonzero)
            time_shift=np.diff(time)
            time_new[:ind+1]=time_shift

            tar_t=time_new[ind]
            time_new[ind]=0.
            #print("times",time_new[:-1],tar_t)
            times.append(time_new[:-1])
            times_tar.append(tar_t)
            
            events.append(event_new[:-1] *0)
            events_tar.append(event_new[-1]*0)

            # times_tar.append(time[-1])
            # times.append(time[:-1])
            
        return torch.FloatTensor(times), torch.FloatTensor(times_tar),torch.LongTensor(events),torch.LongTensor(events_tar)


class EicuDataset_new:
    def __init__(self, eventTrainIn,timeTrainIn,seq_len):
        
        self.time_seqs=timeTrainIn
        self.event_seqs = eventTrainIn


    


    def __getitem__(self, item):
        return self.time_seqs[item], self.event_seqs[item]

    def __len__(self):
        return len(self.time_seqs)

    @staticmethod
    def to_features(batch):
        times, events = [], []
        for time, event in batch:
            # print("time1",len(time),time)
            # time = np.array([time[0]] + time)
            # time = np.diff(time)
            # print("time2",len(time),time)
            # print("event",len(event),event)
            times.append(time[:-1])
            events.append(event[:-1] *0)
        return torch.FloatTensor(times), torch.LongTensor(events)


    @staticmethod
    def to_features_1(batch):
        times, events = [], []
        times_tar, events_tar = [], []
        times_abs=[]
        for time, event in batch:
            # print("event",event)
            #print("time",time)
            time_new=np.copy(time)
            event_new=np.copy(event)

            ind=np.max(np.nonzero(time_new))
            time_nonzero=time_new[:ind+1]

            time_shift=time_nonzero-shift(time_nonzero,1)
            time_new[:ind+1]=time_shift

            tar_t=time_new[ind]
            time_new[ind]=0.
            #print("times",time,time_new[:-1],time_nonzero,tar_t)
            #print("times",time_new[:-1],tar_t)
            times.append(time_new[:-1])
            times_tar.append(tar_t)
            times_abs.append(np.copy(time)[:-1])
            #print("tar_t",tar_t)
            
            events.append(event_new[:-1] *0)
            events_tar.append(event_new[-1]*0)

            # times_tar.append(time[-1])
            # times.append(time[:-1])
            
        return torch.FloatTensor(times), torch.FloatTensor(times_tar),torch.LongTensor(events),torch.LongTensor(events_tar),torch.FloatTensor(times_abs)

    @staticmethod
    def to_features_2(batch):
        times, events = [], []
        times_tar, events_tar = [], []
        for time, event in batch:
            # print("event",event)
            # print("time",time)
            time_new=np.copy(time)
            event_new=np.copy(event)

            ind=np.max(np.nonzero(time_new))
            time_nonzero=time_new[:ind+1]
            time_nonzero= np.array([time_nonzero[0]] + time_nonzero)
            time_shift=np.diff(time)
            time_new[:ind+1]=time_shift

            tar_t=time_new[ind]
            time_new[ind]=0.
            #print("times",time_new[:-1],tar_t)
            times.append(time_new[:-1])
            times_tar.append(tar_t)
            
            events.append(event_new[:-1] *0)
            events_tar.append(event_new[-1]*0)

            # times_tar.append(time[-1])
            # times.append(time[:-1])
            
        return torch.FloatTensor(times), torch.FloatTensor(times_tar),torch.LongTensor(events),torch.LongTensor(events_tar)


class ATMDataset:
    def __init__(self, config, subset):
        data = pandas.read_csv("data/"+subset+"_day.csv")
        self.subset = subset
        n=100
        self.id = list(data['id'])
        self.time = list(data['time'])
        self.event = list(data['event'])
        self.config = config
        self.seq_len = config.seq_len
        self.time_seqs, self.event_seqs = self.generate_sequence()
        # self.time_seqs = self.time_seqs[:n]
        # self.event_seqs = self.event_seqs[:n]
        self.statistic()

    def generate_sequence(self):
        MAX_INTERVAL_VARIANCE = 1
        pbar = tqdm(total=len(self.id) - self.seq_len + 1)
        time_seqs = []
        event_seqs = []
        cur_end = self.seq_len - 1
        while cur_end < len(self.id):
            pbar.update(1)
            cur_start = cur_end - self.seq_len + 1
            if self.id[cur_start] != self.id[cur_end]:
                cur_end += 1
                continue

            subseq = self.time[cur_start:cur_end + 1]
            # if max(subseq) - min(subseq) > MAX_INTERVAL_VARIANCE:
            #     if self.subset == "train":
            #         cur_end += 1
            #         continue

            time_seqs.append(self.time[cur_start:cur_end + 1])
            event_seqs.append(self.event[cur_start:cur_end + 1])
            cur_end += 1
        return time_seqs, event_seqs

    def __getitem__(self, item):
        return self.time_seqs[item], self.event_seqs[item]

    def __len__(self):
        return len(self.time_seqs)

    @staticmethod
    def to_features(batch):
        times, events = [], []
        for time, event in batch:
            #print("time0",len(time),time)
            time = np.array([time[0]] + time)
            time = np.diff(time)
            # print("time",len(time),time)
            # print("event",len(event),event)
            times.append(time)
            events.append(event)
        return torch.FloatTensor(times), torch.LongTensor(events)

    def statistic(self):
        print("TOTAL SEQs:", len(self.time_seqs))
        # for i in range(10):
        #     print(self.time_seqs[i], "\n", self.event_seqs[i])
        intervals = np.diff(np.array(self.time))
        for thr in [0.001, 0.01, 0.1, 1, 10, 100]:
            print("<",{thr}, "=", {np.mean(intervals < thr)})

    def importance_weight(self):
        count = Counter(self.event)
        percentage = [count[k] / len(self.event) for k in sorted(count.keys())]
        for i, p in enumerate(percentage):
            print("event",{i}," =", {p * 100},"%")
        weight = [len(self.event) / count[k] for k in sorted(count.keys())]
        return weight


def abs_error(pred, gold):
    return np.mean(np.abs(pred - gold))


def clf_metric(pred, gold, n_class):
    gold_count = Counter(gold)
    pred_count = Counter(pred)
    prec = recall = 0
    pcnt = rcnt = 0
    for i in range(n_class):
        match_count = np.logical_and(pred == gold, pred == i).sum()
        if gold_count[i] != 0:
            prec += match_count / gold_count[i]
            pcnt += 1
        if pred_count[i] != 0:
            recall += match_count / pred_count[i]
            rcnt += 1
    prec /= pcnt
    recall /= rcnt
    print("pcnt=",{pcnt}, "rcnt=",{rcnt})
    f1 = 2 * prec * recall / (prec + recall)
    return prec, recall, f1
