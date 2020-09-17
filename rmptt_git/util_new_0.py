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
#import keys_vocab
import pandas as pd

#from util_eicu_0 import process_data_generate_onlyevents,patientinfo

#feature_keys,vocab_sizes=keys_vocab.get_keys_vocab()


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def softmax(x):
    x = np.exp(x)
    return x / x.sum()




def read_nursechart(seq_len):

    filepath='/Users/aishwaryaya/dir_1/eicu_data/eicuData/'
    chart_fname=filepath+'nurseCharting.csv.gz'
    chartdf=pd.read_csv(chart_fname)

    gcs_df=chartdf.loc[chartdf['nursingchartcelltypevallabel']=='Score (Glasgow Coma Scale)']

    gcs_group=gcs_df.groupby('patientunitstayid')#['nursingchartoffset']

    time_seq=[]
    event_seq=[]
    for i,group_df in gcs_group:
        time=[]
        event=[]
        sort_df=group_df.sort_values(by=['nursingchartoffset'])
        gcs_val=sort_df['nursingchartvalue'].astype(int).to_numpy()
        time_val=sort_df['nursingchartoffset'].to_numpy()
        less_8_time=time_val[(gcs_val <=8)]
        #time.append(less_8_time)
        time=list(less_8_time)

        gcs_val3=list(gcs_val)
        gcs_val1 = np.array([gcs_val3[0]] + gcs_val3)
        gcs_val2 = np.diff(gcs_val1)

        less_time=time_val[gcs_val2<0]

        time.extend(less_time)

        

        time=np.unique(np.array(time))
        time=np.sort(time)

        if len(time) <= 2:
            continue

        if len(time)>seq_len:
            time=time[:seq_len]

        event=np.ones_like(time)
        time_seq.append(time)
        event_seq.append(event)
        
    return time_seq,event_seq
    


def read_data(seq_len):

    time_seq,event_seq=read_nursechart(seq_len)
    fileno=len(time_seq)
    print("fileno",fileno)
    n=int(fileno*0.8) #2
    
    
    #test_files=[files[n+1]]
    eventTrain = event_seq[:n]
    timeTrain = time_seq[:n]
    eventTest =event_seq[n:]
    timeTest = time_seq[n:]
    
    print("number of train_files ",len(timeTrain))
    print("number of test_files ",len(timeTest))

    maxTime = max(itertools.chain((max(x) for x in timeTrain), (max(x) for x in timeTest)))
    minTime = min(itertools.chain((min(x) for x in timeTrain), (min(x) for x in timeTest)))
    # minTime, maxTime = 0, 1

    eventTrainIn = [x for x in eventTrain]
    eventTrainOut = [x for x in eventTrain]
    timeTrainIn = [[(y - minTime) / (maxTime - minTime) for y in x] for x in timeTrain]
    timeTrainOut = [[(y - minTime) / (maxTime - minTime) for y in x] for x in timeTrain]


    eventTestIn = [x[:-1] for x in eventTest]
    eventTestOut = [x[1:] for x in eventTest]
    timeTestIn = [[(y - minTime) / (maxTime - minTime) for y in x] for x in timeTest]
    timeTestOut = [[(y - minTime) / (maxTime - minTime) for y in x] for x in timeTest]

    return eventTrainIn,timeTrainIn,eventTestIn,timeTestIn,maxTime,minTime




class EicuDataset_new:
    def __init__(self, eventTrainIn,timeTrainIn,seq_len):
        
        self.time_seqs=timeTrainIn
        self.event_seqs = eventTrainIn

        self.seq_len = seq_len
    


    def __getitem__(self, item):
        return self.time_seqs[item], self.event_seqs[item],self.seq_len

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
        gold_abs=[]

        def padding(in_seq,seq_len):
            l=len(in_seq)
            seq_new=np.copy(in_seq)
            if l >= seq_len:
                return seq_new[:seq_len]

            else:
                pad_seq=np.zeros([seq_len-l])
                seq_new=list(seq_new)
                seq_new.extend(list(pad_seq))
                return np.array(seq_new)


        for time, event, seq_len in batch:
            # print("event",event)
            # print("time",time)
            # print("seq_len",seq_len)
            time_new=np.copy(time)
            event_new=np.copy(event)

            time_new=list(time_new)
            time_new = np.array([time_new[0]] + time_new)
            time_new_diff = np.diff(time_new)

            

            tar_t=time_new_diff[-1]
            g_abs=time_new[-1]
            #print("times",time_new[:-1],tar_t)
            time_in=time_new_diff[:-1]
            time_in=padding(time_in,seq_len)
            time_ab=padding(np.copy(time)[:-1],seq_len)
            #print("time_in",time_in)
            times.append(time_in)
            times_tar.append(tar_t)
            times_abs.append(time_ab)
            gold_abs.append(g_abs)
            
            event_in=event_new[:-1]

            event_in=padding(event_in,seq_len)
            #print("event_in",event_in)
            events.append( event_in*0)
            events_tar.append(event_new[-1]*0)


            # times_tar.append(time[-1])
            # times.append(time[:-1])
            
        return torch.FloatTensor(times), torch.FloatTensor(times_tar),torch.LongTensor(events),torch.LongTensor(events_tar),torch.FloatTensor(times_abs),torch.FloatTensor(gold_abs)


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
