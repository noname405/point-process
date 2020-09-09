import sys
sys.path.append('/Users/aishwaryaya/dir_1/eicu_data/data/tf_rmtpp/src/')
import keys_vocab
import numpy as np
from collections import defaultdict
import itertools
import os
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import re




class patientinfo(object):
    def __init__(self,  admit_id, timestamp):
        #self.patient_id = patient_id
        self.admit_id = admit_id
        self.timestamp = timestamp
        self.infusion = []
        self.labs = {}
        self.med = []
        self.nursechart={}
        self.nursecare=[]
        self.nurseassess=[]
        self.gcs=[]
        self.decreased_gcs=[]
        self.less8_gcs=[]
        


#feature_keys,vocab_sizes=keys_vocab.get_keys_vocab()
# max_num_codes=10
# min_num_codes=1
# BPTT=50

def process_data_generate_onlyevents(train_data,min_num_codes,max_num_codes,step_size,feature_keys,div60=True):
    train_event_list={}
    train_time_list={}

    out_event_list=[]
    out_time_list=[]
    train_mask_list=[]
    
    in_time_list=[]
    in_event_list=[]
    
    for key in feature_keys:
            train_event_list[key]=[]
            train_time_list[key]=[]
    
#     for f in files:
#         file_gcs=os.path.join(event_train_path,f)
#         with open(file_gcs,"rb") as pklfile:
#             train_data=pickle.load(pklfile)            
        
#     file_gcs=os.path.join(event_train_path,f)
#     with open(file_gcs,"rb") as pklfile:
#         train_data=pickle.load(pklfile)           
    
    for k,pat_data in train_data.items():
        train_event={}
        train_time={}
        med_count=0
        infusion_count=0
        nurseassess_count=0
        nursecare_count=0
        for key in feature_keys:
            train_event[key]=np.zeros([step_size+1,max_num_codes],dtype=np.int32)
        for key in feature_keys:
            train_time[key]=np.zeros([step_size+1,max_num_codes],dtype=np.int32)

        out_event=np.zeros(step_size+1,dtype=np.int32)
        out_time=np.zeros(step_size+1,dtype=np.float32)

        train_mask=np.zeros(step_size+1,dtype=np.int32)
        t=0
        for tim,eve in pat_data.items():
            out_e=0
            out_t=0.
            if len(eve.decreased_gcs) >0:

                if len(eve.decreased_gcs) >1:
                        print("eve.decreased_gcs",eve.decreased_gcs)
                out_e=1
                out_t=eve.decreased_gcs[0]
            if len(eve.less8_gcs)>0:
                if len(eve.less8_gcs) >1:
                    print("eve.less8_gcs",eve.less8_gcs)
                out_e=1
                out_t=eve.less8_gcs[0]
                  
            if (t < step_size+1 and tim >0 and out_e==1) or tim==0:
                #print("out_e",out_e)  
                train_mask[t]=1
                out_event[t]=out_e
                out_time[t]=out_t
                #print("out_e,out_t",out_e,out_t)
                if len(eve.med) >min_num_codes:
                    #print(eve.med)
                    if len(eve.med) > max_num_codes:
                        eve_med=eve.med[:max_num_codes]
                    else:
                        eve_med=eve.med

                    event_list=[te[0] for te in eve_med]
                    time_list=[te[1] for te in eve_med]
                    train_event['med'][t][:len(eve_med)]= event_list
                    train_time['med'][t][:len(eve_med)]= time_list

                if len(eve.infusion) >min_num_codes:
                    if len(eve.infusion) > max_num_codes:
                        eve_infusion=eve.infusion[:max_num_codes]
                    else:
                        eve_infusion=eve.infusion   

                    event_list=[te[0] for te in eve_infusion]
                    time_list=[te[1] for te in eve_infusion]
                    train_event['infusion'][t][:len(eve_infusion)]=event_list
                    train_time['infusion'][t][:len(eve_infusion)]=time_list

                if len(eve.nurseassess) >min_num_codes:
                    if len(eve.nurseassess) > max_num_codes:
                        eve_nurseassess=eve.nurseassess[:max_num_codes]
                    else:
                        eve_nurseassess=eve.nurseassess

                    event_list=[te[0] for te in eve_nurseassess]
                    time_list=[te[1] for te in eve_nurseassess]
                    train_event['nurseassess'][t][:len(eve_nurseassess)]=event_list
                    train_time['nurseassess'][t][:len(eve_nurseassess)]=time_list

                if len(eve.nursecare) >min_num_codes:
                    if len(eve.nursecare) > max_num_codes:
                        eve_nursecare=eve.nursecare[:max_num_codes]
                    else:
                        eve_nursecare=eve.nursecare
                    event_list=[te[0] for te in eve_nursecare]
                    time_list=[te[1] for te in eve_nursecare]
                    train_event['nursecare'][t][:len(eve_nursecare)]=event_list
                    train_time['nursecare'][t][:len(eve_nursecare)]=time_list

                for lk,lv in eve.labs.items():
                    if len(lv)>min_num_codes:
                        if len(lv) > max_num_codes:
                            lv=lv[:max_num_codes]

                        lk=re.sub('\W+','', lk )                    
                        event_list=[te[0] for te in lv]
                        time_list=[te[1] for te in lv]
                        train_event[lk][t][:len(lv)]=event_list
                        train_time[lk][t][:len(lv)]=time_list

                for lk,lv in eve.nursechart.items():
                    if len(lv)>min_num_codes:
                        if len(lv) > max_num_codes:
                            lv=lv[:max_num_codes]

                        lk=re.sub('\W+','', lk )                    
                        event_list=[te[0] for te in lv]
                        time_list=[te[1] for te in lv]
                        train_event[lk][t][:len(lv)]=event_list
                        train_time[lk][t][:len(lv)]=time_list

                
                if div60:
                    out_time[t]=out_time[t]/60.0
                t=t+1
                #print("out_event",out_event)

        if np.count_nonzero(out_event)<=1:
                #print("zero_out_event",out_event)
                continue
        
        # in_time=np.copy(out_time[:-1])
        # #print("out_time1",out_time[:-1])
        # for ii,ti in enumerate(in_time):
        #     if ii>0 and ti==0:
        #         in_time[ii]=in_time[ii-1]


        # rev_out_time= np.copy(out_time[1:])[::-1]
        # ri = [ii for ii,it in enumerate(rev_out_time) if it>0 ]
        # output_time = np.ones(len(rev_out_time),np.float32)* 1000.
        # if len(ri)>0:
        #     ri=ri[0]
        #     for ii,ti in enumerate(rev_out_time):
        #         if ii >= ri:
        #             if ti > 0:
        #                 d=ti

        #             output_time[ii]=d


        #     output_time=output_time[::-1]

        for key in feature_keys:
            train_event_list[key].append(train_event[key][:-1])
            train_time_list[key].append(train_time[key][:-1])

        train_mask_list.append(train_mask[:-1])

        out_event_list.append(out_event[1:])
        out_time_list.append(out_time[1:])
        #out_time_list.append(output_time)

        in_event_list.append(out_event)
        in_time_list.append(out_time)
        #in_time_list.append(in_time)

        # in_event_list.append(out_event[:-1])
        # #in_time_list.append(out_time[:-1])
        # in_time_list.append(out_time)
 
        
        
    return train_event_list,train_time_list,out_event_list,out_time_list,train_mask_list, in_event_list,in_time_list
 
def read_data_new_train(event_train_path):
    
#     (train_event_list,train_time_list,train_out_event_list,train_out_time_list,train_mask_list,
#             train_in_event_list,train_in_time_list)=process_data_generate(train_files, event_train_path,min_num_codes,
#                                                                                max_num_codes,step_size,feature_keys,div60=div60)
    files=[f for f in os.listdir(event_train_path) if 'patient_traingcs' in f]
    fileno=len(files)
    #fileno=1
    #print("fileno",fileno)
    #np.random.shuffle(files)
    train_fileno= int(fileno*0.8)
    val_fileno=int((fileno-train_fileno)/2)
    
    train_files= files[:train_fileno]
    
    val_files= files[train_fileno:train_fileno+val_fileno]
    
    test_files=files[-val_fileno:]

    train_files=[train_files[0]]
    print("Number of train files: ",len(train_files))
    print("train_files",train_files)
    for f in train_files:
        file_gcs=os.path.join(event_train_path,f)
        with open(file_gcs,"rb") as pklfile:
            train_data=pickle.load(pklfile)  
            
        yield train_data


# training_data = tf_rmtpp.utils_new.read_data_new_train(event_train_path)
# train_data = next(training_data)

# (train_event_list,train_time_list,train_out_event_list,train_out_time_list,
#              train_mask_list, train_in_event_list,
#              train_in_time_list)=process_data_generate_onlyevents(train_data, min_num_codes=self.min_num_codes,
#                                                    max_num_codes=self.max_num_codes,
#                                                    step_size=self.BPTT, feature_keys=self.feature_keys)
#             