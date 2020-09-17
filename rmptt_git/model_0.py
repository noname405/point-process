import torch
from torch import nn
from torch.optim import Adam
import numpy as np
from optimization_0 import BertAdam
from scipy.integrate import quad


class Net(nn.Module):
    def __init__(self, config, lossweight):
        super(Net, self).__init__()
        self.config = config
        self.n_class = config.event_class
        config.emb_dim=0
        #self.embedding = nn.Embedding(num_embeddings=config.event_class, embedding_dim=config.emb_dim)
        #self.emb_drop = nn.Dropout(p=config.dropout)
        self.lstm = nn.LSTM(input_size=config.emb_dim + 1,
                            hidden_size=config.hid_dim,
                            batch_first=True,
                            bidirectional=False)
        self.mlp = nn.Linear(in_features=config.hid_dim, out_features=config.mlp_dim)
        self.mlp_drop = nn.Dropout(p=config.dropout)
        self.event_linear = nn.Linear(in_features=config.mlp_dim, out_features=config.event_class)
        self.time_linear = nn.Linear(in_features=config.mlp_dim, out_features=1)
        self.Vt= nn.Linear(in_features=config.hid_dim, out_features=1)
        self.soft=torch.nn.Softplus()
        self.set_criterion(lossweight)

    def set_optimizer(self, total_step, use_bert=True):
        if use_bert:
            self.optimizer = BertAdam(params=self.parameters(),
                                      lr=self.config.lr,
                                      warmup=0.1,
                                      t_total=total_step)
        else:
            self.optimizer = Adam(self.parameters(), lr=self.config.lr)

    def set_criterion(self, weight):
        self.event_criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(weight))
        print("------time_criterion",self.config.model)
        if self.config.model == 'rmtpp':
            self.intensity_w = nn.Parameter(torch.tensor(0.1, dtype=torch.float, device='cpu'))
            self.intensity_b = nn.Parameter(torch.tensor(0.1, dtype=torch.float, device='cpu'))
            
            self.time_criterion = self.RMTPPLoss_1
        else:
            self.intensity_w = nn.Parameter(torch.tensor(0.1, dtype=torch.float, device='cpu'))
            self.time_criterion = self.poisson

    def RMTPPLoss(self, pred, gold):
        loss = torch.mean(pred + self.intensity_w * gold + self.intensity_b +
                          (torch.exp(pred + self.intensity_b) -
                           torch.exp(pred + self.intensity_w * gold + self.intensity_b)) / self.intensity_w)
        return -1 * loss

    def RMTPPLoss_1(self, pred, gold,state_Vt,wt_soft_plus):
        
        log_lambda_ = (state_Vt + (-gold * wt_soft_plus) + self.intensity_b)
        lambda_ = torch.exp(torch.min(torch.tensor(50.0), log_lambda_))

        log_f_star = (log_lambda_ - (1.0 / wt_soft_plus) * torch.exp(torch.min(torch.tensor(50.0), state_Vt + self.intensity_b)) +
                                          (1.0 / wt_soft_plus) * lambda_)

        loss = torch.mean(log_f_star)
        return -1 * loss


    def poisson(self, pred, gold,state_Vt,wt_soft_plus):
        
        log_lambda_ = -wt_soft_plus
        
        lambda_ = torch.exp(torch.min(torch.tensor(50.0), log_lambda_))

        log_f_star = (log_lambda_ - lambda_ * gold)

        loss = torch.mean(log_f_star)
        return -1 * loss

    def forward(self, input_time, input_events):
        event_embedding = self.embedding(input_events)
        event_embedding = self.emb_drop(event_embedding)
        lstm_input = torch.cat((event_embedding, input_time.unsqueeze(-1)), dim=-1)
        hidden_state, _ = self.lstm(lstm_input)

        # hidden_state = torch.cat((hidden_state, input_time.unsqueeze(-1)), dim=-1)
        mlp_output = torch.tanh(self.mlp(hidden_state[:, -1, :]))
        mlp_output = self.mlp_drop(mlp_output)
        event_logits = self.event_linear(mlp_output)
        time_logits = self.time_linear(mlp_output)
        return time_logits, event_logits

    def forward_1(self, input_time, input_events):
        # event_embedding = self.embedding(input_events)
        # event_embedding = self.emb_drop(event_embedding)
        #lstm_input = torch.cat((event_embedding, input_time.unsqueeze(-1)), dim=-1)
        lstm_input= input_time.unsqueeze(-1)
        hidden_state, _ = self.lstm(lstm_input)
        
        state_Vt=self.Vt(hidden_state[:, -1, :])
        wt_soft_plus = self.soft(self.intensity_w )
        # hidden_state = torch.cat((hidden_state, input_time.unsqueeze(-1)), dim=-1)
        mlp_output = torch.tanh(self.mlp(hidden_state[:, -1, :]))
        mlp_output = self.mlp_drop(mlp_output)
        event_logits = self.event_linear(mlp_output)
        time_logits = self.time_linear(mlp_output)
        return time_logits, event_logits,state_Vt,wt_soft_plus

    def dispatch(self, tensors):
        for i in range(len(tensors)):
            #tensors[i] = tensors[i].cuda().contiguous()
            tensors[i] = tensors[i].contiguous()
        return tensors

    def train_batch(self, batch):
        time_tensor, event_tensor = batch
        time_input, time_target = self.dispatch([time_tensor[:, :-1], time_tensor[:, -1]])
        event_input, event_target = self.dispatch([event_tensor[:, :-1], event_tensor[:, -1]])
        print("time_input",time_input.size(),time_target.size(),event_input.size(),event_target.size())
        time_logits, event_logits = self.forward(time_input, event_input)
        loss1 = self.time_criterion(time_logits.view(-1), time_target.view(-1))
        print("event_target.view(-1)",event_logits.view(-1, self.n_class),event_target.view(-1))
        loss2 = self.event_criterion(event_logits.view(-1, self.n_class), event_target.view(-1))
        loss = self.config.alpha * loss1 + loss2
        loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss1.item(), loss2.item(), loss.item()

    def train_batch_1(self, batch):
        # time_tensor, event_tensor = batch
        # time_input, time_target = self.dispatch([time_tensor[:, :-1], time_tensor[:, -1]])
        # event_input, event_target = self.dispatch([event_tensor[:, :-1], event_tensor[:, -1]])

        time_input, time_target,event_input, event_target=batch
        #print("time_input",time_input.size(),time_target.size(),event_input.size(),event_target.size())
        time_logits, event_logits = self.forward(time_input, event_input)
        loss1 = self.time_criterion(time_logits.view(-1), time_target.view(-1))
        #print("event_target.view(-1)",event_logits.view(-1, self.n_class),event_target.view(-1))
        loss2 = self.event_criterion(event_logits.view(-1, self.n_class), event_target.view(-1))
        loss = self.config.alpha * loss1 + loss2
        loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss1.item(), loss2.item(), loss.item()


    def train_batch_2(self, batch):
        # time_tensor, event_tensor = batch
        # time_input, time_target = self.dispatch([time_tensor[:, :-1], time_tensor[:, -1]])
        # event_input, event_target = self.dispatch([event_tensor[:, :-1], event_tensor[:, -1]])

        time_input, time_target,event_input, event_target,time_input_abs,gold_abs=batch
        #print("time_input",time_input.size(),time_target.size(),event_input.size(),event_target.size())
        
        time_logits, event_logits,state_Vt,wt_soft_plus = self.forward_1(time_input, event_input)
        loss1 = self.time_criterion(time_logits.view(-1), time_target.view(-1),state_Vt,wt_soft_plus)
        #print("event_target.view(-1)",event_logits.view(-1, self.n_class),event_target.view(-1))
        loss2 = self.event_criterion(event_logits.view(-1, self.n_class), event_target.view(-1))
        loss = loss1 #+ loss2
        loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss1.item(), loss2.item(), loss.item()

    def predict(self, batch):
        time_tensor, event_tensor = batch
        time_input, time_target = self.dispatch([time_tensor[:, :-1], time_tensor[:, -1]])
        event_input, event_target = self.dispatch([event_tensor[:, :-1], event_tensor[:, -1]])
        time_logits, event_logits = self.forward(time_input, event_input)
        event_pred = np.argmax(event_logits.detach().cpu().numpy(), axis=-1)
        time_pred = time_logits.detach().cpu().numpy()
        return time_pred, event_pred


    def predict_1(self, batch):
        # time_tensor, event_tensor = batch
        # time_input, time_target = self.dispatch([time_tensor[:, :-1], time_tensor[:, -1]])
        # event_input, event_target = self.dispatch([event_tensor[:, :-1], event_tensor[:, -1]])

        time_input, time_target,event_input, event_target,time_input_abs=batch
        time_logits, event_logits = self.forward(time_input, event_input)
        event_pred = np.argmax(event_logits.detach().cpu().numpy(), axis=-1)
        time_pred = time_logits.detach().cpu().numpy()
        return time_pred, event_pred


    def predict_2(self, batch):
        # time_tensor, event_tensor = batch
        # time_input, time_target = self.dispatch([time_tensor[:, :-1], time_tensor[:, -1]])
        # event_input, event_target = self.dispatch([event_tensor[:, :-1], event_tensor[:, -1]])

        def quad_func(t, c, w):
            """This is the t * f(t) function calculating the mean time to next event,
            given c, w."""
            return c * t * np.exp(-w * t + (c / w) * (np.exp(-w * t) - 1))


        time_input, time_target,event_input, event_target,time_input_abs,gold_abs=batch
        time_logits, event_logits,state_Vt,wt_soft_plus = self.forward_1(time_input, event_input)
        bt=self.intensity_b.detach().cpu().numpy()
        hvt=state_Vt.detach().cpu().numpy()
        wt=wt_soft_plus.detach().cpu().numpy()
        time_in_seq=time_input_abs.detach().cpu().numpy()
        
        
        preds_i = []
        C = np.exp(hvt + bt).reshape(-1)

        for c_, t_last in zip(C, time_in_seq[:, -1]):
            args = (c_, wt)
            val, _err = quad(quad_func, 0, np.inf, args=args)
            #preds_i.append(t_last + val)
            preds_i.append(val)

        
        all_time_preds=np.asarray(preds_i).T
       
    
        event_pred = np.argmax(event_logits.detach().cpu().numpy(), axis=-1)
        time_pred = time_logits.detach().cpu().numpy()
        return all_time_preds, event_pred


    def predict_3(self, batch):
        # time_tensor, event_tensor = batch
        # time_input, time_target = self.dispatch([time_tensor[:, :-1], time_tensor[:, -1]])
        # event_input, event_target = self.dispatch([event_tensor[:, :-1], event_tensor[:, -1]])

        def quad_func(t, c,w):
            """This is the t * f(t) function calculating the mean time to next event,
            given c, w."""
            return  t * np.exp(-w  - (np.exp(-w )* (t-c)))

        time_input, time_target,event_input, event_target,time_input_abs,gold_abs=batch
        time_logits, event_logits,state_Vt,wt_soft_plus = self.forward_1(time_input, event_input)
        # bt=self.intensity_b.detach().cpu().numpy()
        # hvt=state_Vt.detach().cpu().numpy()
        wt=wt_soft_plus.detach().cpu().numpy()
        time_in_seq=time_input.detach().cpu().numpy()
        time_in_abs=time_input_abs.detach().cpu().numpy()
        
        
        preds_i = []
        #C = np.exp(hvt + bt).reshape(-1)
        # print("time_input_abs",time_in_abs.shape,time_in_abs)
        # print("tnonzero",[(np.nonzero(t)) for t in time_in_abs])
        t_last_list=[t[np.max(np.nonzero(t))] for t in time_in_abs]

        for t_last in t_last_list:
            #t_last=0
            args = (t_last, wt)
            val, _err = quad(quad_func, t_last, np.inf, args=args)
            #preds_i.append(t_last + val)
            preds_i.append(val)

        
        all_time_preds=np.asarray(preds_i).T
       
    
        event_pred = np.argmax(event_logits.detach().cpu().numpy(), axis=-1)
        time_pred = time_logits.detach().cpu().numpy()
        return all_time_preds, event_pred, np.exp(wt)
