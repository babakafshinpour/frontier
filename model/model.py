import torch
import torch.autograd
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class CytokineDynamic(torch.nn.Module):
    def __init__(self,n_features,n_hidden, n_output, n_layes):
        super(CytokineDynamic, self).__init__()
        self.n_features = n_features
        self.n_hidden  = n_hidden# number of hidden states
        self.n_layers = 2 # number of LSTM layers (stacked)
        
        #input_size – The number of expected features in the input x
        #hidden_size – The number of features in the hidden state h
        #num_layers – Number of recurrent layers. E.g., setting num_layers=2 would mean stacking two RNNs together to form a    stacked RNN, with the second RNN taking in outputs of the first RNN and computing the final results. Default: 1
        #nonlinearity – The non-linearity to use. Can be either 'tanh' or 'relu'. Default: 'tanh'
        
        self.l_gru    = torch.nn.GRU(self.n_features,self.n_hidden,self.n_layers,batch_first = True)
        self.l_linear = torch.nn.Linear(self.n_hidden,n_output)
        
        
    def init_hidden(self, batch_size:int):

        # Even with batch_first = True this remains same as docs

        hidden_state = torch.zeros(self.n_layers,batch_size,self.n_hidden)
        self.hidden = hidden_state#, cell_state)


    def forward(self, x):        
        
        out_list = []
        for i in range(len(x)):
            rnn_out, hidden = self.l_gru(x[i].unsqueeze(0),self.hidden)
            out_temp = self.l_linear(rnn_out)
            out_list.append(out_temp[:,-1:,:])
        
        out = torch.cat(out_list,dim=0)
        return  out
    
    def forward_all_actions(self, x):
        out_list = []
        for i in range(len(x)):
            rnn_out, hidden = self.l_gru(x[i].unsqueeze(0).cuda(1),self.hidden)
            out_temp = self.l_linear(rnn_out)
            out_list.append(out_temp[-1,:,:])
            
        return  out_list

class Critic(CytokineDynamic):
    def __init__(self, input_size, hidden_size, output_size):
        super(Critic, self).__init__(n_features=input_size,n_hidden=hidden_size,
                                     n_output=output_size, n_layes=1)
        self.init_hidden(1)
    def forward(self, state, action):
        """
        Params state and actions are torch tensors
        """
        out_list = []
        for i in range(len(state)):
            x = torch.cat([state[i].cuda(1), action[i].cuda(1)], 1)
            rnn_out, hidden = self.l_gru(x.unsqueeze(0),self.hidden)
            out_temp = self.l_linear(rnn_out)
            out_list.append(out_temp[:,-1:,:])
        
        out = torch.cat(out_list,dim=0)[:,:,-1]
        return out
        
class Actor(CytokineDynamic):
    def __init__(self, input_size, hidden_size, output_size, learning_rate = 3e-4):
        super(Actor, self).__init__(n_features=input_size,n_hidden=hidden_size,
                                     n_output=output_size, n_layes=1)
        self.init_hidden(1)
        self.sigmoid = nn.Sigmoid().cuda(1)
    
    def forward(self, x, bypass_gru = False): 
        #print(x)
        #import pdb
        #pdb.set_trace()
        y = super(Actor, self).forward(x)
        return self.sigmoid(y)
        

class Critic_NN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Critic_NN, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size).cuda(1)
        self.linear2 = nn.Linear(hidden_size, hidden_size).cuda(1)
        self.linear3 = nn.Linear(hidden_size, hidden_size).cuda(1)
        self.linear4 = nn.Linear(hidden_size, hidden_size).cuda(1)
        self.linear5 = nn.Linear(hidden_size, output_size).cuda(1)
        
    def forward(self, state, action):
        """
        Params state and actions are torch tensors
        """
        
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        x = self.linear5(x)
        
        
        return x

class Actor_NN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Actor_NN, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size).cuda(1)
        self.linear2 = nn.Linear(hidden_size, hidden_size).cuda(1)
        self.linear3 = nn.Linear(hidden_size, hidden_size).cuda(1)
        self.linear4 = nn.Linear(hidden_size, hidden_size).cuda(1)
        self.linear5 = nn.Linear(hidden_size, output_size).cuda(1)
        self.sigmoid = nn.Sigmoid().cuda(1)
        
    def forward(self, state):
        """
        Param state is a torch tensor
        """
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        x = self.linear5(x)
        x = self.sigmoid(x)
        return x