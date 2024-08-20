import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import T
from collections import deque, namedtuple
import random
import resnet

Transition = namedtuple('Transition',
                        ('state', 'label','q_value','action','next_state', 'reward', 'terminal'))

class DQN_tick(nn.Module):
    def __init__(self,observation:int,num_class:int):
        super(DQN_tick,self).__init__()
        self.linear1=nn.Linear(observation,1024)
        self.linear2=nn.Linear(1024,128)
        self.linear3=nn.Linear(128,num_class)
        self.relu=nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(7)

    def forward(self,x):
        x=self.avgpool(x)
        bs,feature_size=x.view((x.shape[0],x.shape[1])).shape
        x=self.linear1(x.squeeze())
        x=self.relu(x)
        x=self.linear2(x)
        x=self.relu(x)
        output=self.linear3(x)
        #output=output.reshpe(bs,4,-1)
        return output.cpu()
    
class DQN(nn.Module):
    def __init__(self,env,args,obs_ch=2048,number_actions=2,num_class=7):
        super(DQN,self).__init__()
        self.number_actions = number_actions
        self.obs_ch=obs_ch
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.pretrain_network=resnet.resnet50(pretrained=True).train(False).to(self.device)
        self.q_network=DQN_tick(obs_ch,num_class).to(self.device)
        self.target_network=DQN_tick(obs_ch,num_class).to(self.device)
        self.target_network.train(False)
        self.copy_to_target_network()
        for p in self.target_network.parameters():
            p.requires_grad = False
        self.optimiser =torch.optim.Adam(self.q_network.parameters(), lr=args.lr)
        self.env=env

    def copy_to_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def train_q_network(self,batch,current_reward,discount_factor,step):
        self.optimiser.zero_grad()
        current_state=torch.stack([batch[item].state[0][step] for item in range(len(batch))]).squeeze() #.to(self.device)

        next_state=torch.stack([batch[item].next_state[0][step] for item in range(len(batch))]).squeeze() #.to(self.device)
        terminal= torch.tensor([batch[item].terminal[0][0] for item in range(len(batch))]).type(torch.int)
        rewards=torch.clamp(torch.tensor([batch[item].reward[0][step][0]
                                          for item in range(len(batch))]).type(torch.float32),-4,4) #.to(self.device)
        #labels=torch.tensor([batch[item].label[0][step][0] for item in range(len(batch))])
        next_reward=0
        # current_reward=current_reward.to(self.device)
        #weighted rewards
        if step==0 or step==2:
            next_reward=current_reward+rewards-0.1
        elif step==1:
            next_reward=current_reward+rewards-0.1
        elif step==3:
            next_reward=current_reward+rewards-0.1

        y=self.target_network.forward(next_state)
        max_target_net = y.max(-1)[0]

        netwrok_prediction=self.q_network.forward(current_state)
        isNotOver=(torch.ones(*terminal.shape)-terminal) #.to(self.device)
        #Bellman equation
        batch_labels_tensor = rewards + isNotOver.squeeze() * \
                        (discount_factor * max_target_net.detach())
        #actions=torch.tensor([batch[item].action[0][step][0]
        #                                  for item in range(len(batch))])
        q_values=torch.stack([batch[item].q_value[0][step][0]
                                          for item in range(len(batch))])
        action=q_values.max(dim=-1)[1] #.to(self.device)
        y_pred=torch.gather(netwrok_prediction,-1,action.unsqueeze(-1))
        loss=torch.nn.SmoothL1Loss()(batch_labels_tensor.flatten(),y_pred.flatten())
        loss.backward()
        self.optimiser.step()
        return loss.item(), next_reward

class ReplayMemory(object):
    def __init__(self,max_size,state_shape,agents=1):
        self.max_size=int(max_size)
        self.state_shape=state_shape
        self.agents=agents
        #self.state=np.zeros((self.agents, self.max_size) + state_shape, dtype='uint8')
        #self.action=np.zeros((self.agents,self.max_size), dtype='int32')
        #self.reward = np.zeros((self.agents, self.max_size), dtype='float32')
        #self.isOver = np.zeros((self.agents, self.max_size), dtype='bool')

        #set up a plug in and pop out queue
        self.memory=deque([],maxlen=max_size)

    def append_transition(self,*args):
        self.memory.append(Transition(*zip(*args)))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
