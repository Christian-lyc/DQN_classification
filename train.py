import numpy as np
from DQN_tick import DQN
from DQN_tick import ReplayMemory
from tick_env import TickEnv
import torch
import torch.nn.functional as F
from test import Test
import argparse
import os
from dataset import Dataset
import math

class Training(object):
    def __init__(self,dataset,env,image_size,args,agents=1,action_step=4,eps=1,min_eps=0.1,number_actions=2,number_action_seq=4,
                num_class=7,gamma=0.9):
        self.env=env
        self.image_size=(image_size,image_size)
        self.buffer_size=args.memory_size
        self.batch_size=args.batch_size
        self.gamma=gamma
        self.steps_per_episode=args.steps_per_episode
        self.eps=eps
        self.min_eps=min_eps
        self.number_actions=number_actions
        self.number_action_seq=number_action_seq
        self.max_episodes=args.epochs
        self.delta=args.delta
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model=DQN(env,args,num_class=num_class)
        self.buffer=ReplayMemory(int(args.memory_size),self.image_size)
        self.action_step=action_step
        self.agents=agents
        self.num_class=num_class
        self.test=Test(env,image_size,self.model,args,agents=1,max_steps=len(dataset.test_loader))
        self.task=args.task

    def train(self):
        self.init_memory(self.buffer_size,self.steps_per_episode)

        rewards_epochs = []
        best_prec1=0
        for episode in range(self.max_episodes):
            obs=self.env.reset(self.task)
            
            losses=[]

            for step in range(self.steps_per_episode):
                obs_set = []
                action_set = []
                reward_set = []
                batch=[]
                next_obs_set = []
                label_set = []
                q_value_set=[]
                terminal_set = []
                rewards=torch.zeros(self.batch_size)
                extract_obs = self.model.pretrain_network(obs.to(self.device))
                initial_reward=torch.zeros(self.batch_size)
                for i in range(self.action_step):

                    actions,q_values=self.get_next_actions(extract_obs)
                    next_obs,reward,terminal, info=self.env.step(np.copy(actions),q_values,i,obs)
                    with torch.no_grad():
                        next_obs=F.interpolate(next_obs,size=[obs.shape[2],obs.shape[3]])
                    next_extract_obs = self.model.pretrain_network(next_obs.to(self.device))
                    obs_set.append(extract_obs.clone().detach())
                    action_set.append(actions)
                    label_set.append(env._label.clone().detach())
                    reward_set.append(reward)
                    q_value_set.append(q_values.clone().detach())
                    next_obs_set.append(next_extract_obs.clone().detach())
                    terminal_set.append(terminal)
                    extract_obs=next_extract_obs
                    obs = next_obs
                    if i==0:
                        batch=self.buffer.sample(self.batch_size)
                    loss,rewards=self.model.train_q_network(batch,initial_reward,self.gamma,i)

                    losses.append(loss)
                    initial_reward=rewards
                if terminal[0]:
                    break
                self.buffer.append_transition((obs_set, label_set,q_value_set,action_set,next_obs_set, reward_set, terminal_set))
                rewards_epochs.append(torch.mean(rewards).cpu())
                if step%10==0:
                    print('epochs:{0} \t' 'batch_reward:{1} \t'.format(episode,torch.mean(rewards)))
                

            self.model.copy_to_target_network()
#            self.eps = self.min_eps + (self.eps - self.min_eps) * math.exp(-(episode / self.max_episodes))
            self.eps=max(self.min_eps,self.eps-self.delta)
            if episode>4800:
                self.task ='test'
                acc,pre,label=self.validation_epoch(episode,self.task)
                best=acc>best_prec1
                best_prec1 = max(acc, best_prec1)
                if best:
                    np.save('pred.npy', np.array(pre))
                    np.save('labels.npy', np.array(label))
                    net_state_dict=self.model.state_dict()
                    torch.save({
                        'epoch': episode,
                        'net_state_dict': net_state_dict},
                        os.path.join(save_dir, 'best.pt'))
                self.task = 'train'

            
        np.save('reward.npy',np.array(rewards_epochs))

    def init_memory(self,buffer_size, steps_per_episode=20):
        while len(self.buffer) < buffer_size:
            obs = self.env.reset(self.task)
            
            for _ in range(steps_per_episode):
                obs_set=[]
                action_set=[]
                reward_set=[]
                next_obs_set=[]
                label_set=[]
                q_value_set=[]
                terminal_set = []
                extract_obs = self.model.pretrain_network(obs.to(self.device))
                for i in range(self.action_step):

                    actions, q_values = self.get_next_actions(extract_obs)
                    next_obs, reward, terminal, info = self.env.step(actions, q_values,i,obs)
                    with torch.no_grad():
                        next_obs=F.interpolate(next_obs,size=[obs.shape[2],obs.shape[3]])
                        next_extract_obs = self.model.pretrain_network(next_obs.to(self.device))
                    obs_set.append(extract_obs.clone().detach())
                    next_obs_set.append(next_extract_obs.clone().detach())
                    action_set.append(actions)
                    label_set.append(env._label.clone().detach())
                    reward_set.append(reward)
                    q_value_set.append(q_values.clone().detach())
                    terminal_set.append(terminal)
                    extract_obs = next_extract_obs
                    obs = next_obs
                if terminal[0]:
                    break
                self.buffer.append_transition((obs_set, label_set,q_value_set,action_set, next_obs_set,reward_set, terminal_set))
                


    def get_next_actions(self, obs):
        if np.random.random() < self.eps:#
            #
            q_values = torch.rand(self.agents, self.num_class)*torch.sqrt(torch.tensor(0.01))
#            output=F.softmax(q_values, -1)
            #actions = np.random.randint(self.number_actions, size=self.agents)
            class_tick = q_values.max(dim=-1)[1]
            actions = ~(class_tick[0] == env._label[0])
            obs=obs.cpu()
        else:
            actions, q_values = self.get_greedy_actions(
                    obs)
        return int(actions),q_values

    def get_greedy_actions(self, obs):
        #target network
        q_values=self.model.target_network.forward(obs).detach()
        #output = F.softmax(q_values, -1)
        class_tick=q_values.max(dim=-1)[1]
        actions=~(class_tick==env._label[0])

        return actions,q_values.unsqueeze(0).to('cpu')

    def validation_epoch(self,episode,task):
        self.model.q_network.train(False)
        reward,acc,pre,label=self.test.test(task)
        print('episode:{0} \t' 'batch_reward:{1} \t' 'Acc:{2}'.format(episode,reward[0],acc))
        self.model.q_network.train(True)
        return acc,pre,label


parser = argparse.ArgumentParser(description='PyTorch Tick Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-j', '--workers', default=40, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=10000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument(
        '--memory_size',
        help="""Number of transitions stored in exp replay buffer.
                If too much is allocated training may abruptly stop.""",
        default=1e4, type=int) #1e5
parser.add_argument(
        '--discount',
        help='Discount factor used in the Bellman equation',
        default=0.9, type=float)
parser.add_argument(
        '--steps_per_episode', help='Maximum steps per episode',
        default=20, type=int)
parser.add_argument(
        '--delta',
        help="""Amount to decreases epsilon each episode,
                for the epsilon-greedy policy""",
        default=1e-4, type=float)
parser.add_argument(
        '--task',choices=['train','test'],default='train')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
torch.cuda.set_device('cuda:0')
device = torch.device("cuda:0")
traindir = os.path.join(args.data, 'train')
testdir=os.path.join(args.data, 'val')
dataset=Dataset(traindir,testdir,args)
env=TickEnv(dataset,args,image_dim=[224,224])
save_dir=""
train=Training(dataset,env,224,args).train()







