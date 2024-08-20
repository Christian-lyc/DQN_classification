
import torch
import torch.nn.functional as F
import numpy as np
from collections import Counter
class Test(object):
    def __init__(self,env,image_size,model,args,agents,max_steps,action_step=4):
        self.env=env
        self.model=model
        self.agents=agents
        self.image_size = (image_size, image_size)
        self.action_step=action_step
        self.max_steps=max_steps
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size=args.batch_size

    def test(self,task):


        steps=0
        terminal = [False]
        tick_classes=[]
        labels=[]
        class_tick=0
        rewards=0
        q_value=[]
        while steps < self.max_steps:
            obs = self.env.reset(task)

            extract_obs = self.model.pretrain_network(obs.to(self.device))
            for step in range(50):
                for i in range(self.action_step):
                    q_values=self.model.q_network.forward(extract_obs)
                    #output = F.softmax(q_values, -1)
                    class_tick = q_values.max(dim=-1)[1]
                    actions = ~(class_tick.to('cpu') == self.env._label[0])
                    next_obs, reward, terminal, info = self.env.step(np.copy(actions), q_values, i, obs)
                    next_obs = F.interpolate(next_obs, size=[obs.shape[2], obs.shape[3]])
                    next_extract_obs = self.model.pretrain_network(next_obs.to(self.device))
                    extract_obs = next_extract_obs
                    obs=next_obs
                    rewards+=reward
                if terminal[0]:
                    break
            tick_classes.append(class_tick.to('cpu').numpy())
            labels.append(self.env._label[0].numpy())
            steps += 1
            q_value.append(q_values.detach().numpy())
        comparison = [x == y for x, y in zip(tick_classes, labels)]
        count_stats = Counter(comparison)
        acc=count_stats[True]/len(comparison)
        if acc>1:
            np.save('pred.npy', np.array(tick_classes))
            np.save('labels.npy', np.array(labels))
        return rewards/self.max_steps,acc, q_value,labels
