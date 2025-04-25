import environment
from DQN_CNN import DQN
import random
import torch
path = "Data/parameters1"
class DQN_Agent:
    def __init__(self,parameters_path =None,train =False,env =None):
        self.parameters_path=parameters_path
        self.train=train
        self.env =env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.DQN : DQN = DQN(device=self.device)
        

    def setTrainMode (self):
          if self.train:
              self.DQN.train()
          else:
              self.DQN.eval()

    def getAction(self, state_cnn, epoch = 0, events= None, train =True): 
        
        states_cnn, actions = self.env.state_actions(state_cnn)
        
        if train:
            epsilon = self.DQN.epsilon_greedy(epoch)
            rnd = random.random()
            if rnd < epsilon:
                idx = random.randrange(len(actions))
                action = actions[idx]
                state_cnn = states_cnn[idx]
                return action, state_cnn
        
        with torch.no_grad():
            Q_values = self.DQN(states_cnn)
        max_index = torch.argmax(Q_values)
        return actions[max_index], states_cnn[max_index]
        
    
    def get_Q_values(self,state):
        Q_values = self.DQN(state)
        return Q_values

    def get_Action_Values (self, next_states):
        Q_values_lst = []
        for state_cnn in next_states:
            states_cnn, _ = self.env.state_actions(state_cnn)
            with torch.no_grad():
                Q_values = self.DQN(states_cnn)
            Q_values_lst.append(torch.max(Q_values))
        
        res = torch.vstack(Q_values_lst)
        return res

    
    def loadModel (self, file):
        self.model = torch.load(file)
    
    def save_param (self, path):
        self.DQN.save_params(path)

    def load_params (self, path):
        self.DQN.load_params(path)
