import gym
import numpy as np
import torch
from gym import spaces

from cytokine.cytokine_model_tiny import Cytokine1


class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        steriod_max,
        observation_min,
        observation_max,
        step_size,
        eps_perb=0,
        behaviour=None,
        cytokine_model="standard",
        **kwargs,
    ):
        super(CustomEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects

        self.params = kwargs.copy()
        self.cytokine_model = cytokine_model
        self.behaviour = behaviour
        self.eps_perb = eps_perb
        if cytokine_model == "normalized":
            raise RuntimeError("Not Implemented")
            """
            self.cytokine = CytokineNormalized(**self.params)
            if behaviour is not None:
                self.cytokine.set_behaviour(behaviour)
            self.cytokine.perturb_parms(eps_perb)
            """
        else:
            self.cytokine = Cytokine1(**self.params)

        self.step_size = 1  # (1 hour)
        self.current_time = 0
        # Example when using discrete actions:
        self.action_space = spaces.Box(
            low=np.array([0]), high=np.array([steriod_max])
        )  # 3 grams per day ?

        # Example for using image as input:
        self.observation_space = spaces.Box(
            low=np.array(observation_min), high=np.array(observation_max)
        )  # e.g. TNF-alpha

        self.observation_list = []
        self.action_list = []

        self.drug_model = DE_GRU(n_features=2, n_hidden=20, n_output=1, n_layes=2)
        self.drug_model.init_hidden(batch_size=1)
        self.drug_model.load_model("./drug_efficacy_model.mdl")

        self.dynamic_type = DT_GRU(n_features=1, n_hidden=20, n_output=5, n_layes=2)
        self.dynamic_type.init_hidden(batch_size=1)
        self.dynamic_type.load_model("./behaviorual_detection_model.mdl")

    def call_dynamic_type(self, y):
        # y seqlen x 1
        self.dynamic_type.init_hidden(batch_size=1)
        # print("dynamic",y[np.newaxis,...])
        ttt = self.dynamic_type.forward(torch.FloatTensor(y[np.newaxis, ...]).cuda(1))
        out = ttt.detach().cpu().numpy().squeeze()
        out = out.reshape(-1, 5)  # seqlen x 5
        return out

    def call_drug_model(self, y):
        # y seqlen x 2
        # print("drug",y[np.newaxis,...])
        self.drug_model.init_hidden(batch_size=1)
        ttt = self.drug_model.forward(torch.FloatTensor(y[np.newaxis, ...]).cuda(1))
        out = ttt.detach().cpu().numpy().squeeze()
        out = out.reshape(-1, 1)  # seqlen x 1
        return out

    # step every hour but problem solved in smaller time-scale
    def step(self, action_in):
        # Execute one time step within the environment

        # action = action[0]
        # if (sum(self.action_list[-24:]) + action)>3:

        #    action = np.array([3 - sum(self.action_list[-24:])])
        #    if action<0:
        #        action = np.array([0])

        sact = sum(self.action_list[-24:])
        """
        if (sact+action[-1])>3:
            action0 = 3 - sact - action[-1]
            if action0<0:
                action0 = 0
        
            action = [action0]
        """

        action = [a for a in action_in]
        if action[-1] < 0:
            action[-1] = 0
        self.cytokine.current_values[-1] += action
        old_pro = self.cytokine.current_values[0]
        observation, old_observation = self.cytokine.step_to(
            self.current_time + self.step_size
        )
        self.current_time += self.step_size

        all_obs_list = observation.tolist()  #  + old_observation.tolist() + [sact]
        observation = np.array(all_obs_list)
        # print(observation)
        self.observation_list.append(observation[0])
        self.action_list.append(action[-1])

        # reward  we can calculate the sum of last 24 hours and observation[1] # cytokine? to form a loss
        # done   48 hours should be done=True

        s = np.mean(self.action_list[-24:])
        reward_sigmoid = 1 + s
        if observation[0] == 0:
            reward = 1
        reward = -observation[0] - s + 1

        # -np.exp(observation[0])#1 - (observation[0]/(0.1 + old_observation[0]))   #np.exp(-observation[0])         #reward_sigmoid/(0.1+observation[0])
        # (observation[0,1] - observation[0,0])/observation[0,1] #np.max(self.observation_list)

        # print(reward)

        done = self.current_time == 48
        info = {}
        info["action"] = action

        # import pdb
        # pdb.set_trace()
        # print(self.observation_list)

        dynamic_type, drug_eff = self.get_dynamic_drug_eff()

        # import pdb
        # pdb.set_trace()
        observation = np.array(observation.tolist() + [s] + dynamic_type + drug_eff)

        return observation, reward, done, info

    def get_dynamic_drug_eff(self):
        all_observation_seq = np.array(self.observation_list)[:, np.newaxis]
        temp = all_observation_seq.copy()
        normalizer = (
            np.cumsum(temp, axis=0) / np.arange(1, temp.shape[0] + 1, 1)[:, np.newaxis]
        )
        temp = temp / normalizer
        all_observation_seq = temp

        all_action_seq = np.array(self.action_list)[:, np.newaxis]

        dynamic_type = self.call_dynamic_type(all_observation_seq)
        dynamic_type = dynamic_type[-1].tolist()

        inp = np.concatenate((all_observation_seq, all_action_seq), axis=1)
        drug_eff = self.call_drug_model(inp)
        drug_eff = drug_eff[-1, :].tolist()

        return dynamic_type, drug_eff

    def reset(self, p0):
        # Reset the state of the environment to an initial state
        self.params["p0"] = p0
        if self.cytokine_model == "normalized":
            raise RuntimeError("Not implemented!")
            """
            self.cytokine = CytokineNormalized(**self.params)
            if self.behaviour is not None:
                self.cytokine.set_behaviour(self.behaviour)
            self.cytokine.perturb_parms(self.eps_perb)
            """
        else:
            self.cytokine = Cytokine1(**self.params)
        self.current_time = 0
        self.action_list = []
        self.observation_list = []
        # print(self.cytokine.current_values[np.newaxis,:].shape)
        # print(np.array([0])[np.newaxis,:].shape)
        all_obs_list = (
            self.cytokine.current_values.tolist()
        )  # + self.cytokine.current_values.tolist() + [0]
        observation = np.array(all_obs_list)

        # np.concatenate([self.cytokine.current_values[np.newaxis,:],
        #                             self.cytokine.current_values[np.newaxis,:],np.array([0])[np.newaxis,:]],axis=1)

        self.observation_list.append(observation[0])
        self.action_list.append(0)
        # import pdb
        # pdb.set_trace()
        dynamic_type, drug_eff = self.get_dynamic_drug_eff()
        observation = np.array(
            observation[:-1].tolist() + [0] + dynamic_type + drug_eff
        )
        return observation

    def render(self, mode="human", close=False):
        # Render the environment to the screen
        pass
