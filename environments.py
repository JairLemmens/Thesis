import numpy as np
import cv2 as cv
import torch

class Canvas_env:
    def __init__(self,imsize=32,max_radius=10,criterium=None,max_steps=200):
        self.criterium = criterium
        self.imsize = imsize
        self.max_radius = max_radius
        self.reset()
        self.max_steps = max_steps

    def reset(self):
        self.canvas = np.zeros([self.imsize,self.imsize])
        self.state = torch.zeros(self.canvas.shape,dtype=torch.float32)
        self.prev_score = 0
        self.max_score = 0
        self.steps = 0
        return(torch.tensor(self.canvas,dtype=torch.float32),torch.tensor(self.prev_score))

    #observation_, reward, done, info = env.step(action)
    def step(self,action):
        """
        action contains x,y,size and colour
        """
        done = False
        
        action = np.clip(action,0,1)

        coords = np.round(action[:2]*self.imsize).astype('int32')
        radius = np.round(action[2]*self.max_radius).astype('int32')
        colour = int(np.round(action[3]))

        self.canvas = cv.circle(self.canvas,coords,radius,colour,-1)

        state = torch.tensor(self.canvas,dtype=torch.float32)

        
        if self.criterium != None:
            score = self.criterium.score(state)
        else:
            score = 0
        
        #reward = score-self.prev_score
        #reward = score-self.max_score
        reward = (state-self.state).mean()*((self.max_steps-self.steps)/self.max_steps)

        self.prev_score = score
        self.state = state
        
        if score > self.max_score:
            if self.max_score>.15:
                
                reward = reward + (score-self.max_score)*100
                self.max_score = score
            # if self.max_score >.1:
            #     reward = reward+(1+(score-self.max_score)*100)
            #     self.max_score = score
            # else:
            #     self.max_score = score
        
        return(state,self.prev_score,reward,done)

    
