
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os
import torch
import numpy as np 
import matplotlib.pyplot as plt
import pickle


class Grayscale_imgs(Dataset):
    def __init__(self,path):
        self.path = path
        self.filenames = os.listdir(self.path)
        self.transform_norm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[.5], std =[.5])])

    def __len__(self):
        return(len(self.filenames))
    
    def __getitem__(self, index):

        filename = self.filenames[index]
        sample = Image.open(f'{self.path}/{filename}')
        sample,_,_ = sample.resize((32, 32), Image.LANCZOS).split()
        #applying the normalization to the sample        
        sample = self.transform_norm(sample)
        
        #swapping the axis to be compatible with the model
        return(sample)

# path = r"D:/Architecture Jair/Computational intelligence for integrated design/CIPython/Data/Input"
# dataloader = DataLoader(Grayscale_imgs(path) , batch_size = 64, shuffle=True)





def save_latent_space_img(model, scale=1.0, n=12, digit_size=32, figsize=15,name="./latent_space_imgs/none.png"):
    # display a n*n 2D manifold of digits
    figure = np.zeros((digit_size * n, digit_size * n))

    # construct a grid 
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = torch.tensor([[xi, yi]], dtype=torch.float)
            model.eval()
            z_sample = model.project_to_decoder(z_sample)
            x_decoded = model.decoder(z_sample.unsqueeze(-1).unsqueeze(-1))
            digit = x_decoded[0].detach().cpu().reshape(digit_size, digit_size)
            figure[i * digit_size : (i + 1) * digit_size, j * digit_size : (j + 1) * digit_size,] = digit

    plt.figure(figsize=(figsize, figsize))
    plt.title('VAE Latent Space Visualization')
    start_range = digit_size // 2
    end_range = n * digit_size + start_range
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("mean, z [0]")
    plt.ylabel("var, z [1]")
    plt.imsave(name,figure)
    return(None)




class Generated_Jaccard_Mnist(Dataset):
    def __init__(self,path):
        self.path = path
        self.filenames = os.listdir(self.path)

    def __len__(self):
        return(len(self.filenames))
    
    def __getitem__(self, index):

        filename = self.filenames[index]
        with open(f'{self.path}/{filename}','rb') as f:
            sample = pickle.load(f)
    

        return(sample)


def discount_rewards(rewards, gamma=0.99):
    """
    Return discounted rewards based on the given rewards and gamma param.
    """
    new_rewards = [float(rewards[-1])]
    for i in reversed(range(len(rewards)-1)):
        new_rewards.append(float(rewards[i]) + gamma * new_rewards[-1])
    return np.array(new_rewards[::-1])

def calculate_gaes(rewards, values, gamma=0.99, decay=0.97):
    """
    Return the General Advantage Estimates from the given rewards and values.
    Paper: https://arxiv.org/pdf/1506.02438.pdf
    """
    next_values = np.concatenate([values[1:], [0]])
    deltas = [rew + gamma * next_val - val for rew, val, next_val in zip(rewards, values, next_values)]

    gaes = [deltas[-1]]
    for i in reversed(range(len(deltas)-1)):
        gaes.append(deltas[i] + decay * gamma * gaes[-1])

    return np.array(gaes[::-1])


def generate_random_episode(env,steps = 50):
    actions = []
    states = []
    values = []
    canvas,value = env.reset()
    states.append(canvas)
    values.append(value)
    ep_reward = 0 

    for _ in range(steps):
        action = torch.rand(1,4)
        action = torch.rand(1,4)
        action*= torch.tensor([[1,1,.75,1]])
        action+= torch.tensor([[0,0,.25,0]])
        canvas,value,reward,done = env.step(action[0].detach().numpy())
        ep_reward+=reward
        actions.extend(action)
        states.append(canvas)
        values.append(value)
    return(torch.stack(actions),torch.stack(states),torch.tensor(values).unsqueeze(dim=-1),ep_reward)




class Offline_Data(Dataset):
    def __init__(self,buffer_size=100):
        self.states = []
        self.actions = []
        self.scores = []
        self.buffer_size= buffer_size

    def add_episode(self,states,actions,scores):
        if len(self.states) == self.buffer_size:
            self.states = self.states[1:]
            self.actions = self.actions[1:]
            self.scores = self.scores[1:]
        self.states.append(states)
        self.actions.append(actions)
        self.scores.append(scores)

    def __getitem__(self, index):
        states = self.states[index]
        scores = self.scores[index]
        actions = self.actions[index]
        return {'states': states, 'scores': scores, 'actions': actions}
    
    def __len__(self):
        return len(self.states)


class SwissData(Dataset):

    def __init__(self,path):
        self.path = path
        self.filenames = os.listdir(f'{self.path}/Output/')
        self.transforms = transforms.Compose([transforms.ToTensor(), transforms.Resize((32,32),antialias=False)])


    def __len__(self):
        return(len(self.filenames))
    
    def __getitem__(self, index):
            
        filename = self.filenames[index]
        segmentation = Image.open(f'{self.path}/Output/{filename}')   
        segmentation = self.transforms(segmentation)
        grayscale = Image.open(f'{self.path}/Input/{filename}')   
        grayscale = self.transforms(grayscale)
        
        return(grayscale,segmentation)
    


    
def paretofront(input, minimize = True, softness = .1):
    inputPoints = input.tolist()

    def dominates(row, candidateRow,softness=0.0):
        domcount = 0
        for n in range(len(row)):
            if row[n] > 0: 
                if row[n]/candidateRow[n] >= 1+softness:
                    domcount+=1
            else:
                if row[n]/candidateRow[n] <= 1+softness:
                    domcount+=1
        return(domcount==len(row))

    paretoPoints = []
    candidateRowNr = 0
    dominatedPoints = []
    paretoIndices = []
    while True:
        candidateRow = inputPoints[candidateRowNr]
        inputPoints.remove(candidateRow)
        rowNr = 0
        nonDominated = True
        while len(inputPoints) != 0 and rowNr < len(inputPoints):
            row = inputPoints[rowNr]
            if dominates(candidateRow, row,softness=softness):
                # If it is worse on all features remove the row from the array
                inputPoints.remove(row)
                dominatedPoints.append(row)
            elif dominates(row, candidateRow,softness=softness):
                nonDominated = False
                dominatedPoints.append(candidateRow)
                rowNr += 1
            else:
                rowNr += 1

        if nonDominated:
            # add the non-dominated point to the Pareto frontier
            paretoPoints.append(candidateRow)
            paretoIndices.append(np.where(input == candidateRow)[0][0])

        if len(inputPoints) == 0:
            break
    
    
    return np.array(paretoPoints), np.array(dominatedPoints), np.array(paretoIndices)