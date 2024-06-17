import torch
import torch.nn as nn
from VAE_modules_20_11 import Decoder, Encoder
import math
import numpy as np 
from utils import paretofront


"""
Written by Jair Lemmens
For educational use only
"""



class Observer(nn.Module):
    def __init__(self, depths =[5,3,3,1,1,1],dims=[1,8,16,32,16,8],latent_sigmoid=False):
        super().__init__()
        self.encoder = Encoder(depths,dims)
        self.decoder = Decoder(depths,dims)
        self.latent_sigmoid= latent_sigmoid

    def enc(self,state):
        latent = self.encoder(state)
        if self.latent_sigmoid:
            latent = latent.sigmoid()
        return(latent.squeeze())
    
    def dec(self,latent):
        reconstruction = self.decoder(latent)
        reconstruction = reconstruction.sigmoid()
        return(reconstruction.squeeze())
    
    def forward(self,state):
        latent = self.encoder(state)
        if self.latent_sigmoid:
            latent = latent.sigmoid()
        reconstruction = self.decoder(latent)
        reconstruction = reconstruction.sigmoid()
        return(latent.squeeze(),reconstruction.squeeze())

    def time_series_as_batch(self,states,simul_series=1):
        latents = []
        reconstructions = []

        for subbatch in states.view(-1,states.shape[1]*simul_series,*states.shape[2:]):
            latent,reconstruction = self(subbatch.unsqueeze(1))
            latents.append(latent)
            reconstructions.append(reconstruction)
            
        latents = torch.stack(latents).view(*states.shape[:2],-1)
        reconstructions = torch.stack(reconstructions).view(states.shape)
        return(latents,reconstructions)


"""
USE TO PRETRAIN OBSERVER

mse = nn.MSELoss()
optimizer = torch.optim.Adam(observer.parameters(), lr=1e-3)
losses = []
num_eps = 500

pbar = tqdm(range(num_eps))
for episode in enumerate(pbar):

    env.criterium = Jaccard_Index_Criterium(next(iter(dataloader))[0].squeeze(0,1))
    actions, states, values, _ = generate_random_episode(env)
    
    _,reconstruction = observer(states.unsqueeze(1))
    loss = mse(reconstruction,states)
    optimizer.zero_grad()
    loss.mean().backward()
    optimizer.step()
    pbar.set_postfix(loss = loss.item())
    losses.append(loss.item())

plt.plot(gaussian_filter1d(losses, sigma=10),label='MSE')

i = np.random.randint(0,reconstruction.shape[0]-1)
f, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(reconstruction[i].detach())
ax2.imshow(states[i])        
        
torch.save(observer.state_dict(), './trained_modules/Observer_depths=_,dims=_,latent_sigmoid=_.pt')
observer.load_state_dict(torch.load('./trained_modules/Observer_depths=_,dims=_,latent_sigmoid=_.pt'))
"""



class Variational_Observer(nn.Module):
    def __init__(self, depths =[5,3,3,1,1,1],dims=[1,8,16,32,16,8],beta=1):
        super().__init__()
        self.encoder = Encoder(depths,dims)
        self.decoder = Decoder(depths,dims)
        
        self.mean = nn.Linear(dims[-1],dims[-1]) #add sigmoid
        self.var = nn.Linear(dims[-1],dims[-1])

        self.beta = beta
        
    def enc(self,state):
        latent = self.encoder(state).squeeze()
        mean = self.mean(latent)
        return(mean.squeeze())
    
    def dec(self,latent):
        reconstruction = self.decoder(latent).sigmoid()
        return(reconstruction.squeeze())
    
    def reparam(self, mean, var):
        epsilon = torch.randn_like(var)
        return(mean+var*epsilon)

    def train_step(self,state):
        latent = self.encoder(state).squeeze()
        mean = self.mean(latent)
        var = self.var(latent)
        z = self.reparam(mean,var).unsqueeze(-1).unsqueeze(-1)
        reconstruction = self.decoder(z).sigmoid()
        loss = self.loss(state,reconstruction,mean,var)
        return(reconstruction.squeeze(),loss)
    
    def forward(self,state):
        latent = self.encoder(state).squeeze()
        mean = self.mean(latent)
        reconstruction = self.decoder(mean.unsqueeze(-1).unsqueeze(-1)).sigmoid()
        return(mean.squeeze(),reconstruction.squeeze())
    
    def loss(self, state, recon_state, mean, var):
        recon_loss = nn.functional.mse_loss(state,recon_state)
        KLD = - 0.5 * torch.sum(1+ var - mean.pow(2) - var.exp())
        return(recon_loss+self.beta*KLD)
    
# observer = Variational_Observer(depths=[3,3,3,3,3,3],dims=[1,32,64,128,64,32],beta=.05).to('cuda')
# losses = []

"""
Use to pretrain variational observer
optimizer = torch.optim.Adam(observer.parameters(), lr=1e-6)

num_eps = 1

pbar = tqdm(range(num_eps))
for episode in enumerate(pbar):

    env.criterium = Jaccard_Index_Criterium(next(iter(dataloader))[0].squeeze(0,1))
    actions, states, values, _ = generate_random_episode(env,steps=200)
    
    states = states.to('cuda')
    reconstruction,loss = observer.train(states.unsqueeze(1))
    optimizer.zero_grad()
    loss.mean().backward()
    optimizer.step()
    pbar.set_postfix(loss = loss.item())
    losses.append(loss.item())

    if episode[0]%50==0:
        clear_output(wait=True)
        plt.yscale('log')
        plt.plot(gaussian_filter1d(losses, sigma=10))
        plt.show()

        
torch.save(observer.state_dict(), './trained_modules/observer=Variational_Observer(depths=[3,3,3,3,3,3],dims=[1,32,64,128,64,32],beta=.05).pt')
observer.load_state_dict(torch.load('./trained_modules/observer=Variational_Observer(depths=[3,3,3,3,3,3],dims=[1,32,64,128,64,32],beta=.05).pt'))
"""

class Skipped_MLP_Block(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(size,size*4),nn.GELU(),nn.Linear(size*4,size))
    def forward(self,x):
        return(x + self.mlp(x))


class StackedMLP(nn.Module):
    def __init__(self, input_size=32, output_size = 32,num_layers=1):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(Skipped_MLP_Block(input_size))
        layers.append(nn.Linear(input_size,output_size))
        self.network = nn.Sequential(*layers)
    def forward(self, x):
        return(self.network(x))
    

class StackedMLP_norm(nn.Module):
    def __init__(self, input_size=32, output_size = 32,num_layers=1):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(Skipped_MLP_Block(input_size))
            layers.append(nn.BatchNorm1d(input_size))
        layers.append(nn.Linear(input_size,output_size))
        self.network = nn.Sequential(*layers)
    def forward(self, x):
        return(self.network(x))


class Self_Attention_Head(nn.Module):
    def __init__(self, dim, block_size,masked = True,num_heads =4):
        super().__init__()

        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        
        self.attention = nn.MultiheadAttention(dim,num_heads=num_heads,batch_first=True)

        self.masked = masked
        self.register_buffer('tril', torch.tril(torch.ones(block_size,block_size)))

        self.project = nn.Linear(dim,dim)

    def forward(self,x):

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        if self.masked:
            attn,attn_weights = self.attention(q,k,v,attn_mask=self.tril[:q.shape[-2],:q.shape[-2]])
        else:
            attn,attn_weights = self.attention(q,k,v)
        out = self.project(attn)
        return(x+out) 

    def get_attn(self,x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        if self.masked:
            attn,attn_weights = self.attention(q,k,v,attn_mask=self.tril[:q.shape[-2],:q.shape[-2]])
        else:
            attn,attn_weights = self.attention(q,k,v)
        return(attn_weights) 

class Transformer(nn.Module):
    def __init__(self,input_dim,hidden_dim,num_layers=1,masked=True,mask_size = 200):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim,hidden_dim))
        for _ in range(num_layers):
            layers.append(Self_Attention_Head(hidden_dim,mask_size,masked))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(Skipped_MLP_Block(hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
        self.network= nn.Sequential(*layers)

    def forward(self,x):
        return(self.network(x))

    

""""
DEPRECATED
class Working_memory(nn.Module):
    def __init__(self, short_mem_size = (5,9),hidden_size=18, output_size=256, num_layers = 1):
        super(Working_memory, self).__init__()

        #layers
        self.lstm = nn.LSTM(input_size = short_mem_size[-1],hidden_size=hidden_size, num_layers = num_layers)
        self.project = nn.Linear(short_mem_size[0]*hidden_size,output_size)
        #set variables
        self.output_size = output_size
        self.short_mem_size = short_mem_size
        self.mem_cntr = 0
        #initialize memory
        self.reset()
        
    def forward(self,state):
        index = self.mem_cntr % self.short_mem_size[0]
        self.short_mem[index] = state
        self.mem_cntr += 1
        lstm_out,(self.hidden_state,self.cell_state) = self.lstm(self.short_mem,(self.hidden_state,self.cell_state))

        perceived_state = self.project(lstm_out.flatten())
        return(perceived_state)
    
    def reset(self):
        self.short_mem = torch.zeros(self.short_mem_size)
        _,(self.hidden_state,self.cell_state) = self.lstm(self.short_mem)

"""





class WorldModel(nn.Module):
    def __init__(self, observer_depths=[3,3,3,3,3,3], observer_dims = [1,16,32,64,32,16], transformer_layers=3, score_predictor_layers=1, state_predictor_layers=3 , action_size=4, episode_length=200, device= 'cuda', alpha = 0.5, beta = .5, lr = 1e-4):
        super().__init__()
        self.episode_length = episode_length

        self.observer_output_size = observer_dims[-1]
        self.observer = Observer(depths=observer_depths,dims=observer_dims,latent_sigmoid=True).to(device)
        self.temporal_analyser = Transformer(obs_size=observer_dims[-1],num_layers=transformer_layers,masked=True,episode_length=episode_length).to(device)
        self.score_predictor = StackedMLP(input_size=2*observer_dims[-1]+1,output_size=1,num_layers=score_predictor_layers).to(device)
        self.state_predictor = StackedMLP(input_size=observer_dims[-1]+action_size,output_size=16,num_layers=state_predictor_layers).to(device)

        self.alpha = alpha
        self.beta = beta
        self.optimiser = torch.optim.Adam(list(self.temporal_analyser.parameters()) + list(self.score_predictor.parameters())+ list(self.observer.parameters())+ list(self.state_predictor.parameters()), lr=lr)
        self.optimiser_state = torch.optim.Adam(list(self.observer.parameters())+ list(self.state_predictor.parameters()), lr=lr)
        self.device =  device
        
        self.dyn_losses = []
        self.score_losses = []
        self.recon_losses = []

        self.training_steps = 0

    
    def train_batch(self,dataloader):
        data = next(iter(dataloader))

        batch_size,ep_length = data['states'].shape[:2]

        states = data['states'].to(self.device).flatten(0,1)
        scores = data['scores'].to(self.device).detach()
        actions = data['actions'].to(self.device)

        obs, reconstructions = self.observer(states.unsqueeze(1))

        obs = obs.reshape((batch_size,ep_length,obs.shape[-1]))

        permutation = torch.randperm(self.episode_length)
        prev_obs = obs[:,:-1]
        new_obs = obs[:,1:]
        prev_scores = scores[:,:-1]
        target_scores = scores[:,1:]


        temporal_latents = self.temporal_analyser(torch.concatenate([prev_obs[:,permutation],prev_scores[:,permutation]],dim=-1))
        pred_scores = self.score_predictor(torch.concatenate([new_obs[:,permutation],temporal_latents],dim=-1))
            
        pred_obs = self.state_predictor(torch.concatenate([prev_obs,actions],dim=-1)).sigmoid()

        dyn_loss = ((pred_obs-new_obs)**2).mean()
        score_loss = ((pred_scores-target_scores[:,permutation])**2).mean()
        recon_loss = ((reconstructions-states)**2).mean()
        loss = (dyn_loss+self.alpha*score_loss+self.beta*recon_loss)

        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        self.dyn_losses.append(dyn_loss.item())
        self.score_losses.append(score_loss.item())
        self.recon_losses.append(recon_loss.item())

        return(dyn_loss.item(),score_loss.item(),recon_loss.item())

    def train_state_pred(self,dataloader):
        data = next(iter(dataloader))

        batch_size,ep_length = data['states'].shape[:2]

        states = data['states'].to(self.device).flatten(0,1)
        
        actions = data['actions'].to(self.device)

        obs, reconstructions = self.observer(states.unsqueeze(1))

        obs = obs.reshape((batch_size,ep_length,obs.shape[-1]))
        
        prev_obs = obs[:,:-1]
        new_obs = obs[:,1:]
            
        pred_obs = self.state_predictor(torch.concatenate([prev_obs,actions],dim=-1)).sigmoid()

        dyn_loss = ((pred_obs-new_obs)**2).mean()
        recon_loss = ((reconstructions-states)**2).mean()
        loss = (dyn_loss+self.beta*recon_loss)

        self.optimiser_state.zero_grad()
        loss.backward()
        self.optimiser_state.step()

        self.dyn_losses.append(dyn_loss.item())
        
        self.recon_losses.append(recon_loss.item())

        return(dyn_loss.item(),recon_loss.item())



class Quantizer(nn.Module):
    def __init__(self, codebook_size, latent_dim, codebook_alpha = 1):
        super().__init__()
        self.embedding = nn.Embedding(codebook_size, latent_dim)
        self.codebook_alpha = codebook_alpha
        self.embedding.weight.data.uniform_(-1/codebook_size,1/codebook_size)
    def forward(self,x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(-1,C)
        dist = torch.cdist(x,self.embedding.weight)
        min_encoding_indices = torch.argmin(dist, dim=-1)
        quant_out = torch.index_select(self.embedding.weight, 0, min_encoding_indices.view(-1))
        x = x.reshape((-1, x.size(-1)))

        commmitment_loss = ((quant_out.detach() - x) ** 2).mean()
        codebook_loss = ((quant_out - x.detach()) ** 2).mean()
        quantize_loss = commmitment_loss+self.codebook_alpha*codebook_loss

        quant_out = x + (quant_out - x).detach()
        quant_out = quant_out.reshape((B, H, W, C)).permute(0, 3, 1, 2)
        min_encoding_indices = min_encoding_indices.reshape((-1, quant_out.size(-2), quant_out.size(-1)))
        return quant_out, quantize_loss, min_encoding_indices

    def get_indices(self,idx):
        quant_out = torch.index_select(self.embedding.weight,0,idx.view(-1)).reshape(*idx.shape,-1).permute(0,3,1,2)
        return(quant_out)

class VQVAE(nn.Module):
    def __init__(self, depths=[1,1,1,1], dims = [1,32,16,1], codebook_size=4, quantizer_alpha = 1,  codebook_alpha = 1, device = 'cpu'):
        super().__init__()
        self.device = device
        self.quantizer_alpha = quantizer_alpha
        self.encoder = Encoder(depths,dims).to(device)
        self.quantizer = Quantizer(codebook_size,dims[-1],codebook_alpha).to(device)
        self.decoder = Decoder(depths,dims).to(device)

    def enc(self,sample):
        x = self.encoder(sample)
        x, _,indices = self.quantizer(x)
        return(x,indices)

    def forward(self,sample):
        x = self.encoder(sample)
        x, quantizer_loss,_ = self.quantizer(x)
        x = self.decoder(x).sigmoid()
        loss = ((sample-x)**2).mean()+self.quantizer_alpha*quantizer_loss

        return(x,loss)
    
# optimiser = torch.optim.Adam(vqvae.parameters(),lr = 2e-4)

# for epoch in range(1000):
#     pbar = tqdm(bufferloader)
#     for n,data in enumerate(pbar):
        
#         actions, states, scores, _ = generate_random_episode(env,steps=50)
#         buffer.add_episode(states,actions,scores)
#         gray, _ = next(iter(swissdata_loader))
#         gray = gray.roll(list(np.random.randint(-10,10,2)),(-1,-2))
#         buffer.add_episode(gray.squeeze(), actions,scores)

#         state = data['states'].flatten(0,1).unsqueeze(-3).to(vqvae.device)
     
#         optimiser.zero_grad()
#         recon,loss = vqvae(state)
#         loss.backward()
#         optimiser.step()
#         losses.append(loss.cpu().detach())

#         if n%10==0:
#             clear_output(wait=True)    
#             f, (ax1, ax2,ax3,ax4) = plt.subplots(1, 4,figsize=(30, 5))
#             ax1.plot(gaussian_filter1d(losses, sigma=10))
#             ax1.set_yscale('log')

#             embedding_loc = vqvae.quantizer.embedding.weight.cpu().detach().numpy()
#             ax2.scatter(embedding_loc[:,0],embedding_loc[:,1])
#             i = np.random.randint(len(state))
#             ax3.imshow(state[i].cpu().detach().permute(1,2,0))
#             ax4.imshow(recon[i].cpu().detach().permute(1,2,0))
            
#             plt.show()


class Temporal_Analyser(nn.Module):
    def __init__(self,dim, num_tokens, num_layers = 3, episode_len= 199):
        super().__init__()
        self.codebook = nn.Embedding(num_tokens,4)
        self.codebook.weight.data.uniform_(-1/num_tokens,1/num_tokens)
        self.score_embedding = nn.Linear(1,64)
        self.transformer = Transformer(64,dim,num_layers,mask_size=episode_len,masked=False)
    
    def forward(self, indices,scores):
        obs_tokens = self.get_codebook_tokens(indices)
        score_embeddings = self.score_embedding(scores)
        embeddings = obs_tokens+score_embeddings
        temporal_latents = self.transformer(embeddings)
        return(temporal_latents)
    
    def get_attn(self, indices,scores):
        obs_tokens = self.get_codebook_tokens(indices)
        
        score_embeddings = self.score_embedding(scores)
        embeddings = obs_tokens+score_embeddings
        attn = self.transformer.network[1].get_attn(embeddings)
        return(attn)
    
    def get_codebook_tokens(self,indices):
        return(torch.index_select(self.codebook.weight,0,indices.flatten()).reshape([*indices.shape[:2],-1]))
    


class GeneticAlgorithm():
    def __init__(self, poolsize=100 ,gene_shape = (4,4,3), num_parents=25,num_offspring =50, maximize = True, shift_rate = .1, replace_rate=.05, pareto_softness= 0):
        super().__init__()
        self.poolsize = poolsize
        self.gene_shape = gene_shape
        self.num_parents = num_parents
        self.num_offspring = num_offspring
        self.maximize = maximize
        self.shift_rate = shift_rate
        self.replace_rate = replace_rate
        self.pareto_softness = pareto_softness
        self.genes = torch.randn(poolsize,*gene_shape)
    
    def select_parents(self,fitness):
        if len(fitness.shape)>1:
            paretopoints,dominated,indices= paretofront(fitness.detach().cpu().numpy(),minimize=False,softness=self.pareto_softness)
            #sort by mean distance to other points on the paretofront to reward novelty if there are too many points
            distances = torch.tensor([np.linalg.norm(point-paretopoints,axis=-1).mean() for point in paretopoints])
            return(self.genes[torch.tensor(indices)[distances.argsort(descending=True)][:self.num_parents]])
        else:
            return(self.genes[fitness.argsort(descending=self.maximize)[:self.num_parents]])
    
    def parent_indices(self,fitness):
        if len(fitness.shape)>1:
            paretopoints,dominated,indices= paretofront(fitness.detach().cpu().numpy(),minimize=False,softness=self.pareto_softness)
            #sort by mean distance to other points on the paretofront to reward novelty if there are too many points
            distances = torch.tensor([np.linalg.norm(point-paretopoints,axis=-1).mean() for point in paretopoints])
            return(torch.tensor(indices)[distances.argsort(descending=True)][:self.num_parents])
        else:
            return(fitness.argsort(descending=self.maximize)[:self.num_parents])
        
    def crossover(self,parent_genes):
        permutation = torch.randperm(self.num_offspring)
        parent_gene_length = math.ceil(self.gene_shape[0]/2)
        offspring_genes = parent_genes.repeat(math.ceil(self.poolsize/len(parent_genes)),*([1]*(len(parent_genes.shape)-1)))[permutation]
        offspring_genes = torch.concat([torch.roll(offspring_genes,1,dims=0)[:,:parent_gene_length],offspring_genes[:,parent_gene_length:]],dim=1)
        #self.genes = torch.concat([parent_genes,offspring_genes],dim=0)
        return(offspring_genes)
    
    def mutate_shift(self,genes,rate=.1):
        return(genes+torch.randn(genes.shape[1:])*rate)
    
    def mutate_replace(self,genes,rate=.05):
        mask = torch.rand(genes.shape[:-1])>(1-rate)
        mask = mask.unsqueeze(-1).repeat_interleave(genes.shape[-1],dim=-1)
        return(torch.where(mask,torch.randn_like(genes),genes))
    
    def step(self,fitness):
        parent_genes = self.select_parents(fitness)
        offspring_genes = self.crossover(parent_genes)
        offspring_genes = self.mutate_shift(offspring_genes,self.shift_rate)
        offspring_genes = self.mutate_replace(offspring_genes,self.replace_rate)
        self.genes = torch.concat([parent_genes,offspring_genes,torch.randn(self.poolsize-len(parent_genes)-self.num_offspring,*self.gene_shape)],dim=0)
        return(self.genes)
