
import torch.nn as nn


class Conv_block(nn.Module):
    def __init__(self,dim,dConv_kernel_size=7):
        super().__init__()
        self.depth_conv = nn.Conv2d(dim,dim,kernel_size=dConv_kernel_size,padding=int((dConv_kernel_size-1)/2),groups=dim)
        self.norm = nn.BatchNorm2d(dim)
        self.conv_1 = nn.Conv2d(dim,dim*4,kernel_size=1)
        self.act = nn.GELU()
        self.conv_2 = nn.Conv2d(dim*4,dim,kernel_size=1)

    def forward(self,x):
        input = x
        x = self.depth_conv(x)
        x = self.norm(x)
        x = self.conv_1(x)
        x = self.act(x)
        x = self.conv_2(x)
        return(x+input)
    

"""
Convolutional encoder based on the ConvNext paper. it repeats the Conv_Blocks depth times per layer and follows them with a
downscaling operation.
"""
class Encoder(nn.Module):
    def __init__(self, depths=[3, 3, 9, 3],dims=[96, 192, 384, 768],dConv_kernel_size=5):
        super().__init__()
        self.layers = nn.ModuleList()

        for layer_n,depth in enumerate(depths):
            for sublayer_n in range(depth):
                self.layers.append(Conv_block(dims[layer_n],dConv_kernel_size))
            if layer_n < len(depths)-1:
                self.layers.append(nn.Conv2d(dims[layer_n],dims[layer_n+1],kernel_size= 2, stride = 2))

    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
        return(x)
    
"""
The convolutional decoder is used in the pretraining fase where it forms the autoencoder together with the convolutional encoder.
The goal of this is to teach the encoder how to create dense image embeddings without relying on the more unpredictable transformer.
"""
class Decoder(nn.Module):
    def __init__(self ,depths=[3, 3, 9, 3],dims=[96, 192, 384, 768],dConv_kernel_size=5):
        super().__init__()
        self.depths = list(reversed(depths))
        self.dims = list(reversed(dims))
        self.layers = nn.ModuleList()
        for layer_n,depth in enumerate(self.depths):

            for _ in range(depth):
                self.layers.append(Conv_block(self.dims[layer_n],dConv_kernel_size))
            if layer_n < len(depths)-1:     
                self.layers.append(nn.ConvTranspose2d(self.dims[layer_n],self.dims[layer_n+1],kernel_size=2,stride=2))

    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
        return(x)