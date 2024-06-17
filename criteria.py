import torch
# from Tools.ladybug_tools import Climate
from Tools.geometry_tools import extract_contours, triangulate, polygons_from_opencv, substring_from_points, substrings_from_parameters,image_to_shapely
import shapely
import numpy as np
from Tools.sunlight_tools import shapely_shadow_matrix
import matplotlib.pyplot as plt
# import cv2 as cv
# from torchvision import transforms
# import numpy as np

class Jaccard_Index_Criterium():
    def __init__(self,reference):
        self.reference = reference
    
    def score(self,sample):
        intersection = torch.sum(sample*self.reference)
        union = torch.sum(torch.max(sample,self.reference))
        index =intersection/union
        if index.isnan():
            print(f'intersection {intersection}')
            print(f'union {union}') 
            index = 0
        return(index)
    

class Lighting_Criterium():
    def __init__(self,climate,target_irradiance=100,levels =5, level_height =5):
        self.target_irradiance = target_irradiance
        self.sky_vectors = climate.sky_vectors
        self.intensities = torch.tensor(climate.intensities,dtype=torch.float32)
        self.total_irradiance = climate.intensities.sum()
        self.levels = levels
        self.level_height = level_height
    def score(self,multipoly):
        shadow_matrix = shapely_shadow_matrix(multipoly,self.sky_vectors)
        shadow = torch.einsum('i,ijk->jk',self.intensities,torch.tensor(shadow_matrix,dtype=torch.float32))
        light = self.total_irradiance-shadow
        mask = shadow_matrix[-1]
        sufficient = torch.where(light*mask>self.target_irradiance,1,0).sum()/mask.sum()
        return(sufficient.item())
    

class Area_Criterium():
    def __init__(self,target,sharpness=None,skew=20):
        self.target = target
        self.max_val = 1
        if sharpness == None:
            self.sharpness = 1/(target/10)
        else:
            self.sharpness = sharpness
        self.skew = skew
        testx = np.arange(0,target*2)
        y = self.score(testx)
        self.max_val = max(y)
        plt.plot(testx,y/self.max_val)
        plt.title('AreaGraph')
        plt.show
    def score(self,x):
        if self.skew >= 0:
            return(2/(2**(self.sharpness*(x-self.target))+2**-(self.skew*self.sharpness*(x-self.target)))/self.max_val)
        else:
            return(2/(2**(-self.skew*self.sharpness*(x-self.target))+2**-(self.sharpness*(x-self.target)))/self.max_val)
        

# class MomentOfInertia_Criterium():
#     def __init__(self):
#         self.transform = transforms.Resize((255,255),antialias=False)
#     def score(self,sample):
#         sample = self.transform(sample)
#         final = (np.stack([sample,sample,sample],axis = -1)*255).astype(np.uint8).squeeze()
#         contours, hierarchy = extract_contours(final,epsilon_factor=.005)
#         polygons = polygons_from_opencv(contours,hierarchy)
#         centroid = [polygons.centroid.x,polygons.centroid.y]
#         grid = np.stack(np.mgrid[:255,:255],axis=-1)
#         distance = np.linalg.norm(centroid - grid,axis=-1)**2
#         score = sum(distance.flatten()*sample.flatten().numpy())/sample.sum()/10000
#         return(score)
