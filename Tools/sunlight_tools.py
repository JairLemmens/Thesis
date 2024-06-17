import numpy as np
import cv2 as cv 
import shapely
from Tools.geometry_tools import draw_polygon

#with intensity 
# def tris_shadow(tris,sky_vectors,intensities,target_z=0,scale=1,imsize=255):
    
#     offset_tris_total = np.zeros([sky_vectors.shape[0],tris.shape[0],3,2],dtype='int32')

#     xy= tris[:,:,:2]
#     z = tris[:,:,2]
#     for n,vector in enumerate(sky_vectors):
#         temp_z = np.swapaxes(np.swapaxes(np.tile(z-target_z,[2,1,1]),0,1),1,2)
#         temp_v = np.multiply(vector[:2],temp_z)*scale
#         offset_tris_total[n] = np.add(xy,temp_v).astype('int32')

#     shadow_matrix = np.zeros([145,imsize,imsize,1],dtype='float64')
#     for intensity, offset_tris,shadow_layer in zip(intensities,offset_tris_total,shadow_matrix):
#         cv.fillPoly(shadow_layer,offset_tris,intensity)

#     return(shadow_matrix)


def shapely_shadow_matrix(multipolygon,sky_vectors,target_z=0,scale=1,imsize=32):
    def project_z(poly,vector=[1,1],target_z=0):
        offset = np.einsum('j,k->kj',vector,poly[:,-1]-target_z)
        lower_mask = np.tile(poly[:,-1]-target_z>0,(2,1)).swapaxes(1,0)
        poly[:,:2]+=offset
        poly[:,:2] = np.where(lower_mask,poly[:,:2],0)
        poly[:,-1] = target_z
        return(poly)
    
    shadow_matrix = np.zeros([145,imsize,imsize],dtype='int32')
    for sky_vector,shadow_layer in zip(sky_vectors,shadow_matrix):
        temp_polygon = shapely.transform(multipolygon,lambda x: project_z(x,sky_vector[:2]*scale,target_z), include_z=True)
        if temp_polygon.area > 20:    
            temp_polygon = shapely.unary_union(shapely.force_2d(temp_polygon).buffer(0))
            draw_polygon(temp_polygon,shadow_layer)
                    
    return(shadow_matrix)