import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import bpy
import bmesh
import torch.nn as nn
import mathutils

depsgraph = bpy.context.evaluated_depsgraph_get()

obj = bpy.data.objects['TestingObj']

bm = bmesh.new()

bm.from_object(obj,depsgraph)

bm.verts.ensure_lookup_table()



contours = []

for floor in bm.faces:
    height = floor.calc_center_median().z
    contour = []
    for v in floor.verts:
        contour.append(v.co[:2])
    contours.append(np.round(np.array(contour)))

segment = np.zeros([32,32,3])

im = np.zeros([32,32,3])

cv.fillPoly(im,[contours[0].astype(np.int32)],(1,1,1))

cv.fillPoly(segment,[contours[0].astype(np.int32)+[0,0]],[0,0,1])
cv.fillPoly(segment,[contours[0].astype(np.int32)+[10,0]],[0,1,0])
cv.fillPoly(segment,[contours[0].astype(np.int32)+[0,-10]],[1,0,0])

segment*=im

im = torch.tensor(im,dtype=torch.float).permute(-1,0,1)
segment = torch.tensor(segment,dtype=torch.float).permute(-1,0,1)





with torch.no_grad():
    ones_conv = nn.Conv2d(3,3,kernel_size=2,stride = 1, bias=False,groups=3)
    ones_conv.weight.fill_(1)
    
    adjacent = ones_conv(segment)

    corners = torch.where(adjacent.clamp(0,1).sum(0)==2,1,0)*torch.where(adjacent.sum(0)<=3,1,0)
    coords = corners.nonzero().roll(1,1)+torch.tensor([1,1])



attribute_values = [i for i in range(12)]

    

point = mathutils.Vector(np.array((*coords[0],0)))

num_iters = len(bm.edges)-1
for i in range(num_iters):
    bm.edges.ensure_lookup_table()
    edge = bm.edges[i]
    _verts = edge.verts

    closest,parameter = mathutils.geometry.intersect_point_line(point,_verts[0].co,_verts[1].co)
    
    distance = (closest-point).length
    if distance < 1:
        
        edge,_ = bmesh.utils.edge_split(edge,_verts[0],parameter)
        


attribute_values = []
for edge in bm.edges:
    middle = ((edge.verts[0].co+edge.verts[1].co)/2)
    channel = adjacent.permute(1,2,0)[round(middle[1]),round(middle[0])].sort()[1][-1]
    attribute_values.append(channel.item())

print(attribute_values)


living_obj = bpy.data.objects['living']

bm.to_mesh(living_obj.data)

attribute = living_obj.data.attributes.new(name="type", type="INT", domain="EDGE")

attribute.data.foreach_set("value", attribute_values)



plt.imsave('/home/jlemmens/Desktop/tempims/corners.png',corners)

plt.imsave('/home/jlemmens/Desktop/tempims/im.png',im.permute(1,2,0).numpy())
plt.imsave('/home/jlemmens/Desktop/tempims/seg.png',segment.permute(1,2,0).numpy())
