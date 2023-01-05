import numpy as np
import cv2
import yaml

from FaceBoxes import FaceBoxes
from TDDFA import TDDFA
from Sim3DR import RenderPipeline
from utils.functions import draw_landmarks, plot_image
from utils.render import render
from utils.depth import depth
from utils.tddfa_util import _to_ctype

import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation as R

def load_3d_obj(obj_path):                                                   
    vers = []                                                            
    colors = []                                                          
    tri = []                                                             
    with open(obj_path, 'r') as f:                                       
        lines = f.readlines()                                            
        for line in lines:                                               
            ps = line.split(" ")                                         
            if ps[0] == "v":                                             
                vers.append([float(ps[1]), float(ps[2]), float(ps[3])])  
                colors.append([float(ps[4]), float(ps[5]), float(ps[6])])
            elif ps[0] == "f":                                           
                tri.append([int(ps[1]), int(ps[2]), int(ps[3])])         
            else:                                                        
                raise NotImplementedError                                
    vers = np.array(vers, dtype=np.float32) 
    colors = np.array(colors, dtype=np.float32)
    tri = np.array(tri, dtype=np.int32)                                                  
    print(vers.shape, colors.shape, tri.shape)                              
    return vers, colors, tri                                             


# In[3]:


vers, colors, tri = load_3d_obj("examples/results/example_0_obj.obj")
img = cv2.imread("examples/inputs/example_0.jpg")
cfg = {                            
    'intensity_ambient': 0.3,      
    'color_ambient': (1, 1, 1),    
    'intensity_directional': 0.6,  
    'color_directional': (1, 1, 1),
    'intensity_specular': 0.1,     
    'specular_exp': 5,             
    'light_pos': (0, 0, 100),        
    'view_pos': (0, 0, 100)          
}                                  
                                   
render_app = RenderPipeline(**cfg) 

overlap = np.zeros_like(img)
overlap = render_app(vers, tri, overlap, texture=colors.copy())
plot_image(overlap)

# Rotate
# x_l = [-20, -15, -10, 0, 10, 20]
x_l = [-20 + i for i in range(0, 41, 2)]
y_l = [-20 + i for i in range(0, 41, 2)]
count = 0


f, axs = plt.subplots(3, 3)
for j, x in enumerate(x_l):
    for i, y in enumerate(y_l):
        Ry = R.from_euler('y', y, degrees=True).as_matrix()
        Rx = R.from_euler('x', x, degrees=True).as_matrix()
        overlap = np.zeros_like(img)
        vers1 = Rx@Ry@(vers.T)
        vers1 = _to_ctype(vers1.T).astype(np.float32)

        overlap = render_app(vers1, tri, overlap, texture=colors.copy())
        overlap = overlap[...,::-1]
        overlap = overlap[..., ::-1]
        cv2.imwrite(f"examples/results/d_face_0/{count:04}.png", overlap)
        count += 1

f.suptitle('Openness 100%')
plt.tight_layout()
