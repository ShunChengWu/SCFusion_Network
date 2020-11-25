if __name__ == '__main__' and __package__ is None:
    from os import sys
    sys.path.append('../')
import torch
import math
import numpy as np

# try to import meshing_occupancy
import importlib
im_mo = importlib.util.find_spec("meshing_occupancy")
found_mo = im_mo is not None
if found_mo:
    import meshing_occupancy as mo


offsets=[[-0.5,-0.5,-0.5],
         [-0.5,-0.5,0.5],
         [-0.5,0.5,-0.5],
         [-0.5,0.5,0.5],
         [0.5,-0.5,-0.5],
         [0.5,-0.5,0.5],
         [0.5,0.5,-0.5],
         [0.5,0.5,0.5]
         ]
index_offset=[
    [0,1,3],[0,3,2],
    [4,0,2],[4,2,6],
    [5,4,6],[5,6,7],
    [1,5,7],[1,7,3],
    [2,3,7],[2,7,6],
    [1,0,4],[1,4,5]
]
color_lists = [
    [255,255,255],
    [255,0,0],
    [255,255,0],
    [0,255,0],
    [0,255,255],
     [0,0,255],
     [255,0,255]
]

NYU_13_CLASSEScolour_code = [[  255,   255,   255],
       [  0,   0, 255],
       [232,  88,  47],
       [  0, 217,   0],
       [148,   0, 240],
       [222, 241,  23],
       [255, 205, 205],
       [  0, 223, 228],
       [106, 135, 204],
       [116,  28,  41],
       [240,  35, 235],
       [  0, 166, 156],
       [249, 139,   0],
       [225, 228, 194]]

NYU_13_CLASSES = [(0,'Unknown'),
                  (1,'Bed'),
                  (2,'Books'),
                  (3,'Ceiling'),
                  (4,'Chair'),
                  (5,'Floor'),
                  (6,'Furniture'),
                  (7,'Objects'),
                  (8,'Picture'),
                  (9,'Sofa'),
                  (10,'Table'),
                  (11,'TV'),
                  (12,'Wall'),
                  (13,'Window')
]

SunCG_11_Color = [
    [255,255,255],
    [253,210,250],
    [214,199,137],
    [205,216,253],
    [86,52,135],
    [234,249,71],
    [254,19,75],
    [160,185,156],
    [237,125,49],
    [22,217,17],
    [150,182,217],
    [31,102,130],
    ]
SunCG_11_CLASSES = [
    (0,"None"),
    (1,"Ceiling"),
    (2,"Floor"),
    (3,"Wall"),
    (4,"Window"),
    (5,"Chair"),
    (6,"Bad"),
    (7,"Sofa"),
    (8,"Table"),
    (9,"TV"),
    (10,"Furniture"),
    (11,"Objects")
]

def occupancy_meshing(label_num, 
                      volume:torch.Tensor, 
                      volume_label:torch.LongTensor= None, 
                      # label_to_color = None, 
                      threshold = 0, 
                      cube_scale=0.8):
    """
    Parameters
    ----------
    volume : torch.Tensor
        DESCRIPTION.
    volume_label : torch.LongTensor, optional
        DESCRIPTION. The default is None.
    label_to_color : TYPE, optional
        DESCRIPTION. The default is None.
    threshold : TYPE, optional
        DESCRIPTION. The default is 0.
    cube_scale : TYPE, optional
        DESCRIPTION. The default is 0.8.

    Raises
    ------
    Exception
        DESCRIPTION.

    Returns
    -------
    vertices : TYPE
        DESCRIPTION.
    triangles : TYPE
        DESCRIPTION.
    colors : TYPE
        DESCRIPTION.

    """
    # if len(volume.shape) != 3:
    #     raise Exception('Input tensor must be [x,y,z]')
    # X = volume.size(0)
    # Y = volume.size(1)
    # Z = volume.size(2)
    # count = 0
    # index_counter = 0
    # use_color = volume_label is not None
    
    # counter = (volume > threshold).sum()
    # count = counter*8
    # index_counter = counter * 12
    # vertices = torch.FloatTensor(count,3)
    # triangles = torch.IntTensor(index_counter,3)
    # if use_color:
    #     labels = torch.LongTensor(count,1)
    # count=0
    # index_counter = 0
    # for x in range(X):
    #     for y in range(Y):
    #         for z in range(Z):
    #             if volume[x,y,z] > threshold:
    #                 for index in index_offset:
    #                     triangles[index_counter] = torch.IntTensor(index)+ count
    #                     index_counter+=1
    #                 for off in offsets:
    #                     vertices[count] = torch.FloatTensor([x,y,z])+torch.FloatTensor(off)*cube_scale 
    #                     if use_color:
    #                         labels[count] = volume_label[x,y,z]
    #                     count+=1
    # colors = torch.LongTensor(labels.size(0),3)
    # for i in range(colors.size(0)):
    #     colors[i,:] = torch.LongTensor(NYU_13_CLASSEScolour_code[labels[i]])
    if found_mo is not None:
        if label_num is 14:
            t1 = torch.from_numpy(np.array(NYU_13_CLASSEScolour_code))
        elif label_num is 12:
            t1 = torch.from_numpy(np.array(SunCG_11_Color))
        else:
            raise RuntimeError('label num doesn\'t support')
        print('meshing...')
        # vertices, triangles, labels = mo.forward(volume=volume.float().cuda(), label=volume_label.int().cuda(), 
        #                                           threshold=threshold,voxel_scale = cube_scale, use_cuda = True)
        
        vertices, triangles, labels = mo.forward(volume=volume.float().cpu(), label=volume_label.int().cpu(), 
                                                  threshold=threshold,voxel_scale = cube_scale, use_cuda = False)
        print('meshing...done')
        
        print('colouring...')
        color = mo.label2color(labels.cuda(), t1.cuda(), use_cuda = True)
        # color = mo.label2color(labels.cpu(), t1.cpu(), use_cuda = False)
        print('colouring...done')
        
        vertices = vertices.cpu()
        triangles = triangles.cpu()
        colors = color.cpu()
    else:
        print('occupancy_meshing requires meshing_occupancy. Install it from extension: python build.py install')
    
    return vertices, triangles, colors


def write_ply(path, 
              vertices:torch.FloatTensor, 
              triangles:torch.IntTensor=None, 
              colors:torch.LongTensor=None, 
              normals:torch.FloatTensor=None):
    mo.write_ply(path, vertices, triangles, colors, normals)

    # if len(list(vertices.shape)) != 2:
    #     raise Exception('input tensor must be [num,3]')
    # with open(path, 'w+') as f:
    #     f.write('ply\n')
    #     f.write('format ascii 1.0\n')
    #     f.write('element vertex {}\n'.format(vertices.size(0)))
    #     f.write('property float x\n')
    #     f.write('property float y\n')
    #     f.write('property float z\n')
    #     if colors is not None:
    #         f.write('property uchar red\n')
    #         f.write('property uchar green\n')
    #         f.write('property uchar blue\n')
    #     if normals is not None:
    #         f.write('property float nx\n')
    #         f.write('property float ny\n')
    #         f.write('property float nz\n')
    #         f.write('property float curvature\n')
    #     if triangles is not None:
    #         f.write('element face {}\n'.format(triangles.size(0)))
    #     else:
    #         f.write('element face {}\n'.format(math.floor(vertices.size(0)/3)))
    #     f.write('property list uchar int vertex_indices\n')
    #     f.write('end_header\n')
    #     for i in range(vertices.size(0)):
    #         f.write("{} {} {}".format(vertices[i,0],vertices[i,1],vertices[i,2]))
    #         if colors is not None:
    #             f.write(" {} {} {}".format(colors[i,0],colors[i,1],colors[i,2]))
    #         if normals is not None:
    #             f.write(" {} {} {} 1".format(normals[i,0],normals[i,1],normals[i,2]))
    #         f.write('\n')
            
    #     if triangles is not None:
    #         for i in range(triangles.size(0)):
    #             f.write('3 {} {} {}\n'.format(triangles[i,0], triangles[i,1], triangles[i,2]))
    #     else:
    #         for i in range(math.floor(vertices.size(0)/3)):
    #             f.write('3 {} {} {}\n'.format(3*i, 3*i+1,3*i+2))
                
# import importlib
# mcubes_spec = importlib.util.find_spec("mcubes")
# found = mcubes_spec is not None
# if found:
#     import mcubes
#     X, Y, Z = np.mgrid[:4, :4, :4]
#     u = (X-2)**2 + (Y-2)**2 + (Z-2)**2 - 1**2
#     vertices, triangles = mcubes.marching_cubes(u, 0)
#     mcubes.export_obj(vertices, triangles, "sphere.obj")
# else:
#     print('Sampling require mcubes. Install it with  pip install --upgrade PyMCubes')
if __name__ == '__main__':
    import numpy as np
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter()
    
    size = 4
    X, Y, Z = np.mgrid[:size, :size, :size]
    u = (X-2)**2 + (Y-2)**2 + (Z-2)**2 - 1**2
    a = torch.from_numpy(u)
    label = torch.randint(0,size,list(a.size()))
    color = list(np.random.choice(range(256), size=3))
    t1 = torch.from_numpy(np.array(NYU_13_CLASSEScolour_code))
    
    # vertices,triangles, labels= occupancy_meshing(a, label,cube_scale=0.2)
    # colors = torch.LongTensor(labels.size(0),3)
    # for i in range(colors.size(0)):
    #     colors[i,:] = torch.LongTensor(NYU_13_CLASSEScolour_code[labels[i]])
        
    vertices,triangles, color= occupancy_meshing(a, label,cube_scale=0.2)

    # writer.add_mesh('my_mesh', vertices=vertices.unsqueeze(0), colors=colors.unsqueeze(0), faces=triangles.unsqueeze(0))
    
    u = (X-2)**2 + (Y-2)**2 + (Z-2)**2 - 2**2
    a = torch.from_numpy(u)
    # vertices,triangles, labels= occupancy_meshing(a, label,cube_scale=0.8)
    # colors = torch.LongTensor(labels.size(0),3)
    # for i in range(colors.size(0)):
    #     colors[i,:] = torch.LongTensor(NYU_13_CLASSEScolour_code[labels[i]])
    vertices, triangles, colors= occupancy_meshing(a, label,cube_scale=0.8)
    
    # writer.add_mesh('my_mesh2', vertices=vertices.unsqueeze(0), colors=colors.unsqueeze(0), faces=triangles.unsqueeze(0))
    
    # import mcubes
    # mcubes.export_obj(vertices, triangles, "sphere.obj")
    write_ply('test.ply', vertices, triangles, colors)
    
    import meshing_occupancy as mo
    # CPU
    vertices, triangles, labels = mo.forward(volume=a.float().cpu(), label=label.int().cpu(), 
                                             threshold=0,voxel_scale = 0.8, use_cuda = False)
    vertices_cpu = vertices.cpu()#.to(torch.device("cpu"))
    triangles_cpu = triangles.cpu()#.to(torch.device("cpu"))
    labels_cpu = labels.cpu()#.to(torch.device("cpu"))
    
    colors_cpu = mo.label2color(labels.cpu(), t1.cpu(), use_cuda = False)

    write_ply('test_cpu.ply', vertices_cpu, triangles_cpu, colors_cpu)

    # CUDA
    vertices, triangles, labels = mo.forward(volume=a.float().cuda(), label=label.int().cuda(), threshold=0, voxel_scale = 0.8, use_cuda = True)
    vertices_cuda = vertices.cpu()#.to(torch.device("cpu"))
    triangles_cuda = triangles.cpu()#.to(torch.device("cpu"))
    labels_cuda = labels.cpu()#.to(torch.device("cpu"))
    
    colors_cuda = mo.label2color(labels.cuda(), t1.cuda(), use_cuda = True)
    colors_cuda = colors_cuda.cpu()
    write_ply('test_cuda.ply', vertices_cuda, triangles_cuda, colors_cuda)
    
    
    
    # write_ply('test.ply',vertices,triangles,colors)
    