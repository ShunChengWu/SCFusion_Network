//
// Created by Shun-Cheng Wu on 2020-01-12.
//

#ifndef OCCUPANCYMESHING_MESHING_OCCUPANCY_SHARED_H
#define OCCUPANCYMESHING_MESHING_OCCUPANCY_SHARED_H

int classifyVoxel(int size, float* volume, float threshold);

int process (int size, int X, int Y, int Z, float *volume_data, int *in_label_data,
             float *vertices_data, int *faces_data, int *label_data, float threshold, float voxel_scale);

int label2color(int size, int *label, long *color, long *output);
#endif //OCCUPANCYMESHING_MESHING_OCCUPANCY_SHARED_H
