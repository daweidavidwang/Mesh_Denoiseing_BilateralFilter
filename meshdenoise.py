from openmesh import *
import numpy as np
import queue
import math
import copy
import datetime


mesh = TriMesh()

mesh = read_trimesh("Noisy.obj")

mesh_ori = copy.deepcopy(mesh)

iter_num = 10
def getAdaptiveVertexNeighbor(mesh, vh, sigma_c):
    mark = [False for _ in range(mesh.n_vertices())]
    vertex_neighbor = []
    queue_vertex_handle = queue.Queue(maxsize=0)
    mark[vh.idx()] = True
    queue_vertex_handle.put(vh)
    radius = 2.0 * sigma_c
    ci = mesh.point(vh)

    while not queue_vertex_handle.empty():
        vh = queue_vertex_handle.get()
        vertex_neighbor.append(vh)
        for vv_it in mesh.vv(vh):
            vh_neighbor = vv_it
            if(mark[vh_neighbor.idx()] == False):
                cj = mesh.point(vh_neighbor)
                length = np.linalg.norm(cj - ci)
                if(length <= radius):
                    queue_vertex_handle.put(vh_neighbor)
                mark[vh_neighbor.idx()] = True
    return vertex_neighbor


for iter in range(iter_num):
    start_time = datetime.datetime.now() 
    points = []
    mesh.request_face_normals()
    mesh.request_vertex_normals()
    mesh.update_normals()

    sigma_c = 0

    index = 0
    for v_it in mesh.vertices():
        pi = mesh.point(v_it)
        ni = mesh.normal(v_it)
        max = 1e10
        for vv_it in mesh.vv(v_it):
            pj = mesh.point(vv_it)
            length = np.linalg.norm(pi - pj)
            if length < max:
                max = length
        sigma_c = max

        vertex_neighbor = getAdaptiveVertexNeighbor(mesh, v_it, sigma_c)
        average_off_set = 0
        off_set_dis =[]
        for neighbor_vh in vertex_neighbor:
            pj = mesh.point(neighbor_vh)
            t = np.dot((pj - pi),ni)
            t = math.sqrt(t*t)
            average_off_set = average_off_set + t
            off_set_dis.append(t)
        average_off_set = average_off_set / float(len(vertex_neighbor))
        offset = 0
        for j in range(len(off_set_dis)):
            offset += (off_set_dis[j] - average_off_set) * (off_set_dis[j] - average_off_set)
            offset /= float(len(off_set_dis))

        sigma_s = (math.sqrt(offset) + 1.0e-12) if (math.sqrt(offset) < 1.0e-12) else math.sqrt(offset)
        sigma_s = sigma_s*2
        sum = 0 
        normalizer = 0
        for iv in vertex_neighbor:
            pj = mesh.point(iv)
            t = np.linalg.norm(pi - pj)
            h = np.dot((pj - pi),ni)
            wc = math.exp(-t*t/2*(sigma_c *sigma_c))
            ws = math.exp(-h*h/2*(sigma_s *sigma_s))
            sum = sum + (wc * ws * h)
            normalizer = normalizer + (wc * ws)

        points.append(pi + ni * (sum / normalizer))
        index = index + 1
    
    index = 0
    for v_it in mesh.vertices():
        #print(np.linalg.norm(mesh.point(v_it)-points[index]))
        mesh.set_point(v_it, points[index])
        #print(np.linalg.norm(mesh.point(v_it)-points[index]))
        index = index + 1
    
    diff = 0
    for vh_ori,vh_denoised in zip(mesh_ori.vertices(), mesh.vertices()):
        p_ori = mesh_ori.point(vh_ori)
        p_de = mesh.point(vh_denoised)
        diff = diff + np.linalg.norm(p_ori-p_de)
    print("model diff:"+str(diff))
    end_time = datetime.datetime.now() 
    print("iter time:"+str((end_time-start_time).seconds))

write_mesh("denoised.obj", mesh)