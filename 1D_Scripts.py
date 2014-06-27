# -*- coding: utf-8 -*-   

# ##### BEGIN GPL LICENSE BLOCK #####
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 2
#  of the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software Foundation,
#  Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
#
# ##### END GPL LICENSE BLOCK #####  

bl_info = {
    "name": "1D_Scripts",                     
    "author": "Alexander Nedovizin, Paul Kotelevets aka 1D_Inc (concept design)",
    "version": (0, 2, 25),
    "blender": (2, 6, 8),
    "location": "View3D > Toolbar",
    "category": "Mesh"
}  

# http://dl.dropboxusercontent.com/u/59609328/Blender-Rus/1D_Scripts.py

import bpy,bmesh, mathutils, math
from mathutils import Vector


def find_index_of_selected_vertex(mesh):  
    selected_verts = [i.index for i in mesh.vertices if i.select]  
    verts_selected = len(selected_verts)  
    if verts_selected <1:  
        return None                            
    else:  
        return selected_verts  


def find_extreme_select_verts(mesh, verts_idx):
    res_vs = []
    edges = mesh.edges  
 
    for v_idx in verts_idx:
        connecting_edges = [i for i in edges if v_idx in i.vertices[:] and i.select]  
        if len(connecting_edges) == 1: 
            res_vs.append(v_idx)
    return res_vs
    

def find_connected_verts_simple(me, found_index):  
    edges = me.edges  
    connecting_edges = [i for i in edges if found_index in i.vertices[:] and \
        me.vertices[i.vertices[0]].select and me.vertices[i.vertices[1]].select]  
    if len(connecting_edges) == 0: 
        return []
    else:  
        connected_verts = []  
        for edge in connecting_edges:  
            cvert = set(edge.vertices[:])   
            cvert.remove(found_index)                            
            vert = cvert.pop()
            connected_verts.append(vert)  
        return connected_verts  


def find_connected_verts(me, found_index, not_list):  
    edges = me.edges  
    connecting_edges = [i for i in edges if found_index in i.vertices[:]]  
    if len(connecting_edges) == 0: 
        return []
    else:  
        connected_verts = []  
        for edge in connecting_edges:  
            cvert = set(edge.vertices[:])   
            cvert.remove(found_index)                            
            vert = cvert.pop()
            if not (vert in not_list) and me.vertices[vert].select:
                connected_verts.append(vert)  
        return connected_verts  
    
    
def find_all_connected_verts(me, active_v, not_list=[], step=0):
    vlist = [active_v]
    not_list.append(active_v)
    step+=1
    list_v_1 = find_connected_verts(me, active_v, not_list)              

    for v in list_v_1:
        list_v_2 = find_all_connected_verts(me, v, not_list, step) 
        vlist += list_v_2
    return vlist  


def bm_vert_active_get(bm):
    for elem in reversed(bm.select_history):
        if isinstance(elem, (bmesh.types.BMVert, bmesh.types.BMEdge, bmesh.types.BMFace)):
            return elem.index, str(elem)[3:4]     
    return None, None


def to_store_coner(obj_name, bm, mode):
    config = bpy.context.window_manager.paul_manager
    active_edge, el = bm_vert_active_get(bm)
    old_name_c = config.object_name_store_c
    old_coner1 = config.coner_edge1_store 
    old_coner2 = config.coner_edge2_store 
    
    def check():
        if mode=='EDIT_MESH' and \
          (old_name_c != config.object_name_store_c or \
           old_coner1 != config.coner_edge1_store or \
           old_coner2 != config.coner_edge2_store):
           config.flip_match = False
    
    if active_edge != None and el=='E':
        mesh = bpy.data.objects[obj_name].data
        config.object_name_store_c = obj_name
        config.coner_edge1_store = active_edge
        verts = bm.edges[active_edge].verts
        v0 = verts[0].index
        v1 = verts[1].index
        edges_idx = [i.index for i in mesh.edges \
            if (v1 in i.vertices[:] or v0 in i.vertices[:])and i.select \
            and i.index!=active_edge] 
        if edges_idx:
            config.coner_edge2_store = edges_idx[0]
            check()
            return True
        
    if active_edge != None and el=='V':
        mesh = bpy.data.objects[obj_name].data
        config.object_name_store_c = obj_name
        
        v2_l = find_all_connected_verts(mesh, active_edge,[],0)
        control_vs = find_connected_verts_simple(mesh, active_edge)
        if len(v2_l)>2 and len(control_vs)==1:
            v1 = v2_l.pop(1)
            edges_idx = []
            for v2 in v2_l[:2]:
                edges_idx.extend([i.index for i in mesh.edges \
                    if v1 in i.vertices[:] and v2 in i.vertices[:]] )
            
            if len(edges_idx)>1:
                config.coner_edge1_store = edges_idx[0]
                config.coner_edge2_store = edges_idx[1]
                check()
                return True
    
    check()
    config.object_name_store_c = ''
    config.coner_edge1_store = -1
    config.coner_edge2_store = -1
    if mode =='EDIT_MESH':
        config.flip_match = False   
        print_error('Two edges is not detected')
        print('Error: align 05')
    return False


def to_store_vert(obj_name, bm):
    config = bpy.context.window_manager.paul_manager
    active_edge, el = bm_vert_active_get(bm)
    old_edge1 = config.active_edge1_store
    old_edge2 = config.active_edge2_store
    old_name_v = config.object_name_store_v
    
    def check():
        if old_name_v != config.object_name_store_v or \
           old_edge1 != config.active_edge1_store or \
           old_edge2 != config.active_edge2_store:
           config.flip_match = False    
    
    if active_edge != None and el=='E':
        mesh = bpy.data.objects[obj_name].data
        config.object_name_store_v = obj_name
        config.active_edge1_store = active_edge
        verts = bm.edges[active_edge].verts
        v0 = verts[0].index
        v1 = verts[1].index
        edges_idx = [i.index for i in mesh.edges \
            if (v1 in i.vertices[:] or v0 in i.vertices[:])and i.select \
            and i.index!=active_edge] 
        if edges_idx:
            config.active_edge2_store = edges_idx[0]
            check()
            return True
        
    if active_edge != None and el=='V':
        mesh = bpy.data.objects[obj_name].data
        config.object_name_store_v = obj_name
        
        v2_l = find_all_connected_verts(mesh, active_edge,[],0)
        control_vs = find_connected_verts_simple(mesh, active_edge)
        if len(v2_l)>2 and len(control_vs)==1:
            v1 = v2_l.pop(1)
            edges_idx = []
            for v2 in v2_l[:2]:
                edges_idx.extend([i.index for i in mesh.edges \
                    if v1 in i.vertices[:] and v2 in i.vertices[:]] )
                
            if len(edges_idx)>1:
                config.active_edge1_store = edges_idx[0]
                config.active_edge2_store = edges_idx[1]
                check()
                return True
    
    check()
    config.object_name_store_v = ''
    config.active_edge1_store = -1
    config.active_edge2_store = -1
    config.flip_match = False   
    print_error('Side is undefined')
    print('Error: 3dmatch 10')
    return
    
    
def to_store(obj_name, bm):
    config = bpy.context.window_manager.paul_manager
    active_edge, el = bm_vert_active_get(bm)
    if active_edge != None and el=='E':
        config.object_name_store = obj_name
        config.edge_idx_store = active_edge
        verts = bm.edges[active_edge].verts
        config.vec_store = (verts[1].co - verts[0].co) * \
            bpy.data.objects[obj_name].matrix_world.to_3x3().transposed()
        return
    
    if active_edge != None and el=='V':
        obj_act = bpy.context.active_object
        mesh = obj_act.data
        v2_l = find_index_of_selected_vertex(mesh)
        if len(v2_l)==2:
            v1 = active_edge
            v2_l.pop(v2_l.index(v1))
            v2 = v2_l[0]
            edges_idx = [i.index for i in mesh.edges \
                if v1 in i.vertices[:] and v2 in i.vertices[:]] 
                
            if edges_idx:
                config.edge_idx_store = edges_idx[0]

            config.object_name_store = obj_name
            config.vec_store = (mesh.vertices[v1].co - mesh.vertices[v2].co) * \
                bpy.data.objects[obj_name].matrix_world.to_3x3().transposed()
            return
                
    config.object_name_store = ''
    config.edge_idx_store = -1
    config.vec_store = mathutils.Vector((0,0,0))
    print_error('Active edge is not detected')
    print('Error: align 02')


def select_mesh_rot(me, matrix):
    verts = [v for v in me.verts if v.select==True]
    for v in verts:
        v.co = v.co*matrix


def store_align(vts='edge', mode='EDIT_MESH'):
    bpy.ops.object.editmode_toggle()
    bpy.ops.object.editmode_toggle()
    
    obj = bpy.context.active_object
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)  
    result = True
    
    if vts=='vert':
        to_store_vert(obj.name, bm)
    elif vts=='edge':
        to_store(obj.name, bm)
    else:
        # vts=='coner':
        result = to_store_coner(obj.name, bm, mode)
    
    bm.free()   
    return result


def getNormalPlane(vecs, mat):
    if len(vecs)<3:
        return None
    
    out_ = []
    vec_c = mathutils.Vector((0,0,0))
    for v in vecs:
        vec  = v*mat
        out_.append(vec)
        vec_c+=vec
    
    vec_c = vec_c / len(vecs) 
                                       
    v = out_[1]-out_[0]
    w = out_[2]-out_[0]
    A = v.y*w.z - v.z*w.y
    B = -v.x*w.z + v.z*w.x
    C = v.x*w.y - v.y*w.x
    D = -out_[0].x*A - out_[0].y*B - out_[0].z*C   
    
    norm = mathutils.Vector((A,B,C)).normalized()
    return norm


def sign(x):
    if x<0:
        return -1
    else: return 1


def match3D(flip = False):
    mode_ = bpy.context.mode
    store_align('coner', mode_)
    config = bpy.context.window_manager.paul_manager
    if config.object_name_store_v == '' or \
       config.active_edge1_store < 0 or config.active_edge2_store < 0:
        print_error('Stored Vertex is required')
        print('Error: 3dmatch 01')
        return False
    
    if config.object_name_store_c == '':
        if mode_ =='EDIT_MESH':
            print_error('Not specified object')
            print('Error: 3dmatch 02')
            return False
        else:
            config.object_name_store_c = bpy.context.active_object.name
    
    if config.coner_edge1_store == -1 or \
       config.coner_edge2_store == -1:
        if mode_ =='EDIT_MESH':
            #print_error('Not specified object')
            print_error('Stored edges is required')
            print('Error: 3dmatch 03')
            return False
    
    obj_A =  bpy.data.objects[config.object_name_store_v]
    obj_B =  bpy.data.objects[config.object_name_store_c]
    ve1 = obj_A.data.edges[config.active_edge1_store]
    ve2 = obj_A.data.edges[config.active_edge2_store]
    e1 = obj_B.data.edges[config.coner_edge1_store]
    e2 = obj_B.data.edges[config.coner_edge2_store]
    
    # получаем ещё две вершины. Иначе - реджект
    connect_vs = []
    connect_vs.extend(ve1.vertices[:])
    connect_vs.extend(ve2.vertices[:])
    v1 = -1
    for v in connect_vs:
        if connect_vs.count(v)>1:
            v1 = obj_A.data.vertices[v]
            connect_vs.pop(connect_vs.index(v))
            connect_vs.pop(connect_vs.index(v))
            break
    
    if v1 == -1:
        print_error('Active vertex of object_A must have two edges')
        print('Error: 3dmatch 04')
        return False
    
    v2 = obj_A.data.vertices[connect_vs[0]]
    v3 = obj_A.data.vertices[connect_vs[1]]
    
    # вычислить нормаль объекта Б
    if mode_ =='EDIT_MESH':
        lws = list(e1.vertices[:]+e2.vertices[:])
        for l in lws:
            if lws.count(l)>1: 
                lws.pop(lws.index(l))
                w1 = obj_B.data.vertices[lws.pop(lws.index(l))]
        
        w3 = obj_B.data.vertices[lws.pop()]
        w2 = obj_B.data.vertices[lws.pop()]
    else:
        w1,w2,w3 = 0,0,0
    
    mat_w = obj_B.matrix_world.copy()
    k_x = 1
    if mode_ !='EDIT_MESH':
        if config.flip_match: k_x = -1
        else: k_x = 1
        
    if flip!=config.flip_match:
        config.flip_match = flip
        if mode_ =='EDIT_MESH':
            bpy.ops.object.mode_set(mode='EDIT') 
            normal_B = getNormalPlane([w1.co, w2.co, w3.co], mathutils.Matrix())
            normal_z = mathutils.Vector((0,0,1))
            mat_rot_norm = normal_B.rotation_difference(normal_z).to_matrix().to_4x4()
           
            verts = [v for v in obj_B.data.vertices if v.select==True]
            for v in verts:
                v.co = mat_rot_norm * v.co
            
            bpy.ops.transform.resize(value=(1,1,-1), constraint_axis=(False, False, True))
        else:
            k_x *= -1
    
    normal_x = mathutils.Vector((1,0,0)) * k_x
        
    bpy.ops.object.mode_set(mode='EDIT') 
    bpy.ops.object.mode_set(mode='OBJECT')    
    edge_idx = [i.index for i in obj_A.data.edges \
        if v1 in i.vertices[:] and v2 in i.vertices[:]] 
            
    vecA= (v2.co - v1.co) * obj_A.matrix_world.to_3x3().transposed()
    
    if mode_ =='EDIT_MESH':
        v1A = obj_A.matrix_world * v1.co 
        w1B = obj_B.matrix_world * w1.co 

        vecB = (w2.co - w1.co)
        mat_rot = vecB.rotation_difference(vecA).to_matrix().to_4x4()
        
        # rotation1
        bpy.ops.object.mode_set(mode='EDIT') 
        bpy.ops.object.mode_set(mode='OBJECT')
        
        normal_A = getNormalPlane([v1.co, v2.co, v3.co], mathutils.Matrix())
        normal_A = normal_A * obj_A.matrix_world.to_3x3().transposed()
        normal_B = getNormalPlane([w1.co, w2.co, w3.co], mathutils.Matrix())
        mat_rot2 = normal_B.rotation_difference(normal_A).to_matrix().to_4x4()
        
        verts = [v for v in obj_B.data.vertices if v.select==True]
        for v in verts:
            v.co = mat_rot2 * v.co 
        
        
        bpy.ops.object.mode_set(mode='EDIT') 
        bpy.ops.object.mode_set(mode='OBJECT')
        
        vecA= (v2.co - v1.co) * obj_A.matrix_world.to_3x3().transposed()
        vecB = (w2.co - w1.co)
        mat_rot = vecB.rotation_difference(vecA).to_matrix().to_4x4()
        verts = [v for v in obj_B.data.vertices if v.select==True]
        for v in verts:
            v.co = mat_rot * v.co
        
        
        # invert rotation
        bpy.ops.object.mode_set(mode='EDIT') 
        bpy.ops.object.mode_set(mode='OBJECT')
        
        vec1 = mathutils.Vector((0,0,1))
        vec2 = obj_B.matrix_world * vec1
        mat_rot2 = vec1.rotation_difference(vec2).to_matrix().to_4x4()
        mat_tmp = obj_B.matrix_world.copy()
        
        mat_tmp[0][3]=0
        mat_tmp[1][3]=0
        mat_tmp[2][3]=0
        mat_inv = mat_tmp.inverted()
        
        verts = [v for v in obj_B.data.vertices if v.select==True]
        for v in verts:
            v.co = mat_inv * v.co
        
        # location
        bpy.ops.object.mode_set(mode='EDIT') 
        bpy.ops.object.mode_set(mode='OBJECT')
        
        w1B = obj_B.matrix_world * w1.co
        mat_loc = mathutils.Matrix.Translation(v1A-w1B)
        vec_l = mat_inv * (v1A-w1B)
        
        mat_tp = obj_B.matrix_world
        vec_loc = mathutils.Vector((mat_tp[0][3],mat_tp[1][3],mat_tp[2][3]))
        
        verts = [v for v in obj_B.data.vertices if v.select==True]
        for v in verts:
            v.co = v.co + vec_l 
            
        bpy.ops.object.mode_set(mode='EDIT') 
        
    else:
        v1A = obj_A.matrix_world * v1.co
        normal_A = getNormalPlane([v1.co, v2.co, v3.co], mathutils.Matrix())
        normal_A = normal_A * obj_A.matrix_world.to_3x3().transposed()
        normal_z = mathutils.Vector((0,0,1))
        mat_rot1 = normal_z.rotation_difference(normal_A).to_matrix().to_4x4()
        
        vecA = (v2.co - v1.co) * obj_A.matrix_world.to_3x3().transposed()
        vecB = mat_rot1 * normal_x
        mat_rot = vecB.rotation_difference(vecA).to_matrix().to_4x4()
        
        obj_B.matrix_world = mat_rot * mat_rot1
        vec_l = v1A-obj_B.location
        obj_B.location = obj_B.location+vec_l
    '''
    config.variant += 1
    if config.variant>3:
        config.variant = 0'''


def main_align_object(axe='X',project='XY'):
    #print('axe,project',axe,project)
    #bpy.ops.object.mode_set(mode='OBJECT')
    obj_res = bpy.context.active_object
    if obj_res.type=='MESH':
        bpy.ops.object.mode_set(mode='EDIT') 
        bpy.ops.object.mode_set(mode='OBJECT')
    
    config = bpy.context.window_manager.paul_manager
    if config.object_name_store == '':
        print_error('Stored Edge is required')
        print('Error: align 01')
        return False
    
    obj = bpy.data.objects[config.object_name_store]
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)  
    
    # Найдём диагональ Store
    edge_idx = config.edge_idx_store
    verts_edge_store = bm.edges[edge_idx].verts
    vec_diag_store = verts_edge_store[1].co - verts_edge_store[0].co
    
    #obj_res = bpy.context.active_object
    # Развернем объект
    dict_axe = {'X':(1.0,0.0,0.0), 'Y':(0.0,1.0,0.0), 'Z':(0.0,0.0,1.0)}
    aa_vec = dict_axe[axe]
    
    aa = mathutils.Vector(aa_vec) 
    bb = vec_diag_store.normalized()
    
    planes = set(project)
    if 'X' not in planes:
        aa.x=0
        bb.x=0
    if 'Y' not in planes:
        aa.y=0
        bb.y=0
    if 'Z' not in planes:
        aa.z=0
        bb.z=0        

    vec = aa
    q_rot = vec.rotation_difference(bb).to_matrix().to_4x4()
    obj_res.matrix_world *= q_rot
    for obj in bpy.context.scene.objects:
        if obj.select:
            if obj.name!=obj_res.name:
                orig_tmp = obj_res.location-obj.location
                mat_loc = mathutils.Matrix.Translation(orig_tmp)
                mat_loc2 = mathutils.Matrix.Translation(-orig_tmp)
        
                obj.matrix_world *= mat_loc*q_rot*mat_loc2
    return
    
  


def main_align():
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.mode_set(mode='EDIT') 
    
    config = bpy.context.window_manager.paul_manager
    if config.object_name_store == '':
        print_error('Stored Edge is required')
        print('Error: align 01')
        return False
    
    obj = bpy.data.objects[config.object_name_store]
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)  
    
    # Найдём диагональ Store
    edge_idx = config.edge_idx_store
    verts_edge_store = bm.edges[edge_idx].verts
    vec_diag_store = verts_edge_store[1].co - verts_edge_store[0].co
    
    # Получим выделенное ребро
    obj_res = bpy.context.active_object
    mesh_act = obj_res.data
    bm_act = bmesh.new()
    bm_act.from_mesh(mesh_act)  
    
    edge_idx_act, el = bm_vert_active_get(bm_act)
    if edge_idx_act == None:
        print_error('Selection with active edge is required')
        print('Error: align 03')
        return False
    
    d_pos = bpy.context.scene.cursor_location - obj_res.location
    if not config.align_dist_z:  
        for v in bm_act.verts:
            if v.select:
                v.co -= d_pos
        
    
    verts_edge_act = bm_act.edges[edge_idx_act].verts
    vec_diag_act = verts_edge_act[1].co - verts_edge_act[0].co
    
    # Сравниваем
    aa = vec_diag_act 
    if config.align_lock_z:
        aa.z = 0
    aa.normalized()
    
    bb = vec_diag_store
    if config.align_lock_z:
        bb.z = 0
    bb.normalized()
    q_rot = bb.rotation_difference(aa).to_matrix().to_4x4()
    
    select_mesh_rot(bm_act, q_rot)
    verts = [v for v in bm_act.verts if v.select==True]
    pos = (verts_edge_store[0].co + obj.location)\
        - (verts_edge_act[0].co + obj_res.location)
        
    if not config.align_dist_z:
        pos = mathutils.Vector((0,0,0)) #bpy.context.scene.cursor_location
    for v in verts:
        pos_z = v.co.z
        v.co = v.co + pos
        if config.align_lock_z:
            v.co.z = pos_z
    
    if not config.align_dist_z:    
        for v in bm_act.verts:
            if v.select:
                v.co += d_pos
            
    bpy.ops.object.mode_set(mode='OBJECT')
    
    bm_act.to_mesh(mesh_act)
    bm_act.free()
    
    bm.free()
    
    bpy.ops.object.mode_set(mode='EDIT') 
    return True
        

def main_spread(context, mode):
    bpy.ops.object.mode_set(mode='OBJECT') 
    bpy.ops.object.mode_set(mode='EDIT') 
    
    obj = bpy.context.active_object
    me = obj.data

    verts = find_index_of_selected_vertex(me)
    cou_vs = len(verts) - 1
    if verts != None and cou_vs>0:
        extreme_vs = find_extreme_select_verts(me, verts)
        
        if len(extreme_vs) != 2:
            print_error('Single Loop only')
            print('Error: 01')
            return False
        
        list_koeff = []
        
        if mode[0]:
            min_v = min([me.vertices[extreme_vs[0]].co.x,extreme_vs[0]], \
                        [me.vertices[extreme_vs[1]].co.x,extreme_vs[1]])
            max_v = max([me.vertices[extreme_vs[0]].co.x,extreme_vs[0]], \
                        [me.vertices[extreme_vs[1]].co.x,extreme_vs[1]])

            if (max_v[0]-min_v[0]) == 0:
                min_v = [me.vertices[extreme_vs[0]].co.x,extreme_vs[0]]
                max_v = [me.vertices[extreme_vs[1]].co.x,extreme_vs[1]]
            
            sort_list = find_all_connected_verts(me,min_v[1],[])
            
            if len(sort_list) != len(verts):
                print_error('Incoherent loop')
                print('Error: 020')
                return False
            
            step = []
            if mode[3]:
                list_length = []
                sum_length = 0.0
                x_sum = 0.0
                for sl in range(cou_vs):
                    subb = me.vertices[sort_list[sl+1]].co-me.vertices[sort_list[sl]].co
                    length = subb.length
                    sum_length += length
                    list_length.append(sum_length)
                    x_sum += subb.x
                
                for sl in range(cou_vs):
                    tmp = list_length[sl]/sum_length
                    list_koeff.append(tmp)
                    step.append(x_sum * tmp)
            else:
                diap = (max_v[0]-min_v[0])/cou_vs
                for sl in range(cou_vs):
                    step.append((sl+1)*diap)
            
            bpy.ops.object.mode_set(mode='OBJECT') 
            for idx in range(cou_vs):
                me.vertices[sort_list[idx+1]].co.x = me.vertices[sort_list[0]].co.x  + step[idx]

            bpy.ops.object.mode_set(mode='EDIT')  
            
        if mode[1]:
            min_v = min([me.vertices[extreme_vs[0]].co.y,extreme_vs[0]], \
                        [me.vertices[extreme_vs[1]].co.y,extreme_vs[1]])
            max_v = max([me.vertices[extreme_vs[0]].co.y,extreme_vs[0]], \
                        [me.vertices[extreme_vs[1]].co.y,extreme_vs[1]])

            if (max_v[0]-min_v[0]) == 0:
                min_v = [me.vertices[extreme_vs[0]].co.y,extreme_vs[0]]
                max_v = [me.vertices[extreme_vs[1]].co.y,extreme_vs[1]]
            
            sort_list = find_all_connected_verts(me,min_v[1],[])
            if len(sort_list) != len(verts):
                print_error('Incoherent loop')
                print('Error: 021')
                return False

            step = []
            if mode[3]:
                list_length = []
                sum_length = 0.0
                y_sum = 0.0
                if len(list_koeff)==0:
                    for sl in range(cou_vs):
                        subb = me.vertices[sort_list[sl+1]].co-me.vertices[sort_list[sl]].co
                        length = subb.length
                        sum_length += length
                        list_length.append(sum_length)
                        y_sum += subb.y
                    
                    for sl in range(cou_vs):
                        tmp = list_length[sl]/sum_length
                        list_koeff.append(tmp)
                        step.append(y_sum * tmp)
                else:
                    for sl in range(cou_vs):
                        subb = me.vertices[sort_list[sl+1]].co-me.vertices[sort_list[sl]].co
                        y_sum += subb.y
                        tmp = list_koeff[sl]
                        step.append(y_sum * tmp)
                    
            else:
                diap = (max_v[0]-min_v[0])/cou_vs
                for sl in range(cou_vs):
                    step.append((sl+1)*diap)

            bpy.ops.object.mode_set(mode='OBJECT') 
            for idx in range(cou_vs):
                me.vertices[sort_list[idx+1]].co.y = me.vertices[sort_list[0]].co.y  + step[idx]

            bpy.ops.object.mode_set(mode='EDIT')  
            
        if mode[2]:
            min_v = min([me.vertices[extreme_vs[0]].co.z,extreme_vs[0]], \
                        [me.vertices[extreme_vs[1]].co.z,extreme_vs[1]])
            max_v = max([me.vertices[extreme_vs[0]].co.z,extreme_vs[0]], \
                        [me.vertices[extreme_vs[1]].co.z,extreme_vs[1]])

            if (max_v[0]-min_v[0]) == 0:
                min_v = [me.vertices[extreme_vs[0]].co.z,extreme_vs[0]]
                max_v = [me.vertices[extreme_vs[1]].co.z,extreme_vs[1]]
            
            sort_list = find_all_connected_verts(me,min_v[1],[])
            if len(sort_list) != len(verts):
                print_error('Incoherent loop')
                print('Error: 022')
                return False
            
            step = []
            if mode[3]:
                list_length = []
                sum_length = 0.0
                z_sum = 0.0
                if len(list_koeff)==0:
                    for sl in range(cou_vs):
                        subb = me.vertices[sort_list[sl+1]].co-me.vertices[sort_list[sl]].co
                        length = subb.length
                        sum_length += length
                        list_length.append(sum_length)
                        z_sum += subb.z
                    
                    for sl in range(cou_vs):
                        step.append(z_sum * list_length[sl]/sum_length)
                else:
                    for sl in range(cou_vs):
                        subb = me.vertices[sort_list[sl+1]].co-me.vertices[sort_list[sl]].co
                        z_sum += subb.z
                        tmp = list_koeff[sl]
                        step.append(z_sum * tmp)
            else:
                diap = (max_v[0]-min_v[0])/cou_vs
                for sl in range(cou_vs):
                    step.append((sl+1)*diap)
            
            bpy.ops.object.mode_set(mode='OBJECT') 
            for idx in range(cou_vs):
                me.vertices[sort_list[idx+1]].co.z = me.vertices[sort_list[0]].co.z  + step[idx]

            bpy.ops.object.mode_set(mode='EDIT')  
            
    return True


def main_ss(context):
    obj = bpy.context.active_object
    me = obj.data
    
    bpy.ops.object.mode_set(mode='OBJECT') 
    bpy.ops.object.mode_set(mode='EDIT') 
    
    vs_idx = find_index_of_selected_vertex(me)
    if vs_idx:
        x_coos = [v.co.x for v in me.vertices if v.index in vs_idx]
        y_coos = [v.co.y for v in me.vertices if v.index in vs_idx]
        
        min_x = min(x_coos)
        max_x = max(x_coos)
        
        min_y = min(y_coos)
        max_y = max(y_coos)
        
        len_x = max_x-min_x
        len_y = max_y-min_y
        
        if len_y<len_x:
            bpy.ops.transform.resize(value=(1,0,1), constraint_axis=(False,True,False))
        else:
            bpy.ops.transform.resize(value=(0,1,1), constraint_axis=(True,False,False))


def main_offset(x):
    mode_obj=bpy.context.mode=='OBJECT'
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.mode_set(mode='EDIT') 
    
    config = bpy.context.window_manager.paul_manager
    if config.object_name_store == '':
        print_error('Stored Edge is required')
        print('Error: offset 01')
        return False
    
    obj = bpy.context.active_object
    obj_edge = bpy.data.objects[config.object_name_store]
    if obj:
        vec = mathutils.Vector(config.vec_store)
        
        if vec.length != 0:
            vec.normalize()
            vec *= x
        me = obj.data
        
        if not mode_obj:
            bm_act = bmesh.new()
            bm_act.from_mesh(me) 
            
            verts_act = find_index_of_selected_vertex(me)
            vec = vec * obj.matrix_local
            for v_idx in verts_act:
                if not config.shift_lockX:
                    bm_act.verts[v_idx].co.x += vec.x
                if not config.shift_lockY:
                    bm_act.verts[v_idx].co.y += vec.y
                if not config.shift_lockZ:
                    bm_act.verts[v_idx].co.z += vec.z
                
            bpy.ops.object.mode_set(mode='OBJECT')
            bm_act.to_mesh(me)
            bm_act.free()
            bpy.ops.object.mode_set(mode='EDIT') 
        else:
            bpy.ops.object.mode_set(mode='OBJECT')
            if config.shift_local:
                vec=vec*obj.matrix_world
            if not config.shift_lockX:
                if config.shift_local:
                    mat_loc = mathutils.Matrix.Translation((vec.x,0,0))
                else:
                    obj.location.x += vec.x
                    
            if not config.shift_lockY:
                if config.shift_local:
                    mat_loc = mathutils.Matrix.Translation((0,vec.y,0))
                else:
                    obj.location.y += vec.y
                    
            if not config.shift_lockZ:
                if config.shift_local:
                    mat_loc = mathutils.Matrix.Translation((0,0,vec.z))
                else:
                    obj.location.z += vec.z
                    
            if config.shift_local:
                obj.matrix_world*=mat_loc
                
                
def GetDistToCursor():
    mode = bpy.context.mode
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.mode_set(mode='EDIT')
    obj = bpy.context.active_object
    if obj:
        d_pos = bpy.context.scene.cursor_location - obj.location
        center = mathutils.Vector((0,0,0))
        
        if mode=='EDIT_MESH':
            me = obj.data
            mode = 'EDIT'
            bm = bmesh.new()
            bm.from_mesh(me) 
            elem, el = bm_vert_active_get(bm)
            if elem != None:
                if el=='V' and bm.verts[elem].select:
                    center = bm.verts[elem].co
                    #print('VERT')
                elif el=='E':
                    center = mathutils.Vector(bm.edges[elem].verts[1].co+bm.edges[elem].verts[0].co) / 2
                    #print('EDGE')
                elif el=='F':
                    center = bm.faces[elem].calc_center_median()
                    #print('FACE')
                center = center * obj.matrix_world.to_3x3().transposed()
    bpy.ops.object.mode_set(mode=mode)    
    return d_pos - center
        

def GetStoreVecLength():
    config = bpy.context.window_manager.paul_manager
    if config.object_name_store == '':
        print_error('Stored Edge is required')
        print('Error: offset 01')
        return False
    
    vec = mathutils.Vector(config.vec_store)
    return vec.length


def crosspols():
    config = bpy.context.window_manager.paul_manager
    obj = bpy.context.active_object
    if obj.type != 'MESH':
        return
    
    bpy.ops.object.mode_set(mode='OBJECT')  
    bpy.ops.object.mode_set(mode='EDIT')  
    bpy.ops.mesh.select_mode(type='FACE') 
    
    me = obj.data
    
    
    bm = bmesh.new()
    bm.from_mesh(me)
    
    P1 = me.polygons[bm.faces.active.index]
    pols = [p.index for p in me.polygons if p.select and p.index!= P1.index]
    sel_edges = []
    sel_verts = []
    vts_all = [v for v in bm.verts if v.select and v.index not in P1.vertices]
    eds_all = [e for e in bm.edges if e.select and e.verts[0].index not in P1.vertices \
                                               and e.verts[1].index not in P1.vertices]
    
    if not config.filter_verts_top and not config.filter_verts_bottom and not config.filter_edges:
        p1_co = me.vertices[P1.vertices[0]].co
        p1_no = P1.normal
        for pol in pols:
            P2 = me.polygons[pol]
            p2_co = me.vertices[P2.vertices[0]].co
            p2_no = P2.normal
            
            cross_line = mathutils.geometry.intersect_plane_plane(p1_co, p1_no, p2_co, p2_no)
            points = []
            split_ed = []
            for idx, edg in enumerate(P2.edge_keys):
                pt_a = me.vertices[edg[0]].co
                pt_b = me.vertices[edg[1]].co
                cross_pt = mathutils.geometry.intersect_line_plane(pt_a, pt_b, p1_co, p1_no)
                if cross_pt:
                    pose_pt = mathutils.geometry.intersect_point_line(cross_pt, pt_a, pt_b)
                    if pose_pt[1]<=1 and pose_pt[1]>=0:
                        points.append(pose_pt[0])
                        split_ed.append(idx)
                        
                
            if len(points)==2:
                bpy.ops.mesh.select_mode(type='VERT') 
                if not config.SPLIT:
                    v1=bm.verts.new(points[0])
                    v2=bm.verts.new(points[1])
                    bm.verts.index_update() 
                    edge = (v1,v2)
                    edg_i = bm.edges.new(edge)
                    sel_edges.append(edg_i)
                else:
                    """ Функция позаимствована из адона Сверчок нод Bisect """
                    verts4cut = vts_all
                    edges4cut = eds_all
                    faces4cut = [fa for fa in bm.faces if fa.index in pols]
                    edges4cut_idx = [ed.index for ed in eds_all]
                    
                    geom_in = verts4cut + edges4cut + faces4cut
                    res = bmesh.ops.bisect_plane(bm, geom=geom_in, dist=0.00001,
                                                 plane_co=p1_co, plane_no=p1_no, use_snap_center=False,
                                                 clear_outer=config.outer_clear, clear_inner=config.inner_clear)
                    
                    fres = bmesh.ops.edgenet_prepare(bm, edges=[e for e in res['geom_cut']
                                                                if isinstance(e, bmesh.types.BMEdge)])
                    
                    sel_edges = [e for e in fres['edges'] if e.index not in edges4cut_idx]
                    
                    # this needs work function with solid gemometry
                    if config.fill_cuts:
                        fres = bmesh.ops.edgenet_prepare(bm, edges=[e for e in res['geom_cut']
                                                                    if isinstance(e, bmesh.types.BMEdge)])
                        bmesh.ops.edgeloop_fill(bm, edges=fres['edges'])

                    bm.verts.index_update()
                    bm.edges.index_update()
                    bm.faces.index_update()
                    break
           
    if config.filter_verts_top or config.filter_verts_bottom:
        bpy.ops.mesh.select_mode(type='VERT') 
        p1_co = me.vertices[P1.vertices[0]].co
        p1_no = P1.normal
        for v in vts_all:
            res = mathutils.geometry.distance_point_to_plane(v.co, p1_co, p1_no)
            if res>=0:
                if config.filter_verts_top:
                    sel_verts.append(v)
            else:
                if config.filter_verts_bottom:
                    sel_verts.append(v)
            
    if config.filter_edges and not config.filter_verts_top and not config.filter_verts_bottom:
        bpy.ops.mesh.select_mode(type='EDGE') 
        p1_co = me.vertices[P1.vertices[0]].co
        p1_no = P1.normal
        print(eds_all)
        for idx, edg in enumerate(eds_all):
            pt_a = edg.verts[0].co
            pt_b = edg.verts[1].co
            cross_pt = mathutils.geometry.intersect_line_plane(pt_a, pt_b, p1_co, p1_no)
            if cross_pt:
                pose_pt = mathutils.geometry.intersect_point_line(cross_pt, pt_a, pt_b)
                if pose_pt[1]<=1 and pose_pt[1]>=0:
                    sel_edges.append(edg)
            
    bm.edges.index_update()
    for v in bm.verts:
        v.select_set(False)
        bm.select_flush(False)
    for ed in sel_edges:
        ed.select=True
    for ed in sel_verts:
        ed.select=True
        
    bpy.ops.object.mode_set(mode='OBJECT') 
    bm.to_mesh(me)       
    me.update()   
    bm.free()
    bpy.ops.object.mode_set(mode='EDIT')  


class LayoutSSPanel(bpy.types.Panel):
    def axe_select(self, context):
        axes = ['X','Y','Z']
        return [tuple(3 * [axe]) for axe in axes]
    
    def project_select(self, context):
        projects = ['XY','XZ','YZ','XYZ']
        return [tuple(3 * [proj]) for proj in projects]
    
    bl_label = "1D_Scripts"
    bl_idname = "Paul_Operator"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'TOOLS'
    bl_category = '1D'
    #bl_context = "mesh_edit"
    bl_options = {'DEFAULT_CLOSED'}  
    
    bpy.types.Scene.AxesProperty = bpy.props.EnumProperty(items=axe_select)
    bpy.types.Scene.ProjectsProperty = bpy.props.EnumProperty(items=project_select)
    
    @classmethod
    def poll(cls, context):
        #return context.active_object is not None and context.mode == 'EDIT_MESH'
        return context.active_object is not None

    def draw(self, context):
        lt = bpy.context.window_manager.paul_manager
        
        layout = self.layout
        col = layout.column(align=True)
        col.operator("mesh.simple_scale_operator", text='XYcollapse')
        
        split = col.split(percentage=0.15)
        if lt.display:
            split.prop(lt, "display", text="", icon='DOWNARROW_HLT')
        else:
            split.prop(lt, "display", text="", icon='RIGHTARROW')

        spread_op = split.operator("mesh.spread_operator", text = 'Spread Loop')
            
        if lt.display:
            box = col.column(align=True).box().column()
            col_top = box.column(align=True)
            row = col_top.row(align=True)
            row.prop(lt, 'spread_x', text = 'Spread X')
            row = col_top.row(align=True)
            row.prop(lt, 'spread_y', text = 'Spread Y')
            row = col_top.row(align=True)
            row.prop(lt, 'spread_z', text = 'Spread Z')
            row = col_top.row(align=True)
            row = col_top.row(align=True)
            row.prop(lt, 'relation', text = 'Relation')
        
        split = col.split()
        if lt.display_align:
            split.prop(lt, "display_align", text="Aligner", icon='DOWNARROW_HLT')
        else:
            split.prop(lt, "display_align", text="Aligner", icon='RIGHTARROW')
            
        
        if lt.display_align and context.mode == 'EDIT_MESH':
            box = col.column(align=True).box().column()
            col_top = box.column(align=True)
            row = col_top.row(align=True)
            row.operator("mesh.align_operator", text = 'Store Edge').type_op = 1
            row = col_top.row(align=True)
            align_op = row.operator("mesh.align_operator", text = 'Align').type_op = 0
            row = col_top.row(align=True)
            row.prop(lt, 'align_dist_z', text = 'Superpose')
            row = col_top.row(align=True)
            row.prop(lt, 'align_lock_z', text = 'lock Z')

        if lt.display_align and context.mode == 'OBJECT':
            box = col.column(align=True).box().column()
            col_top = box.column(align=True)
            row = col_top.row(align=True)
            row.operator("mesh.align_operator", text = 'Store Edge').type_op = 1
            row = col_top.row(align=True)
            align_op = row.operator("mesh.align_operator", text = 'Align').type_op = 2
            row = col_top.row(align=True)
            row.prop(context.scene,'AxesProperty', text = 'Axis')
            row = col_top.row(align=True)
            row.prop(context.scene,'ProjectsProperty', text = 'Projection')
        
        split = col.split()
        if lt.display_offset:
            split.prop(lt, "display_offset", text="SideShift", icon='DOWNARROW_HLT')
        else:
            split.prop(lt, "display_offset", text="SideShift", icon='RIGHTARROW')
        
        if lt.display_offset:
            box = col.column(align=True).box().column()
            col_top = box.column(align=True)
            row = col_top.row(align=True)
            row.operator("mesh.align_operator", text = 'Store dist').type_op = 1
            row = col_top.row(align=True)
            row.operator("mesh.offset_operator", text = 'Active » Cursor').type_op = 3
            
            row = col_top.row(align=True)
            lockX_op = row.prop(lt,"shift_lockX", text="X", icon='FREEZE')
            lockY_op = row.prop(lt,"shift_lockY", text="Y", icon='FREEZE')
            lockZ_op = row.prop(lt,"shift_lockZ", text="Z", icon='FREEZE')
            row = col_top.row(align=True)
            row.prop(lt,"shift_local", text="Local")
            
            row = col_top.row(align=True)
            split = col_top.split(percentage=0.76)
            split.prop(lt,'step_len', text = 'dist')
            getlenght_op = split.operator("mesh.offset_operator", text="Get dist").type_op = 1
            row = col_top.row(align=True)
            split = col_top.split(percentage=0.5)
            left_op = split.operator("mesh.offset_operator", text="", icon='TRIA_LEFT')
            left_op.type_op = 0
            left_op.sign_op = -1
            right_op = split.operator("mesh.offset_operator", text="", icon='TRIA_RIGHT')
            right_op.type_op = 0
            right_op.sign_op = 1
            row = col_top.row(align=True)
            if context.mode == 'EDIT_MESH':
                row.prop(lt,"shift_copy", text="Copy")
            else:
                row.prop(lt, "instance", text='Instance')
                row = col_top.row(align=True)
                row.prop(lt,"shift_copy", text="Copy")
                            
          
        split = col.split()
        if lt.display_3dmatch:
            split.prop(lt, "display_3dmatch", text="3D Match", icon='DOWNARROW_HLT')
        else:
            split.prop(lt, "display_3dmatch", text="3D Match", icon='RIGHTARROW')
        
        if lt.display_3dmatch:
            box = col.column(align=True).box().column()
            col_top = box.column(align=True)
            row = col_top.row(align=True)
            row.operator("mesh.align_operator", text = 'Store key').type_op = 3
            row = col_top.row(align=True)
            split = row.split(0.33, True)
            split.scale_y = 1.5
            split.operator("mesh.align_operator", text = 'Flip').type_op = 6
            split.operator("mesh.align_operator", text = '3D Match').type_op = 5
        
        split = col.split()
        if lt.disp_cp:
            split.prop(lt, "disp_cp", text="Polycross", icon='DOWNARROW_HLT')
        else:
            split.prop(lt, "disp_cp", text="Polycross", icon='RIGHTARROW')
        
        if lt.disp_cp:
            box = col.column(align=True).box().column()
            col_top = box.column(align=True)
            row = col_top.row(align=True)
            split = row.split()
            if lt.disp_cp_project:
                split.prop(lt, "disp_cp_project", text="Project active", icon='DOWNARROW_HLT')
            else:
                split.prop(lt, "disp_cp_project", text="Project active", icon='RIGHTARROW')
            
            if lt.disp_cp_project:
                row = col_top.row(align=True)
                split = row.split(0.5, True)
                split.scale_y = 1.5
                split.operator("mesh.polycross", text = 'Section').type_op = 0 # section
                split.operator("mesh.polycross", text = 'Cut').type_op = 1 # cross
                row = col_top.row(align=True)
                row.prop(lt,"fill_cuts", text="fill cut")
                row = col_top.row(align=True)
                row.prop(lt,"outer_clear", text="remove front")
                row = col_top.row(align=True)
                row.prop(lt,"inner_clear", text="remove bottom")
                
            row = col_top.row(align=True)
            split = row.split()
            if lt.disp_cp_filter:
                split.prop(lt, "disp_cp_filter", text="Selection Filter", icon='DOWNARROW_HLT')
            else:
                split.prop(lt, "disp_cp_filter", text="Selection Filter", icon='RIGHTARROW')
            
            if lt.disp_cp_filter:
                row = col_top.row(align=True)
                row.scale_y = 1.5
                row.operator("mesh.polycross", text = 'to SELECT').type_op = 0 # section
                row = col_top.row(align=True)
                row.prop(lt,"filter_edges", text="Filter Edges")
                row = col_top.row(align=True)
                row.prop(lt,"filter_verts_top", text="Filter Top")
                row = col_top.row(align=True)
                row.prop(lt,"filter_verts_bottom", text="Filter Bottom")
            
        
class SSOperator(bpy.types.Operator):
    """Tooltip"""
    bl_idname = "mesh.simple_scale_operator"
    bl_label = "SScale operator"
    bl_options = {'REGISTER', 'UNDO'} 
    
    @classmethod
    def poll(cls, context):
        return context.active_object is not None

    def execute(self, context):
        main_ss(context)
        return {'FINISHED'}


class CrossPolsOperator(bpy.types.Operator):
    bl_idname = "mesh.polycross"
    bl_label = "Polycross"
    bl_options = {'REGISTER', 'UNDO'} 
    
    type_op = bpy.props.IntProperty(name = 'type_op', default = 0, options = {'HIDDEN'})
    
    @classmethod
    def poll(cls, context):
        return context.active_object is not None

    def execute(self, context):
        lt = bpy.context.window_manager.paul_manager
        if self.type_op == 0:
            lt.SPLIT = False
        elif self.type_op == 1:
            lt.SPLIT = True
        
        crosspols()
        return {'FINISHED'}


class SpreadOperator(bpy.types.Operator):
    """Tooltip"""
    bl_idname = "mesh.spread_operator"
    bl_label = "Spread operator"
    bl_options = {'REGISTER', 'UNDO'} 
    
    @classmethod
    def poll(cls, context):
        return context.active_object is not None

    def execute(self, context):
        config = bpy.context.window_manager.paul_manager
        if main_spread(context, (config.spread_x, config.spread_y, config.spread_z, config.relation)):
            pass
            #print('spread complete')
        return {'FINISHED'}


class AlignOperator(bpy.types.Operator):
    bl_idname = "mesh.align_operator"
    bl_label = "Align operator"
    bl_options = {'REGISTER', 'UNDO'} 
    
    type_op = bpy.props.IntProperty(name = 'type_op', default = 0, options = {'HIDDEN'})
    
    @classmethod
    def poll(cls, context):
        return context.active_object is not None

    def execute(self, context):
        if self.type_op==1:
            store_align()
            config = bpy.context.window_manager.paul_manager
            config.step_len = GetStoreVecLength()
        elif self.type_op==0:
            main_align()
        elif self.type_op==2:
            scene = bpy.context.scene
            #for obj_a in bpy.context.selected_objects:
            #        bpy.context.scene.objects.active = obj_a
            main_align_object(scene.AxesProperty, scene.ProjectsProperty)
        elif self.type_op==3:
            # Store Vert
            store_align('vert')
        elif self.type_op==4:
            # Store Coner
            store_align('coner') 
        elif self.type_op==5:
            # 3D Match
            match3D(False)
        else:
            # 3d Match Flip
            match3D(True)
        
        return {'FINISHED'}


class OffsetOperator(bpy.types.Operator):
    bl_idname = "mesh.offset_operator"
    bl_label = "Offset operator"
    bl_options = {'REGISTER', 'UNDO'} 
    
    type_op = bpy.props.IntProperty(name = 'type_op', default = 0, options = {'HIDDEN'})
    sign_op = bpy.props.IntProperty(name = 'sign_op', default = 1, options = {'HIDDEN'})
    
    
    @classmethod
    def poll(cls, context):
        return context.active_object is not None

    def execute(self, context):
        config = bpy.context.window_manager.paul_manager
        if self.type_op==0:     # move left / right
            if config.shift_copy:
                if bpy.context.mode=='OBJECT':
                    l_obj=[]
                    ao=bpy.context.active_object.name
                    for obj_a in bpy.context.selected_objects:
                        l_obj.append(obj_a.name)
                    for obj_a in bpy.context.selected_objects:
                        bpy.context.scene.objects.active = obj_a
                        bpy.ops.object.duplicate(linked=config.instance)
                        bpy.ops.object.select_all(action='DESELECT')
                        bpy.ops.object.select_pattern(pattern=obj_a.name)
                    for obj_a_name in l_obj:
                        bpy.context.scene.objects[obj_a_name].select=True
                    bpy.context.scene.objects.active = bpy.data.objects[ao]
                    
                elif bpy.context.mode=='EDIT_MESH':                    
                    bpy.ops.mesh.duplicate()
                
            x = config.step_len * self.sign_op
            if bpy.context.mode=='OBJECT':
                for obj_a in bpy.context.selected_objects:
                    bpy.context.scene.objects.active = obj_a
                    main_offset(x)
            else:
                main_offset(x)
        
        elif self.type_op==1:   # get length
            config.step_len = GetStoreVecLength()
        
        elif self.type_op==2:                   # copy
            copy_offset()
        
        elif self.type_op==3: 
            if config.shift_copy:
                if bpy.context.mode=='OBJECT':
                    l_obj=[]
                    ao=bpy.context.active_object.name
                    for obj_a in bpy.context.selected_objects:
                        l_obj.append(obj_a.name)
                    for obj_a in bpy.context.selected_objects:
                        bpy.context.scene.objects.active = obj_a
                        bpy.ops.object.duplicate(linked=config.instance)
                        bpy.ops.object.select_all(action='DESELECT')
                        bpy.ops.object.select_pattern(pattern=obj_a.name)
                    for obj_a_name in l_obj:
                        bpy.context.scene.objects[obj_a_name].select=True
                    bpy.context.scene.objects.active = bpy.data.objects[ao]
                    
                elif bpy.context.mode=='EDIT_MESH':                    
                    bpy.ops.mesh.duplicate()
            
            vec = GetDistToCursor()
            config.object_name_store = bpy.context.active_object.name
            config.vec_store = vec
            config.step_len = vec.length
            x = config.step_len
            if bpy.context.mode=='OBJECT':
                ao=bpy.context.active_object.name
                for obj_a in bpy.context.selected_objects:
                    bpy.context.scene.objects.active = obj_a
                    main_offset(x)
                bpy.context.scene.objects.active = bpy.data.objects[ao]
            else:
                main_offset(x)
                
            config.step_len = GetStoreVecLength()
        
        elif self.type_op==4:
            act_obj = bpy.context.active_object
            bpy.ops.object.duplicate(linked=config.instance)
            bpy.ops.object.select_all(action='DESELECT')
            bpy.ops.object.select_pattern(pattern=act_obj.name)
            bpy.context.scene.objects.active = bpy.data.objects[act_obj.name]
            
        else:
            pass
            
        self.type_op = 0
        self.sign_op = 1
        return {'FINISHED'}

class paul_managerProps(bpy.types.PropertyGroup):
    """
    Fake module like class
    bpy.context.window_manager.paul_manager
    """
    display = bpy.props.BoolProperty(name = 'display')
    display_align = bpy.props.BoolProperty(name = 'display_align')
    display_offset = bpy.props.BoolProperty(name = 'display_offset')
    display_3dmatch = bpy.props.BoolProperty(name = 'display_3dmatch')
    
    spread_x = bpy.props.BoolProperty(name = 'spread_x', default = False)
    spread_y = bpy.props.BoolProperty(name = 'spread_y', default = False)
    spread_z = bpy.props.BoolProperty(name = 'spread_z', default = True)
    relation = bpy.props.BoolProperty(name = 'relation', default = False)
    edge_idx_store = bpy.props.IntProperty(name="edge_idx_store")   
    object_name_store = bpy.props.StringProperty(name="object_name_store") 
    object_name_store_v = bpy.props.StringProperty(name="object_name_store_v") 
    object_name_store_c = bpy.props.StringProperty(name="object_name_store_c") 
    align_dist_z = bpy.props.BoolProperty(name = 'align_dist_z')
    align_lock_z = bpy.props.BoolProperty(name = 'align_lock_z')
    step_len = bpy.props.FloatProperty(name="step_len")
    vec_store = bpy.props.FloatVectorProperty(name="vec_store")
    vert_store = bpy.props.IntProperty(name="vert_store")
    coner_edge1_store = bpy.props.IntProperty(name="coner_edge1_store")
    coner_edge2_store = bpy.props.IntProperty(name="coner_edge2_store")
    active_edge1_store = bpy.props.IntProperty(name="active_edge1_store", default = -1)
    active_edge2_store = bpy.props.IntProperty(name="active_edge2_store", default = -1)
    variant = bpy.props.IntProperty(name="variant")
    instance = bpy.props.BoolProperty(name="instance")
    flip_match = bpy.props.BoolProperty(name="flip_match")
    
    shift_lockX = bpy.props.BoolProperty(name = 'shift_lockX', default = False)
    shift_lockY = bpy.props.BoolProperty(name = 'shift_lockY', default = False)
    shift_lockZ = bpy.props.BoolProperty(name = 'shift_lockZ', default = False)
    shift_copy = bpy.props.BoolProperty(name = 'shift_copy', default = False)
    shift_local = bpy.props.BoolProperty(name = 'shift_local', default = False)
    
    SPLIT = bpy.props.BoolProperty(name = 'SPLIT', default = False)
    inner_clear = bpy.props.BoolProperty(name = 'inner_clear', default = False)
    outer_clear = bpy.props.BoolProperty(name = 'outer_clear', default = False)
    fill_cuts = bpy.props.BoolProperty(name = 'fill_cuts', default = False)
    filter_edges = bpy.props.BoolProperty(name = 'filter_edges', default = False)
    filter_verts_top = bpy.props.BoolProperty(name = 'filter_verts_top', default = False)
    filter_verts_bottom = bpy.props.BoolProperty(name = 'filter_verts_bottom', default = False)
    disp_cp = bpy.props.BoolProperty(name = 'disp_cp', default = False)
    disp_cp_project = bpy.props.BoolProperty(name = 'disp_cp_project', default = False)
    disp_cp_filter = bpy.props.BoolProperty(name = 'disp_cp_filter', default = False)


class MessageOperator(bpy.types.Operator):
    from bpy.props import StringProperty
    
    bl_idname = "error.message"
    bl_label = "Message"
    type = StringProperty()
    message = StringProperty()
 
    def execute(self, context):
        self.report({'INFO'}, self.message)
        print(self.message)
        return {'FINISHED'}
 
    def invoke(self, context, event):
        wm = context.window_manager
        return wm.invoke_popup(self, width=400, height=200)
 
    def draw(self, context):
        self.layout.label(self.message, icon='BLENDER')


def print_error(message):
    bpy.ops.error.message('INVOKE_DEFAULT', 
        type = "Message",
        message = message)   




classes = [CrossPolsOperator, SSOperator, SpreadOperator, AlignOperator, LayoutSSPanel, MessageOperator, \
    OffsetOperator, paul_managerProps]


addon_keymaps = []  
def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.WindowManager.paul_manager = \
        bpy.props.PointerProperty(type = paul_managerProps) 
    bpy.context.window_manager.paul_manager.display = False
    bpy.context.window_manager.paul_manager.display_align = False
    bpy.context.window_manager.paul_manager.spread_x = False
    bpy.context.window_manager.paul_manager.spread_y = False
    bpy.context.window_manager.paul_manager.spread_z = True
    bpy.context.window_manager.paul_manager.relation = False
    bpy.context.window_manager.paul_manager.edge_idx_store = -1
    bpy.context.window_manager.paul_manager.object_name_store = ''
    bpy.context.window_manager.paul_manager.object_name_store_c = ''
    bpy.context.window_manager.paul_manager.object_name_store_v = ''
    bpy.context.window_manager.paul_manager.active_edge1_store = -1
    bpy.context.window_manager.paul_manager.active_edge2_store = -1
    bpy.context.window_manager.paul_manager.coner_edge1_store = -1
    bpy.context.window_manager.paul_manager.coner_edge2_store = -1
    bpy.context.window_manager.paul_manager.align_dist_z = False
    bpy.context.window_manager.paul_manager.align_lock_z = False
    bpy.context.window_manager.paul_manager.step_len = 1.0
    bpy.context.window_manager.paul_manager.instance = False
    bpy.context.window_manager.paul_manager.display_3dmatch = False
    bpy.context.window_manager.paul_manager.flip_match = False
    bpy.context.window_manager.paul_manager.variant = 0
    bpy.context.window_manager.paul_manager.SPLIT = False
    bpy.context.window_manager.paul_manager.inner_clear = False
    bpy.context.window_manager.paul_manager.outer_clear = False
    bpy.context.window_manager.paul_manager.fill_cuts = False
    bpy.context.window_manager.paul_manager.filter_edges = False
    bpy.context.window_manager.paul_manager.filter_verts_top = False
    bpy.context.window_manager.paul_manager.filter_verts_bottom = False
    
    wm = bpy.context.window_manager
    km = wm.keyconfigs.addon.keymaps.new(name='offset', space_type='VIEW_3D')
    kmi = km.keymap_items.new(OffsetOperator.bl_idname, 'R', 'PRESS', ctrl=False, shift=True)
    addon_keymaps.append((km, kmi))  
    
def unregister():
    for km, kmi in addon_keymaps:
        km.keymap_items.remove(kmi)
    addon_keymaps.clear() 
    
    del bpy.types.WindowManager.paul_manager
    for c in reversed(classes):  
        bpy.utils.unregister_class(c)
    

if __name__ == "__main__":
    register()