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
    "author": "Alexander Nedovizin, Paul Kotelevets aka 1D_Inc (concept design), Nikitron",
    "version": (0, 7, 14),
    "blender": (2, 7, 3),
    "location": "View3D > Toolbar",
    "category": "Mesh"
}  

# http://dl.dropboxusercontent.com/u/59609328/Blender-Rus/1D_Scripts.py

import bpy,bmesh, mathutils, math
from mathutils import Vector
from mathutils.geometry import intersect_line_plane, intersect_point_line
from math import sin, cos, pi, sqrt, degrees
import os, urllib
from bpy.props import BoolProperty
from bpy_extras.io_utils import ExportHelper
from bpy.types import Operator
import time 


list_z = []
mats_idx = []
list_f = []
maloe = 1e-5
steps_smoose = 0


def check_lukap(bm):
    if hasattr(bm.verts, "ensure_lookup_table"): 
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()


#----- Module: extrude along path -------
# author this module: Zmj100
# version 0.5.0.9
# ref: http://blenderartists.org/forum/showthread.php?179375-Addon-Edge-fillet-and-other-bmesh-tools-Update-Jan-11

def edit_mode_out():
    bpy.ops.object.mode_set(mode = 'OBJECT')

def edit_mode_in():
    bpy.ops.object.mode_set(mode = 'EDIT')

def get_adj_v_(list_):
        tmp = {}
        for i in list_:
                try:             tmp[i[0]].append(i[1])
                except KeyError: tmp[i[0]] = [i[1]]
                try:             tmp[i[1]].append(i[0])
                except KeyError: tmp[i[1]] = [i[0]]
        return tmp

def f_1(frst, list_, last):      # edge chain
    fi = frst
    tmp = [frst]
    while list_ != []:
        for i in list_:
            if i[0] == fi:
                tmp.append(i[1])
                fi = i[1]
                list_.remove(i)
            elif i[1] == fi:
                tmp.append(i[0])
                fi = i[0]
                list_.remove(i)
        if tmp[-1] == last:
            break
    return tmp

def f_2(frst, list_):      # edge loop
    fi = frst
    tmp = [frst]
    while list_ != []:
        for i in list_:
            if i[0] == fi:
                tmp.append(i[1])
                fi = i[1]
                list_.remove(i)
            elif i[1] == fi:
                tmp.append(i[0])
                fi = i[0]
                list_.remove(i)
        if tmp[-1] == frst:
            break
    return tmp

def is_loop_(list_fl):
    return True if len(list_fl) == 0 else False

def e_no_(bme, indx, p, p1):
    if hasattr(bme.verts, "ensure_lookup_table"): 
        bme.verts.ensure_lookup_table()
    tmp1 = (bme.verts[indx].co).copy()
    tmp1[0] += 0.1
    tmp1[1] += 0.1
    tmp1[2] += 0.1
    ip1 = intersect_point_line( tmp1, p, p1)[0]
    return tmp1 - ip1

# ------ ------
def f_(bme, dict_0, list_fl, loop):
    check_lukap(bme)
    if loop:
        list_1 = f_2(eap_buf.list_sp[0], eap_buf.list_ek)
        del list_1[-1]
    else:
        list_1 = f_1(eap_buf.list_sp[0], eap_buf.list_ek, list_fl[1] if eap_buf.list_sp[0] == list_fl[0] else list_fl[0])

    list_2 = [ v.index for v in bme.verts if v.select and v.is_valid ]
    n1 = len(list_2)

    list_3 = list_2[:]

    dict_1 = {}
    for k in list_2:
        dict_1[k] = [k]



    n = len(list_1)
    for i in range(n):
        p = (bme.verts[list_1[i]].co).copy()
        p1 = (bme.verts[list_1[(i - 1) % n]].co).copy()
        p2 = (bme.verts[list_1[(i + 1) % n]].co).copy()
        vec1 = p - p1
        vec2 = p - p2
        ang = vec1.angle(vec2, any)

        if round(degrees(ang)) == 180.0 or round(degrees(ang)) == 0.0:
            pp = p - ((e_no_(bme, list_1[i], p, p1)).normalized() * 0.1)
            pn = vec1.normalized()
        else:
            pp = ((p - (vec1.normalized() * 0.1)) + (p - (vec2.normalized() * 0.1))) * 0.5
            pn = ((vec1.cross(vec2)).cross(p - pp)).normalized()

        if loop:      # loop
            if i == 0:
                pass
            else:
                for j in range(n1):
                    v = (bme.verts[list_3[j]].co).copy()
                    bme.verts.new(intersect_line_plane(v, v + (vec1.normalized() * 0.1), pp, pn))
                    bme.verts.index_update()
                    if hasattr(bme.verts, "ensure_lookup_table"): 
                        bme.verts.ensure_lookup_table()
                    list_3[j] = bme.verts[-1].index
                    dict_1[list_2[j]].append(bme.verts[-1].index)

        else:      # path
            if i == 0:
                pass
            elif i == (n - 1):
                pp_ = p - ((e_no_(bme, list_fl[1] if eap_buf.list_sp[0] == list_fl[0] else list_fl[0], p, p1)).normalized() * 0.1)
                pn_ = vec1.normalized()
                for j in range(n1):
                    v = (bme.verts[list_3[j]].co).copy()
                    bme.verts.new(intersect_line_plane(v, v + (vec1.normalized() * 0.1), pp_, pn_))
                    bme.verts.index_update()
                    if hasattr(bme.verts, "ensure_lookup_table"): 
                        bme.verts.ensure_lookup_table()
                    dict_1[list_2[j]].append(bme.verts[-1].index)
            else:
                for j in range(n1):
                    v = (bme.verts[list_3[j]].co).copy()
                    bme.verts.new(intersect_line_plane(v, v + (vec1.normalized() * 0.1), pp, pn))
                    bme.verts.index_update()
                    if hasattr(bme.verts, "ensure_lookup_table"): 
                        bme.verts.ensure_lookup_table()
                    list_3[j] = bme.verts[-1].index
                    dict_1[list_2[j]].append(bme.verts[-1].index)

    # -- -- -- --
    list_4 = [[v.index for v in e.verts] for e in bme.edges if e.select and e.is_valid ]
    n2 = len(list_4)

    for t in range(n2):
        for o in range(n if loop else (n - 1)):
            bme.faces.new([bme.verts[dict_1[list_4[t][0]][o]], bme.verts[dict_1[list_4[t][1]][o]], bme.verts[dict_1[list_4[t][1]][(o + 1) % n]], bme.verts[dict_1[list_4[t][0]][(o + 1) % n]]])
            bme.faces.index_update()
            if hasattr(bme.faces, "ensure_lookup_table"): 
                bme.faces.ensure_lookup_table()

# ------ ------
class eap_buf():
    list_ek = []      # path
    list_sp = []      # start point

# ------ operator 0 ------
class eap_op0(bpy.types.Operator):
    bl_idname = 'eap.op0_id'
    bl_label = '....'

    def execute(self, context):
        edit_mode_out()
        ob_act = context.active_object
        bme = bmesh.new()
        bme.from_mesh(ob_act.data)
        check_lukap(bme)
        eap_buf.list_ek[:] = []
        for e in bme.edges:
            if e.select and e.is_valid:
                eap_buf.list_ek.append([v.index for v in e.verts])
                e.select_set(0)
        bme.to_mesh(ob_act.data)
        edit_mode_in()
        return {'FINISHED'}

# ------ operator 1 ------
class eap_op1(bpy.types.Operator):
    bl_idname = 'eap.op1_id'
    bl_label = '....'

    def execute(self, context):
        edit_mode_out()
        ob_act = context.active_object
        bme = bmesh.new()
        bme.from_mesh(ob_act.data)
        check_lukap(bme)
        eap_buf.list_sp[:] = []
        for v in bme.verts:
            if v.select and v.is_valid:
                eap_buf.list_sp.append(v.index)
                v.select_set(0)
        bme.to_mesh(ob_act.data)
        edit_mode_in()
        return {'FINISHED'}

# ------ operator 2 ------
class eap_op2(bpy.types.Operator):
    bl_idname = 'eap.op2_id'
    bl_label = 'Extrude Along Path'
    bl_options = {'REGISTER', 'UNDO'}

    def draw(self, context):
        layout = self.layout

    def execute(self, context):

        edit_mode_out()
        ob_act = context.active_object
        bme = bmesh.new()
        bme.from_mesh(ob_act.data)
        check_lukap(bme)
        
        dict_0 = get_adj_v_(eap_buf.list_ek)
        list_fl = [i for i in dict_0 if (len(dict_0[i]) == 1)]
        loop = is_loop_(list_fl)
        f_(bme, dict_0, list_fl, loop)

        bme.to_mesh(ob_act.data)
        edit_mode_in()
        return {'FINISHED'}

#-------------- END --- extrude along path ---------
        

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
    check_lukap(bm)
    
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


def getNormalPlane2(vecs, mat):
    if len(vecs)<3:
        return None
    
    out_ = []
    vec_c = mathutils.Vector((0,0,0))
    for v in vecs:
        vec  = mat*v
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


def mk_ob(mesh,name, loc):
    ob = bpy.data.objects.new(name, mesh)
    ob.location = loc
    bpy.context.scene.objects.link(ob)
    return ob  


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
    #if mode_ =='EDIT_MESH':
    if config.coner_edge1_store != -1:
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
        if config.coner_edge1_store == -1:
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
            
        else:
            v1A = obj_A.matrix_world * v1.co
            w1B = obj_B.matrix_world * w1.co 
            vecB = (w2.co - w1.co) * obj_B.matrix_world.to_3x3().transposed()
            
            normal_A = getNormalPlane([v1.co, v2.co, v3.co], mathutils.Matrix())
            normal_A = normal_A * obj_A.matrix_world.to_3x3().transposed()
            normal_B = getNormalPlane([w1.co, w2.co, w3.co], mathutils.Matrix())
            normal_B = normal_B * obj_B.matrix_world.to_3x3().transposed()
            mat_rot1 = normal_B.rotation_difference(normal_A).to_matrix().to_4x4()
            
            vecA = (v2.co - v1.co) * obj_A.matrix_world.to_3x3().transposed()
            vecB = mat_rot1 * vecB
            mat_rot = vecB.rotation_difference(vecA).to_matrix().to_4x4()
            
            obj_B.matrix_world = mat_rot * mat_rot1
            w1B = obj_B.matrix_world * w1.co
            vec_l = v1A-w1B
            obj_B.location = obj_B.location+vec_l


def mirrorside():
    mode_ = bpy.context.mode
    bpy.ops.object.mode_set(mode='OBJECT')  
    bpy.ops.object.mode_set(mode='EDIT')  
    config = bpy.context.window_manager.paul_manager
    if config.object_name_store_v == '' or \
       config.active_edge1_store < 0 or config.active_edge2_store < 0:
        print_error2('Stored Vertex is required','mirrorside 01')
        return False
    
    obj_A =  bpy.data.objects[config.object_name_store_v]
    ve1 = obj_A.data.edges[config.active_edge1_store]
    ve2 = obj_A.data.edges[config.active_edge2_store]
    
    # получаем ещё две вершины. Иначе - реджект
    connect_vs = []
    connect_vs.extend(ve1.vertices[:])
    connect_vs.extend(ve2.vertices[:])
    v1 = -1
    for v in connect_vs:
        if connect_vs.count(v)>1:
            v1 = obj_A.matrix_world * obj_A.data.vertices[v].co
            connect_vs.pop(connect_vs.index(v))
            connect_vs.pop(connect_vs.index(v))
            break
    
    if v1 == -1:
        print_error2('Active vertex of object_A must have two edges', 'mirrorside 04')
        return False
    
    v2 = obj_A.matrix_world * obj_A.data.vertices[connect_vs[0]].co
    v3 = obj_A.matrix_world * obj_A.data.vertices[connect_vs[1]].co
    
    obj_B = bpy.context.scene.objects.active
    normal_B = getNormalPlane([v1, v2, v3], mathutils.Matrix())
    if mode_ =='EDIT_MESH':
        bpy.ops.object.mode_set(mode='EDIT') 
        ref_vts = [v for v in obj_B.data.vertices if v.select==True]
        verts = []
        v_idx_B = []
        for v in ref_vts:
            verts.append(v.co)
            v_idx_B.append(v.index)
        
        bpy.ops.object.mode_set(mode='OBJECT')  
        bm = bmesh.new()
        bm.from_mesh(obj_B.data) 
        check_lukap(bm)
        
        vts = []
        mat_inv = obj_B.matrix_world.inverted()
        for pt_a_ in verts:
            pt_a = obj_B.matrix_world * pt_a_
            pt_b = normal_B + pt_a
            cross_pt = mathutils.geometry.intersect_line_plane(pt_a, pt_b, v1, normal_B)
            
            d_vec = cross_pt-pt_a
            pt_c = cross_pt + d_vec
            v_new = bm.verts.new(mat_inv*pt_c)
            vts.append(v_new)
        
        bm.verts.index_update() 
        check_lukap(bm)
        vts_ = [v.index for v in vts]
        ref_edges = [(e.vertices[0], e.vertices[1]) for e in obj_B.data.edges if e.select==True]
        for e in ref_edges:
            ev0 = v_idx_B.index(e[0])
            ev1 = v_idx_B.index(e[1])
            e = (bm.verts[vts_[ev0]], bm.verts[vts_[ev1]])
            bm.edges.new(e)
        check_lukap(bm)
        
        ref_faces = [(v for v in f.vertices) for f in obj_B.data.polygons if f.select==True]
        for f in ref_faces:
            f_B = []
            for v in f:
                fv = v_idx_B.index(v)
                f_B.append(bm.verts[vts_[fv]])
                
            bm.faces.new(tuple(f_B))
        
        bm.to_mesh(obj_B.data)       
        bm.free() 
        bpy.ops.object.mode_set(mode='EDIT')
        
    elif mode_ =='OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
        bm = bmesh.new()
        bm.from_mesh(obj_B.data) 
        check_lukap(bm)
        
        ref_vtert = bm.verts
        mat_inv = obj_B.matrix_world.inverted()
        for pt_a_ in ref_vtert:
            pt_a = obj_B.matrix_world * pt_a_.co
            pt_b = normal_B + pt_a
            cross_pt = mathutils.geometry.intersect_line_plane(pt_a, pt_b, v1, normal_B)
            
            d_vec = cross_pt-pt_a
            pt_c = cross_pt + d_vec
            ref_vtert[pt_a_.index].co = mat_inv*pt_c
        
        name = obj_B.name+'copy'
        me = bpy.data.meshes.new(name+'_Mesh')
        obj_C = bpy.data.objects.new(name, me)
        # Привязка объекта к сцене
        bpy.context.scene.objects.link(obj_C)
        
        bm.to_mesh(me)       
        me.update()   
        bm.free() 
        

def getorient():
    obj = bpy.context.active_object
    if obj.type!='MESH':
        return False
    
    space_data = bpy.context.space_data
    trans_ori = 'GLOBAL'
    if hasattr(space_data,'transform_orientation'):
        trans_ori = space_data.transform_orientation
    
    bpy.ops.transform.create_orientation(name='1DTEMP', use=True, overwrite=True)
    if trans_ori=='1DTEMP':
        bpy.context.space_data.show_manipulator = not bpy.context.space_data.show_manipulator
    else:
        bpy.context.space_data.show_manipulator = True


def main_align_object(axe='X',project='XY'):
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
    check_lukap(bm)
    
    # Найдём диагональ Store
    edge_idx = config.edge_idx_store
    verts_edge_store = bm.edges[edge_idx].verts
    vec_diag_store = verts_edge_store[1].co - verts_edge_store[0].co
    
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
    check_lukap(bm)
    
    # Найдём диагональ Store
    edge_idx = config.edge_idx_store
    verts_edge_store = bm.edges[edge_idx].verts
    vec_diag_store = verts_edge_store[1].co - verts_edge_store[0].co
    
    # Получим выделенное ребро
    obj_res = bpy.context.active_object
    mesh_act = obj_res.data
    bm_act = bmesh.new()
    bm_act.from_mesh(mesh_act)  
    check_lukap(bm_act)
    
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
        

def main_spread(context, mode, influe):
    conf = bpy.context.window_manager.paul_manager
    
    if conf.shape_spline and influe<51:
        return main_spline(context, mode, influe/50)
    elif conf.shape_spline and influe<101:
        if not conf.spline_Bspline2 or main_spline(context, mode, (100-influe)/50):
            return main_B_spline_2(context, mode, (influe-50)/50)
        else:
            return False
    elif conf.shape_spline and influe<151:
        if not conf.spline_Bspline2 or main_B_spline_2(context, mode, (150-influe)/50):
            return main_B_spline(context, mode, (influe-100)/50)
        else:
            return False
    elif conf.shape_spline and influe<201:
        if not conf.spline_Bspline2 or main_B_spline(context, mode, (200-influe)/50):
            return main_Basier_mid(context, mode, (influe-150)/50)
        else:
            return False
    elif conf.shape_spline and influe>200:
        if conf.spline_Bspline2:
            return main_Basier_mid(context, mode, (250-influe)/50)
        else:
            return False
    
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
    space_data = bpy.context.space_data
    trans_ori = 'GLOBAL'
    if hasattr(space_data,'transform_orientation'):
        trans_ori = space_data.transform_orientation
    
    bpy.ops.object.mode_set(mode='OBJECT') 
    bpy.ops.object.mode_set(mode='EDIT') 
    
    vs_idx = find_index_of_selected_vertex(me)
    if vs_idx:
        x_coos = [v.co.x for v in me.vertices if v.index in vs_idx]
        y_coos = [v.co.y for v in me.vertices if v.index in vs_idx]
        z_coos = [v.co.z for v in me.vertices if v.index in vs_idx]
        
        min_x = min(x_coos)
        max_x = max(x_coos)
        
        min_y = min(y_coos)
        max_y = max(y_coos)
        
        min_z = min(z_coos)
        max_z = max(z_coos)
        
        len_x = max_x-min_x
        len_y = max_y-min_y
        
        if trans_ori=='1DTEMP':
              bpy.ops.transform.resize(value=(1,1,0), constraint_axis=(False,False,True), constraint_orientation=trans_ori)
        else:
            if len_y<len_x:
                bpy.ops.transform.resize(value=(1,0,1), constraint_axis=(False,True,False), constraint_orientation=trans_ori)
            else:
                bpy.ops.transform.resize(value=(0,1,1), constraint_axis=(True,False,False), constraint_orientation=trans_ori)


def main_offset(x):
    mode_obj=bpy.context.mode=='OBJECT'
    mode_obj2 = bpy.context.mode
    bpy.ops.object.mode_set(mode='OBJECT')
    if mode_obj2=='EDIT_MESH':
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
        
        if mode_obj2=='EDIT_MESH':
            bm_act = bmesh.new()
            bm_act.from_mesh(me) 
            check_lukap(bm_act)
            
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
                

def main_rotor(angle_):
    mode_obj=bpy.context.mode=='OBJECT'
    mode_obj2 = bpy.context.mode
    bpy.ops.object.mode_set(mode='OBJECT')
    if mode_obj2=='EDIT_MESH':
        bpy.ops.object.mode_set(mode='EDIT') 
    
    config = bpy.context.window_manager.paul_manager
    if config.object_name_store == '':
        print_error2('Stored key is required', '01 rotor 3D')
        return False
    
    obj = bpy.context.active_object
    obj_cone = bpy.data.objects[config.object_name_store]
    if obj:
        center = mathutils.Vector(config.rotor3d_center)
        axis_ = mathutils.Vector(config.rotor3d_axis)
        
        me = obj.data
        
        if mode_obj2=='EDIT_MESH':
            bm_act = bmesh.new()
            bm_act.from_mesh(me) 
            check_lukap(bm_act)
            
            verts_act = find_index_of_selected_vertex(me)
            vts_all = [v for v in bm_act.verts if v.index in verts_act]
            edg_all = [e for e in bm_act.edges if e.verts[0].index in verts_act and \
                                                  e.verts[1].index in verts_act]
            fcs_all = [f for f in bm_act.faces if f.select==True]
            geom_ = vts_all+edg_all+fcs_all
                
            if config.rotor3d_copy:
                ret = bmesh.ops.spin(bm_act, geom=geom_, cent=center, \
                   space=mathutils.Matrix(), axis=axis_, angle=angle_, steps=1, use_duplicate=True)
                for v in bm_act.verts:
                    v.select_set(False)
                bm_act.select_flush(False)
                for v in ret['geom_last']:
                    v.select=True
            else:
                mat_rot = mathutils.Matrix.Rotation(angle_, 4, axis_)
                bmesh.ops.rotate(bm_act, cent=center, verts=vts_all, \
                    matrix=mat_rot, space=mathutils.Matrix())
                
            bpy.ops.object.mode_set(mode='OBJECT')
            bm_act.to_mesh(me)
            bm_act.free()
            bpy.ops.object.mode_set(mode='EDIT') 
        else:
            bpy.ops.object.mode_set(mode='OBJECT')
            axis_ = axis_ * obj_cone.matrix_world 
            center = obj_cone.matrix_world * center
            loc = center-obj.location
            mat_loc = mathutils.Matrix.Translation(loc)
            mat_loc2 = mathutils.Matrix.Translation(-loc)
            mat_rot = mathutils.Matrix.Rotation(angle_, 4, axis_)
            mat_rot2 = obj.rotation_euler.copy()
            loc_, rot_, sca_ = obj.matrix_world.decompose()
            rot_inv = rot_.to_matrix().to_4x4().inverted()
            
            obj.matrix_world *= rot_inv
            #obj.rotation_euler = mathutils.Euler((0,0,0),'XYZ')
            #bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
            mat_out = mat_loc * mat_rot * mat_loc2
            obj.matrix_world = obj.matrix_world * mat_out * rot_.to_matrix().to_4x4()
            
                
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
            check_lukap(bm)
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


def GetStoreVecAngle():
    bpy.ops.object.editmode_toggle()
    bpy.ops.object.editmode_toggle()
    
    obj = bpy.context.active_object
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)  
    result = True
    check_lukap(bm)
    
    obj_name = obj.name
    config = bpy.context.window_manager.paul_manager
    active_edge, el = bm_vert_active_get(bm)
    old_edge1 = config.active_edge1_store
    old_edge2 = config.active_edge2_store
    old_name = config.object_name_store
    old_step_angle = config.step_angle
    
    if active_edge != None and el=='E':
        config.object_name_store = obj_name
        config.active_edge1_store = active_edge
        Edges = mesh.edges
        verts = bm.edges[active_edge].verts
        v0 = verts[0].index
        v1 = verts[1].index
        edges_idx = [i.index for i in Edges \
            if i.select and i.index!=active_edge and \
            (v1 in i.vertices[:] or v0 in i.vertices[:]) ] 
        if edges_idx:
            config.active_edge2_store = edges_idx[0]
            l_ed2 = [Edges[edges_idx[0]].vertices[0], Edges[edges_idx[0]].vertices[1]]
            (v1, v0) = (v0, v1) if v0 in l_ed2 else (v1, v0)
            l_ed2.pop(l_ed2.index(v1))
            v2 = l_ed2[0]
            v0_ = bm.verts[v0].co 
            v1_ = bm.verts[v1].co
            v2_ = bm.verts[v2].co
            config.step_angle = (v0_-v1_).angle(v2_-v1_,0)
            config.rotor3d_center = v1_
            config.rotor3d_axis = mathutils.geometry.normal(v0_,v1_,v2_)
            print('edge mode',obj_name)
            return True
        
    if active_edge != None and el=='V':
        active_vert = active_edge
        config.object_name_store = obj_name
        
        v2_l = find_all_connected_verts(mesh, active_vert,[],0)
        control_vs = find_connected_verts_simple(mesh, active_vert)
        if len(v2_l)>2 and len(control_vs)==1:
            v1 = v2_l.pop(1)
            edges_idx = []
            for v2 in v2_l[:2]:
                edges_idx.extend([i.index for i in mesh.edges \
                    if v1 in i.vertices[:] and v2 in i.vertices[:]] )
                
            if len(edges_idx)>1:
                config.active_edge1_store = edges_idx[0]
                config.active_edge2_store = edges_idx[1]
                v0_ = bm.verts[active_vert].co
                v1_ = bm.verts[v1].co
                v2_ = bm.verts[v2].co
                config.step_angle = (v0_-v1_).angle(v2_-v1_,0)
                config.rotor3d_center = v1_
                config.rotor3d_axis = mathutils.geometry.normal(v0_,v1_,v2_)
                return True
    
    
    config.object_name_store = ''
    config.active_edge1_store = -1
    config.active_edge2_store = -1
    config.step_angle = 0   
    print_error2('Side is undefined', '01 GetStoreVecAngle')
    
    bm.free()   
    

def select_v_on_plane():
    config = bpy.context.window_manager.paul_manager
    obj = bpy.context.active_object
    if obj.type != 'MESH':
        return
    
    bpy.ops.object.mode_set(mode='OBJECT')  
    bpy.ops.object.mode_set(mode='EDIT')  
    bpy.ops.mesh.select_mode(type='VERT') 
    
    me = obj.data
    bm = bmesh.new()
    bm.from_mesh(me)
    check_lukap(bm)
    
    P1 = me.polygons[bm.faces.active.index]
    pols = [p.index for p in me.polygons if p.select and p.index!= P1.index]
    vts_all = [v for v in bm.verts if v.select and v.index not in P1.vertices]
    p1_co = me.vertices[P1.vertices[0]].co
    p1_no = P1.normal
    dist_max = bpy.context.tool_settings.double_threshold
    
    for v in bm.verts:
        v.select_set(False)
        bm.select_flush(False)
    
    for p2 in vts_all:
        dist = abs(mathutils.geometry.distance_point_to_plane(p2.co, p1_co, p1_no))
        if dist<=dist_max:
            p2.select = True
            
    bpy.ops.object.mode_set(mode='OBJECT') 
    bm.to_mesh(me)       
    me.update()   
    bm.free()
    bpy.ops.object.mode_set(mode='EDIT') 
    

def crosspols():
    config = bpy.context.window_manager.paul_manager
    obj = bpy.context.active_object
    if obj.type != 'MESH':
        return
    
    bpy.ops.object.mode_set(mode='OBJECT')  
    bpy.ops.object.mode_set(mode='EDIT')  
    
    me = obj.data
    
    bm = bmesh.new()
    bm.from_mesh(me)
    check_lukap(bm)
    elem, mode = bm_vert_active_get(bm)
    if mode == None: 
        mode = 'F'
        elem = bm.faces.active.index
        
    if mode == 'V':
        sec_vts = find_all_connected_verts(me, elem, not_list=[])
        
    if mode == 'E':
        v0 = mathutils.Vector(bm.edges[elem].verts[0].co)
        v1 = mathutils.Vector(bm.edges[elem].verts[1].co)
        v2 = v0+mathutils.Vector((0,0,1))
        v3 = v1+mathutils.Vector((0,0,1))
        p1_no_ = mathutils.geometry.normal(v0,v2,v3,v1)
        
    bpy.ops.mesh.select_mode(type='FACE') 
    
    if mode == 'E':
        pols = [p.index for p in me.polygons if p.select]
        vts_all = [v for v in bm.verts if v.select]
        eds_all = [e for e in bm.edges if e.select]
        P1 = me.polygons[bm.faces.active.index]
        list_p1co = [me.vertices[P1.vertices[0]].co]
    elif mode == 'F':
        P1 = me.polygons[bm.faces.active.index]
        pols = [p.index for p in me.polygons if p.select and p.index!= P1.index]
        vts_all = [v for v in bm.verts if v.select and v.index not in P1.vertices]
        eds_all = [e for e in bm.edges if e.select and e.verts[0].index not in P1.vertices \
                                                   and e.verts[1].index not in P1.vertices]
        list_p1co = [me.vertices[P1.vertices[0]].co]
    elif mode == 'V':
        if config.SPLIT:
            sec_vts = [elem]
            
        pols = [p.index for p in me.polygons if p.select]
        vts_all = [v for v in bm.verts if v.select and v.index not in sec_vts]
        eds_all = [e for e in bm.edges if e.select and e.verts[0].index not in sec_vts \
                                                   and e.verts[1].index not in sec_vts]
        list_p1co = [me.vertices[i].co for i in sec_vts]
        
    sel_edges = []
    sel_verts = []
    for l_p1co in list_p1co:
        if not config.filter_verts_top and not config.filter_verts_bottom and not config.filter_edges:
            p1_co = l_p1co
            if mode == 'E':
                p1_co = v0
                p1_no = p1_no_
            elif mode == 'V':
                p1_no = Vector((0,0,1))
            else:
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
                        if mode!= 'V' or len(list_p1co)==1:
                            res = bmesh.ops.bisect_plane(bm, geom=geom_in, dist=0.00001,
                                                         plane_co=p1_co, plane_no=p1_no, use_snap_center=False,
                                                         clear_outer=config.outer_clear, clear_inner=config.inner_clear)
                        else:
                            res = bmesh.ops.bisect_plane(bm, geom=geom_in, dist=0.00001,
                                                     plane_co=p1_co, plane_no=p1_no, use_snap_center=False,
                                                     clear_outer=False, clear_inner=False)
                        
                        fres = bmesh.ops.edgenet_prepare(bm, edges=[e for e in res['geom_cut']
                                                                    if isinstance(e, bmesh.types.BMEdge)])
                        
                        sel_edges = [e for e in fres['edges'] if e.index not in edges4cut_idx]
                        
                        # this needs work function with solid gemometry
                        if config.fill_cuts and len(list_p1co)<2:
                            fres = bmesh.ops.edgenet_prepare(bm, edges=[e for e in res['geom_cut']
                                                                        if isinstance(e, bmesh.types.BMEdge)])
                            bmesh.ops.edgeloop_fill(bm, edges=fres['edges'])

                        bm.verts.index_update()
                        bm.edges.index_update()
                        bm.faces.index_update()
                        check_lukap(bm)
                        break
               
        if config.filter_verts_top or config.filter_verts_bottom:
            bpy.ops.mesh.select_mode(type='VERT') 
            p1_co = l_p1co
            if mode == 'E':
                p1_co = v0
                p1_no = p1_no_
            elif mode == 'V':
                p1_no = Vector((0,0,1))
            else:
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
            p1_co = l_p1co
            if mode == 'E':
                p1_co = v0
                p1_no = p1_no_
            elif mode == 'V':
                p1_no = Vector((0,0,1))
            else:
                p1_no = P1.normal
                
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


def main_spline(context, mode, influe):
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
            print('Error: 01 simple_spline')
            return False
        
        
        sort_list = find_all_connected_verts(me,extreme_vs[0],[])
        all_vts_sort_x = [me.vertices[i].co.x for i in sort_list]
        all_vts_sort_y = [me.vertices[i].co.y for i in sort_list]
        all_vts_sort_z = [me.vertices[i].co.z for i in sort_list]
        
        max_p = [max(all_vts_sort_x), max(all_vts_sort_y), max(all_vts_sort_z)]
        min_p = [min(all_vts_sort_x), min(all_vts_sort_y), min(all_vts_sort_z)]
        diap_p = list(map(lambda a,b: a-b, max_p, min_p))
        
        if len(sort_list) != len(verts):
            print_error('Incoherent loop')
            print('Error: 020 simple_spline')
            return False
        
        list_length = []
        sum_length = 0.0
        for sl in range(cou_vs):
            subb = me.vertices[sort_list[sl+1]].co-me.vertices[sort_list[sl]].co
            sum_length += subb.length
            list_length.append(sum_length)
        
        list_koeff = []
        for sl in range(cou_vs):
            tmp = list_length[sl]/sum_length
            list_koeff.append(tmp)
        
        bpy.ops.object.mode_set(mode='OBJECT') 
        bm = bmesh.new()
        bm.from_mesh(me)
        check_lukap(bm)
        
        pa_idx = bm_vert_active_get(bm)[0]
        if pa_idx==None:
            print_error('Active vert is not detected')
            print('Error: 030 simple_spline')
            return False
        
        pa_sort = sort_list.index(pa_idx)
        if pa_sort == 0: pa_sort = 1
        pa_perc = list_koeff[pa_sort-1]
        p0_ = me.vertices[sort_list[0]].co
        p1_ = me.vertices[pa_idx].co
        p2_ = me.vertices[sort_list[-1]].co
        
        if mode[3]:
            l = len(list_koeff)
            d = 1/l
            list_koeff = list(map(lambda n: d*n, list(range(1,l+1))))
        
        if mode[0]:
            all_vts_sort = [me.vertices[i].co.x for i in sort_list]
            p0 = p0_.x
            p1 = p1_.x - p0
            p2 = p2_.x - p0
            
            t = pa_perc
            if p1==0 or p1==p2:
                new_vts = list(map(lambda t: p2*t**2, list_koeff))
            else:
                b = (p1-pa_perc**2*p2)/(2*pa_perc*(1-pa_perc)+1e-8)
                new_vts = list(map(lambda t: 2*b*t*(1-t)+p2*t**2, list_koeff))
            
            for idx in range(cou_vs):
                me.vertices[sort_list[idx+1]].co.x += (new_vts[idx]+p0-me.vertices[sort_list[idx+1]].co.x)*influe

        if mode[1]:
            all_vts_sort = [me.vertices[i].co.y for i in sort_list]
            p0 = p0_.y
            p1 = p1_.y - p0
            p2 = p2_.y - p0
            
            b = (p1-pa_perc**2*p2)/(2*pa_perc*(1-pa_perc)+1e-8)
            new_vts = list(map(lambda t: 2*b*t*(1-t)+p2*t**2, list_koeff))
            
            for idx in range(cou_vs):
                me.vertices[sort_list[idx+1]].co.y += (new_vts[idx]+p0-me.vertices[sort_list[idx+1]].co.y)*influe
            
        if mode[2]:
            all_vts_sort = [me.vertices[i].co.z for i in sort_list]
            p0 = p0_.z
            p1 = p1_.z - p0
            p2 = p2_.z - p0
            
            b = (p1-pa_perc**2*p2)/(2*pa_perc*(1-pa_perc)+1e-8)
            new_vts = list(map(lambda t: 2*b*t*(1-t)+p2*t**2, list_koeff))
            
            for idx in range(cou_vs):
                me.vertices[sort_list[idx+1]].co.z += (new_vts[idx]+p0-me.vertices[sort_list[idx+1]].co.z)*influe

        me.update()   
        bm.free() 
        
        bpy.ops.object.mode_set(mode='EDIT')  
            
    return True


def main_B_spline(context, mode, influe):
    global steps_smoose
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
            print('Error: 01 B_spline')
            return False
        
        
        sort_list = find_all_connected_verts(me,extreme_vs[0],[])
        all_vts_sort_x = [me.vertices[i].co.x for i in sort_list]
        all_vts_sort_y = [me.vertices[i].co.y for i in sort_list]
        all_vts_sort_z = [me.vertices[i].co.z for i in sort_list]
        
        max_p = [max(all_vts_sort_x), max(all_vts_sort_y), max(all_vts_sort_z)]
        min_p = [min(all_vts_sort_x), min(all_vts_sort_y), min(all_vts_sort_z)]
        diap_p = list(map(lambda a,b: a-b, max_p, min_p))
        
        if len(sort_list) != len(verts):
            print_error('Incoherent loop')
            print('Error: 020 B_spline')
            return False
        
        list_length = []
        sum_length = 0.0
        for sl in range(cou_vs-2):
            subb = me.vertices[sort_list[sl+2]].co-me.vertices[sort_list[sl+1]].co
            sum_length += subb.length
            list_length.append(sum_length)
        
        list_koeff = []
        for sl in range(cou_vs-2):
            tmp = list_length[sl]/sum_length
            list_koeff.append(tmp)
        
        bpy.ops.object.mode_set(mode='OBJECT') 
        bm = bmesh.new()
        bm.from_mesh(me)
        check_lukap(bm)
        
        pa_idx = bm_vert_active_get(bm)[0]
        if pa_idx==None:
            print_error('Active vert is not detected')
            print('Error: 030 B_spline')
            return False
        
        pa_sort = sort_list.index(pa_idx)
        if pa_sort < 2: pa_sort = 2
        if pa_sort > len(sort_list)-3: pa_sort = len(sort_list)-3
        pa_idx = sort_list[pa_sort]
        pa_perc = list_koeff[pa_sort-2]
        p0_ = me.vertices[sort_list[1]].co
        p1_ = me.vertices[pa_idx].co
        p2_ = me.vertices[sort_list[-2]].co
        
        kn1_ = me.vertices[sort_list[0]].co
        kn2_ = me.vertices[sort_list[-1]].co
        nkn1_ = p1_ - kn1_ + p1_
        nkn2_ = p2_ - kn2_ + p2_
        
        if mode[3]:
            l = len(list_koeff)
            d = 1/l
            list_koeff = list(map(lambda n: d*n, list(range(1,l+1))))
        
        if mode[0]:
            all_vts_sort = [me.vertices[i].co.x for i in sort_list]
            p0 = p0_.x
            p1 = p1_.x - p0
            p2 = p2_.x - p0
            knot_1 = nkn1_.x - p0
            knot_2 = nkn2_.x - p0
            
            t = pa_perc
            b = (p1-(4*knot_1*t*(1-t)**3)-(4*t**3*(1-t)*knot_2+p2*t**4))/(4*t**2*(1-t)**2+1e-8)
            new_vts = list(map(lambda t: 4*knot_1*t*(1-t)**3+4*b*t**2*(1-t)**2+4*t**3*(1-t)*knot_2+p2*t**4, list_koeff))
            
            if mode[3]:
                for c in range(steps_smoose):
                    new_vts_ = [0]+new_vts
                    V = [vi for vi in new_vts_]
                    P = list(map(lambda x,y: abs(y-x), V[:-1], V[1:]))
                    L = sum(P)
                    lp = len(P)
                    d = L/lp    
                    l_ = list(map(lambda y: d*y/L, list(range(1,lp+1))))
                    l = list(map(lambda x: x/L, P))

                    tmp = 0
                    for i in range(lp):
                        tmp += l[i]
                        m = l_[i]/tmp
                        list_koeff[i] = m*list_koeff[i]
                    new_vts = list(map(lambda t: 4*knot_1*t*(1-t)**3+4*b*t**2*(1-t)**2+4*t**3*(1-t)*knot_2+p2*t**4, list_koeff))
                
            
            for idx in range(cou_vs-2):
                me.vertices[sort_list[idx+2]].co.x += (new_vts[idx]+p0-me.vertices[sort_list[idx+2]].co.x)*influe

        if mode[1]:
            all_vts_sort = [me.vertices[i].co.y for i in sort_list]
            p0 = p0_.y
            p1 = p1_.y - p0
            p2 = p2_.y - p0
            knot_1 = nkn1_.y - p0
            knot_2 = nkn2_.y - p0
            
            t = pa_perc
            b = (p1-(4*knot_1*t*(1-t)**3)-(4*t**3*(1-t)*knot_2+p2*t**4))/(4*t**2*(1-t)**2+1e-8)
            new_vts = list(map(lambda t: 4*knot_1*t*(1-t)**3+4*b*t**2*(1-t)**2+4*t**3*(1-t)*knot_2+p2*t**4, list_koeff))
            
            if mode[3]:
                for c in range(steps_smoose):
                    new_vts_ = [0]+new_vts
                    V = [vi for vi in new_vts_]
                    P = list(map(lambda x,y: abs(y-x), V[:-1], V[1:]))
                    L = sum(P)
                    lp = len(P)
                    d = L/lp    
                    l_ = list(map(lambda y: d*y/L, list(range(1,lp+1))))
                    l = list(map(lambda x: x/L, P))

                    tmp = 0
                    for i in range(lp):
                        tmp += l[i]
                        m = l_[i]/tmp
                        list_koeff[i] = m*list_koeff[i]
                    new_vts = list(map(lambda t: 4*knot_1*t*(1-t)**3+4*b*t**2*(1-t)**2+4*t**3*(1-t)*knot_2+p2*t**4, list_koeff))
                
            
            for idx in range(cou_vs-2):
                me.vertices[sort_list[idx+2]].co.y += (new_vts[idx]+p0-me.vertices[sort_list[idx+2]].co.y)*influe
            
        if mode[2]:
            all_vts_sort = [me.vertices[i].co.z for i in sort_list]
            p0 = p0_.z
            p1 = p1_.z - p0
            p2 = p2_.z - p0
            knot_1 = nkn1_.z - p0
            knot_2 = nkn2_.z - p0
            
            t = pa_perc
            b = (p1-(4*knot_1*t*(1-t)**3)-(4*t**3*(1-t)*knot_2+p2*t**4))/(4*t**2*(1-t)**2+1e-8)
            new_vts = list(map(lambda t: 4*knot_1*t*(1-t)**3+4*b*t**2*(1-t)**2+4*t**3*(1-t)*knot_2+p2*t**4, list_koeff))
            
            if mode[3]:
                for c in range(steps_smoose):
                    new_vts_ = [0]+new_vts
                    V = [vi for vi in new_vts_]
                    P = list(map(lambda x,y: abs(y-x), V[:-1], V[1:]))
                    L = sum(P)
                    lp = len(P)
                    d = L/lp    
                    l_ = list(map(lambda y: d*y/L, list(range(1,lp+1))))
                    l = list(map(lambda x: x/L, P))

                    tmp = 0
                    for i in range(lp):
                        tmp += l[i]
                        m = l_[i]/tmp
                        list_koeff[i] = m*list_koeff[i]
                    new_vts = list(map(lambda t: 4*knot_1*t*(1-t)**3+4*b*t**2*(1-t)**2+4*t**3*(1-t)*knot_2+p2*t**4, list_koeff))
                
            
            for idx in range(cou_vs-2):
                me.vertices[sort_list[idx+2]].co.z += (new_vts[idx]+p0-me.vertices[sort_list[idx+2]].co.z)*influe
            
        
            
            
            
        me.update()   
        bm.free() 
        
        bpy.ops.object.mode_set(mode='EDIT')  
            
    return True


def main_B_spline_2(context, mode, influe):
    global steps_smoose
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
            print('Error: 01 B_spline')
            return False
        
        
        sort_list = find_all_connected_verts(me,extreme_vs[0],[])
        all_vts_sort_x = [me.vertices[i].co.x for i in sort_list]
        all_vts_sort_y = [me.vertices[i].co.y for i in sort_list]
        all_vts_sort_z = [me.vertices[i].co.z for i in sort_list]
        
        max_p = [max(all_vts_sort_x), max(all_vts_sort_y), max(all_vts_sort_z)]
        min_p = [min(all_vts_sort_x), min(all_vts_sort_y), min(all_vts_sort_z)]
        diap_p = list(map(lambda a,b: a-b, max_p, min_p))
        
        if len(sort_list) != len(verts):
            print_error('Incoherent loop')
            print('Error: 020 B_spline')
            return False
        
        list_length = []
        sum_length = 0.0
        for sl in range(cou_vs):
            subb = me.vertices[sort_list[sl+1]].co-me.vertices[sort_list[sl]].co
            sum_length += subb.length
            list_length.append(sum_length)
        
        list_koeff = []
        for sl in range(cou_vs):
            tmp = list_length[sl]/sum_length
            list_koeff.append(tmp)
        
        bpy.ops.object.mode_set(mode='OBJECT') 
        bm = bmesh.new()
        bm.from_mesh(me)
        check_lukap(bm)
        
        pa_idx = bm_vert_active_get(bm)[0]
        if pa_idx==None:
            print_error('Active vert is not detected')
            print('Error: 030 B_spline')
            return False
        
        list_koeff = [0]+list_koeff
        pa_sort = sort_list.index(pa_idx)
        if pa_sort == 0: 
            pa_perc = 0
            kn1_i = sort_list[0]
            kn2_i = sort_list[pa_sort+1]
        elif pa_sort == len(sort_list)-1:
            pa_perc = 1.0
            kn1_i = sort_list[pa_sort-1]
            kn2_i = sort_list[-1]
        else:
            kn1_i = sort_list[pa_sort-1]
            kn2_i = sort_list[pa_sort+1]
            pa_perc = list_koeff[pa_sort]
        
        kn1_ = me.vertices[kn1_i].co
        kn2_ = me.vertices[kn2_i].co
        
        p0_ = me.vertices[sort_list[0]].co
        p1_ = me.vertices[pa_idx].co
        p2_ = me.vertices[sort_list[-1]].co
        
        if mode[3]:
            l = len(list_koeff)-1
            d = 1/l
            list_koeff = list(map(lambda n: d*n, list(range(0,l+1))))
        
        if mode[0]:
            p0 = p0_.x
            p1 = p1_.x - p0
            p2 = p2_.x - p0
            knot_1 = kn1_.x - p0
            knot_2 = kn2_.x - p0
            
            t = pa_perc
            if knot_1==0 and p1!=0:
                b = (p1-(3*knot_2*t**2*(1-t)+p2*t**3))/(3*t*(1-t)**2+1e-8)
                new_vts = list(map(lambda t: 3*b*t*(1-t)**2+3*knot_2*t**2*(1-t)+p2*t**3, list_koeff))
            elif p1==0:
                new_vts = list(map(lambda t: 2*knot_2*t*(1-t)+p2*t**2, list_koeff))
            elif knot_2==p2 and p1!=p2:
                b = (p1-(3*knot_1*t*(1-t)**2+p2*t**3))/(3*t**2*(1-t)+1e-8)
                new_vts = list(map(lambda t: 3*knot_1*t*(1-t)**2+3*b*t**2*(1-t)+p2*t**3, list_koeff))
            elif p1==p2:
                new_vts = list(map(lambda t: 2*knot_1*t*(1-t)+p2*t**2, list_koeff))
            else:
                b = (p1-(4*knot_1*t*(1-t)**3+4*t**3*(1-t)*knot_2+p2*t**4))/(4*t**2*(1-t)**2+1e-8)
                new_vts = list(map(lambda t: 4*knot_1*t*(1-t)**3+4*b*t**2*(1-t)**2+4*t**3*(1-t)*knot_2+p2*t**4, list_koeff))
            
            if mode[3]:
                for c in range(steps_smoose):
                    new_vts_ = new_vts
                    V = [vi for vi in new_vts_]
                    P = list(map(lambda x,y: abs(y-x), V[:-1], V[1:]))
                    L = sum(P)
                    lp = len(P)
                    d = L/lp    
                    l_ = list(map(lambda y: d*y/L, list(range(1,lp+1))))
                    l = list(map(lambda x: x/L, P))

                    tmp = 1e-8
                    for i in range(lp):
                        tmp += l[i]
                        m = l_[i]/tmp
                        list_koeff[i] = m*list_koeff[i]
                        
                    if knot_1==0 and p1!=0:
                        b = (p1-(3*knot_2*t**2*(1-t)+p2*t**3))/(3*t*(1-t)**2+1e-8)
                        new_vts = list(map(lambda t: 3*b*t*(1-t)**2+3*knot_2*t**2*(1-t)+p2*t**3, list_koeff))
                    elif p1==0:
                        new_vts = list(map(lambda t: 2*knot_2*t*(1-t)+p2*t**2, list_koeff))
                    elif knot_2==p2 and p1!=p2:
                        b = (p1-(3*knot_1*t*(1-t)**2+p2*t**3))/(3*t**2*(1-t)+1e-8)
                        new_vts = list(map(lambda t: 3*knot_1*t*(1-t)**2+3*b*t**2*(1-t)+p2*t**3, list_koeff))
                    elif p1==p2:
                        new_vts = list(map(lambda t: 2*knot_1*t*(1-t)+p2*t**2, list_koeff))
                    else:
                        b = (p1-(4*knot_1*t*(1-t)**3+4*t**3*(1-t)*knot_2+p2*t**4))/(4*t**2*(1-t)**2+1e-8)
                        new_vts = list(map(lambda t: 4*knot_1*t*(1-t)**3+4*b*t**2*(1-t)**2+4*t**3*(1-t)*knot_2+p2*t**4, list_koeff))
            
            for idx in range(cou_vs+1):
                me.vertices[sort_list[idx]].co.x += (new_vts[idx] + p0 - me.vertices[sort_list[idx]].co.x)*influe

        if mode[1]:
            p0 = p0_.y
            p1 = p1_.y - p0
            p2 = p2_.y - p0
            knot_1 = kn1_.y - p0
            knot_2 = kn2_.y - p0
            
            t = pa_perc
            if knot_1==0 and p1!=0:
                b = (p1-(3*knot_2*t**2*(1-t)+p2*t**3))/(3*t*(1-t)**2+1e-8)
                new_vts = list(map(lambda t: 3*b*t*(1-t)**2+3*knot_2*t**2*(1-t)+p2*t**3, list_koeff))
            elif p1==0:
                new_vts = list(map(lambda t: 2*knot_2*t*(1-t)+p2*t**2, list_koeff))
            elif knot_2==p2 and p1!=p2:
                b = (p1-(3*knot_1*t*(1-t)**2+p2*t**3))/(3*t**2*(1-t)+1e-8)
                new_vts = list(map(lambda t: 3*knot_1*t*(1-t)**2+3*b*t**2*(1-t)+p2*t**3, list_koeff))
            elif p1==p2:
                new_vts = list(map(lambda t: 2*knot_1*t*(1-t)+p2*t**2, list_koeff))
            else:
                b = (p1-(4*knot_1*t*(1-t)**3+4*t**3*(1-t)*knot_2+p2*t**4))/(4*t**2*(1-t)**2+1e-8)
                new_vts = list(map(lambda t: 4*knot_1*t*(1-t)**3+4*b*t**2*(1-t)**2+4*t**3*(1-t)*knot_2+p2*t**4, list_koeff))
            
            if mode[3]:
                for c in range(steps_smoose):
                    new_vts_ = new_vts
                    V = [vi for vi in new_vts_]
                    P = list(map(lambda x,y: abs(y-x), V[:-1], V[1:]))
                    L = sum(P)
                    lp = len(P)
                    d = L/lp    
                    l_ = list(map(lambda y: d*y/L, list(range(1,lp+1))))
                    l = list(map(lambda x: x/L, P))

                    tmp = 1e-8
                    for i in range(lp):
                        tmp += l[i]
                        m = l_[i]/tmp
                        list_koeff[i] = m*list_koeff[i]
                    
                    if knot_1==0 and p1!=0:
                        b = (p1-(3*knot_2*t**2*(1-t)+p2*t**3))/(3*t*(1-t)**2+1e-8)
                        new_vts = list(map(lambda t: 3*b*t*(1-t)**2+3*knot_2*t**2*(1-t)+p2*t**3, list_koeff))
                    elif p1==0:
                        new_vts = list(map(lambda t: 2*knot_2*t*(1-t)+p2*t**2, list_koeff))
                    elif knot_2==p2 and p1!=p2:
                        b = (p1-(3*knot_1*t*(1-t)**2+p2*t**3))/(3*t**2*(1-t)+1e-8)
                        new_vts = list(map(lambda t: 3*knot_1*t*(1-t)**2+3*b*t**2*(1-t)+p2*t**3, list_koeff))
                    elif p1==p2:
                        new_vts = list(map(lambda t: 2*knot_1*t*(1-t)+p2*t**2, list_koeff))
                    else:
                        b = (p1-(4*knot_1*t*(1-t)**3+4*t**3*(1-t)*knot_2+p2*t**4))/(4*t**2*(1-t)**2+1e-8)
                        new_vts = list(map(lambda t: 4*knot_1*t*(1-t)**3+4*b*t**2*(1-t)**2+4*t**3*(1-t)*knot_2+p2*t**4, list_koeff))
                
            for idx in range(cou_vs+1):
                me.vertices[sort_list[idx]].co.y += (new_vts[idx] + p0 - me.vertices[sort_list[idx]].co.y)*influe
            
        if mode[2]:
            p0 = p0_.z
            p1 = p1_.z - p0
            p2 = p2_.z - p0
            knot_1 = kn1_.z - p0
            knot_2 = kn2_.z - p0
            
            t = pa_perc
            if knot_1==0 and p1!=0:
                b = (p1-(3*knot_2*t**2*(1-t)+p2*t**3))/(3*t*(1-t)**2+1e-8)
                new_vts = list(map(lambda t: 3*b*t*(1-t)**2+3*knot_2*t**2*(1-t)+p2*t**3, list_koeff))
            elif p1==0:
                new_vts = list(map(lambda t: 2*knot_2*t*(1-t)+p2*t**2, list_koeff))
            elif knot_2==p2 and p1!=p2:
                b = (p1-(3*knot_1*t*(1-t)**2+p2*t**3))/(3*t**2*(1-t)+1e-8)
                new_vts = list(map(lambda t: 3*knot_1*t*(1-t)**2+3*b*t**2*(1-t)+p2*t**3, list_koeff))
            elif p1==p2:
                new_vts = list(map(lambda t: 2*knot_1*t*(1-t)+p2*t**2, list_koeff))
            else:
                b = (p1-(4*knot_1*t*(1-t)**3+4*t**3*(1-t)*knot_2+p2*t**4))/(4*t**2*(1-t)**2+1e-8)
                new_vts = list(map(lambda t: 4*knot_1*t*(1-t)**3+4*b*t**2*(1-t)**2+4*t**3*(1-t)*knot_2+p2*t**4, list_koeff))
            
            if mode[3]:
                for c in range(steps_smoose):
                    new_vts_ = new_vts
                    V = [vi for vi in new_vts_]
                    P = list(map(lambda x,y: abs(y-x), V[:-1], V[1:]))
                    L = sum(P)
                    lp = len(P)
                    d = L/lp    
                    l_ = list(map(lambda y: d*y/L, list(range(1,lp+1))))
                    l = list(map(lambda x: x/L, P))

                    tmp = 1e-8
                    for i in range(lp):
                        tmp += l[i]
                        m = l_[i]/tmp
                        list_koeff[i] = m*list_koeff[i]
                    if knot_1==0 and p1!=0:
                        b = (p1-(3*knot_2*t**2*(1-t)+p2*t**3))/(3*t*(1-t)**2+1e-8)
                        new_vts = list(map(lambda t: 3*b*t*(1-t)**2+3*knot_2*t**2*(1-t)+p2*t**3, list_koeff))
                    elif p1==0:
                        new_vts = list(map(lambda t: 2*knot_2*t*(1-t)+p2*t**2, list_koeff))
                    elif knot_2==p2 and p1!=p2:
                        b = (p1-(3*knot_1*t*(1-t)**2+p2*t**3))/(3*t**2*(1-t)+1e-8)
                        new_vts = list(map(lambda t: 3*knot_1*t*(1-t)**2+3*b*t**2*(1-t)+p2*t**3, list_koeff))
                    elif p1==p2:
                        new_vts = list(map(lambda t: 2*knot_1*t*(1-t)+p2*t**2, list_koeff))
                    else:
                        b = (p1-(4*knot_1*t*(1-t)**3+4*t**3*(1-t)*knot_2+p2*t**4))/(4*t**2*(1-t)**2+1e-8)
                        new_vts = list(map(lambda t: 4*knot_1*t*(1-t)**3+4*b*t**2*(1-t)**2+4*t**3*(1-t)*knot_2+p2*t**4, list_koeff))
            
            for idx in range(cou_vs+1):
                me.vertices[sort_list[idx]].co.z += (new_vts[idx] + p0 - me.vertices[sort_list[idx]].co.z)*influe
            
        me.update()   
        bm.free() 
        
        bpy.ops.object.mode_set(mode='EDIT')  
            
    return True


def main_Basier_mid(context, mode, influe):
    global steps_smoose
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
            print('Error: 01 Basier_mid')
            return False
        
        
        sort_list = find_all_connected_verts(me,extreme_vs[0],[])
        all_vts_sort_x = [me.vertices[i].co.x for i in sort_list]
        all_vts_sort_y = [me.vertices[i].co.y for i in sort_list]
        all_vts_sort_z = [me.vertices[i].co.z for i in sort_list]
        
        max_p = [max(all_vts_sort_x), max(all_vts_sort_y), max(all_vts_sort_z)]
        min_p = [min(all_vts_sort_x), min(all_vts_sort_y), min(all_vts_sort_z)]
        diap_p = list(map(lambda a,b: a-b, max_p, min_p))
        
        if len(sort_list) != len(verts):
            print_error('Incoherent loop')
            print('Error: 020 Basier_mid')
            return False
        
        bpy.ops.object.mode_set(mode='OBJECT') 
        bm = bmesh.new()
        bm.from_mesh(me)
        check_lukap(bm)
        
        pa_idx = bm_vert_active_get(bm)[0]
        if pa_idx==None:
            bm.free() 
            print_error('Active vert is not detected')
            print('Error: 030 Basier_mid')
            return False
        
        pa_sort = sort_list.index(pa_idx)
        
        
        
        list_length_a = []
        list_length_b = []
        sum_length_a = 0.0
        sum_length_b = 0.0
        for sl in range(pa_sort-1):
            subb = me.vertices[sort_list[sl+1]].co-me.vertices[sort_list[sl]].co
            sum_length_a += subb.length
            list_length_a.append(sum_length_a)
        for sl in range(cou_vs-pa_sort-1):
            subb = me.vertices[sort_list[sl+2+pa_sort]].co-me.vertices[sort_list[sl+1+pa_sort]].co
            sum_length_b += subb.length
            list_length_b.append(sum_length_b)
            
        
        
        
        
        list_koeff_a = []
        list_koeff_b = []
        for sl in range(len(list_length_a)):
            tmp = list_length_a[sl]/sum_length_a
            list_koeff_a.append(tmp)
        for sl in range(len(list_length_b)):
            tmp = list_length_b[sl]/sum_length_b
            list_koeff_b.append(tmp)
        
        list_koeff_a = [0]+list_koeff_a
        list_koeff_b = [0]+list_koeff_b
        
        if pa_sort == 0: 
            kn1_i = sort_list[0]
            kn2_i = sort_list[pa_sort+1]
        elif pa_sort == len(sort_list)-1:
            kn1_i = sort_list[pa_sort-1]
            kn2_i = sort_list[-1]
        else:
            kn1_i = sort_list[pa_sort-1]
            kn2_i = sort_list[pa_sort+1]
        
        
        
        nkn1_ = me.vertices[kn1_i].co
        nkn2_ = me.vertices[kn2_i].co
        
        p0_ = me.vertices[sort_list[0]].co
        p1_ = me.vertices[pa_idx].co
        p2_ = me.vertices[sort_list[-1]].co
        
        kn1_ = nkn1_ - p1_ + nkn1_
        kn2_ = nkn2_ - p1_ + nkn2_
        
        if mode[3]:
            la = len(list_koeff_a)-1
            lb = len(list_koeff_b)-1
            if la==0: da=0
            else: da = 1/la
            
            if lb==0: db=0
            else: db = 1/lb
            
            list_koeff_a = list(map(lambda n: da*n, list(range(0,la+1))))
            list_koeff_b = list(map(lambda n: db*n, list(range(0,lb+1))))
        
        
        if mode[0]:
            p0 = p0_.x
            p1 = p1_.x - p0
            p2 = p2_.x - p0
            knot_1 = kn1_.x - p0
            knot_2 = kn2_.x - p0
            pA = nkn1_.x - p0
            pB = nkn2_.x - p0
            nkn1 = nkn1_.x - p0
            nkn2 = nkn2_.x - p0
            
            if nkn1==0 or p1==0:
                new_vts_a = []
                new_vts_b = list(map(lambda t: pB*(1-t)**2+2*knot_2*t*(1-t)+p2*t**2, list_koeff_b))
            elif nkn2==p2 or p1==p2:
                new_vts_a = list(map(lambda t: 2*knot_1*t*(1-t)+pA*t**2, list_koeff_a))
                new_vts_b = []
            else:
                new_vts_a = list(map(lambda t: 2*knot_1*t*(1-t)+pA*t**2, list_koeff_a))
                new_vts_b = list(map(lambda t: pB*(1-t)**2+2*knot_2*t*(1-t)+p2*t**2, list_koeff_b))
            
            if mode[3]:
                for c in range(steps_smoose):
                    new_vts_ = new_vts_a
                    V = [vi for vi in new_vts_]
                    P = list(map(lambda x,y: abs(y-x), V[:-1], V[1:]))
                    L = sum(P)
                    lp = len(P)
                    if lp>0:
                        d = L/lp    
                        l_ = list(map(lambda y: d*y/L, list(range(1,lp+1))))
                        l = list(map(lambda x: x/L, P))

                        tmp = 1e-8
                        for i in range(lp):
                            tmp += l[i]
                            m = l_[i]/tmp
                            list_koeff_a[i] = m*list_koeff_a[i]
                        if nkn1==0 or p1==0:
                            new_vts_a = []
                            new_vts_b = list(map(lambda t: pB*(1-t)**2+2*knot_2*t*(1-t)+p2*t**2, list_koeff_b))
                        elif nkn2==p2 or p1==p2:
                            new_vts_a = list(map(lambda t: 2*knot_1*t*(1-t)+pA*t**2, list_koeff_a))
                            new_vts_b = []
                        else:
                            new_vts_a = list(map(lambda t: 2*knot_1*t*(1-t)+pA*t**2, list_koeff_a))
                            new_vts_b = list(map(lambda t: pB*(1-t)**2+2*knot_2*t*(1-t)+p2*t**2, list_koeff_b))
                    
                    
                    
                    new_vts_ = new_vts_b
                    V = [vi for vi in new_vts_]
                    P = list(map(lambda x,y: abs(y-x), V[:-1], V[1:]))
                    L = sum(P)
                    lp = len(P)
                    if lp>0:
                        d = L/lp    
                        l_ = list(map(lambda y: d*y/L, list(range(1,lp+1))))
                        l = list(map(lambda x: x/L, P))

                        tmp = 1e-8
                        for i in range(lp):
                            tmp += l[i]
                            m = l_[i]/tmp
                            list_koeff_b[i] = m*list_koeff_b[i]
                        if nkn1==0 or p1==0:
                            new_vts_a = []
                            new_vts_b = list(map(lambda t: pB*(1-t)**2+2*knot_2*t*(1-t)+p2*t**2, list_koeff_b))
                        elif nkn2==p2 or p1==p2:
                            new_vts_a = list(map(lambda t: 2*knot_1*t*(1-t)+pA*t**2, list_koeff_a))
                            new_vts_b = []
                        else:
                            new_vts_a = list(map(lambda t: 2*knot_1*t*(1-t)+pA*t**2, list_koeff_a))
                            new_vts_b = list(map(lambda t: pB*(1-t)**2+2*knot_2*t*(1-t)+p2*t**2, list_koeff_b))
                    
            
            if new_vts_a:
                for idx in range(pa_sort):
                    me.vertices[sort_list[idx]].co.x += (new_vts_a[idx] + p0 - me.vertices[sort_list[idx]].co.x)*influe
            if new_vts_b:
                for idx in range(cou_vs-pa_sort):
                    me.vertices[sort_list[idx+pa_sort+1]].co.x += (new_vts_b[idx] + p0 - \
                    me.vertices[sort_list[idx+pa_sort+1]].co.x)*influe
            
        if mode[1]:
            p0 = p0_.y
            p1 = p1_.y - p0
            p2 = p2_.y - p0
            knot_1 = kn1_.y - p0
            knot_2 = kn2_.y - p0
            pA = nkn1_.y - p0
            pB = nkn2_.y - p0
            nkn1 = nkn1_.y - p0
            nkn2 = nkn2_.y - p0
            
            if nkn1==0 or p1==0:
                new_vts_a = []
                new_vts_b = list(map(lambda t: pB*(1-t)**2+2*knot_2*t*(1-t)+p2*t**2, list_koeff_b))
            elif nkn2==p2 or p1==p2:
                new_vts_a = list(map(lambda t: 2*knot_1*t*(1-t)+pA*t**2, list_koeff_a))
                new_vts_b = []
            else:
                new_vts_a = list(map(lambda t: 2*knot_1*t*(1-t)+pA*t**2, list_koeff_a))
                new_vts_b = list(map(lambda t: pB*(1-t)**2+2*knot_2*t*(1-t)+p2*t**2, list_koeff_b))
            
            if mode[3]:
                for c in range(steps_smoose):
                    new_vts_ = new_vts_a
                    V = [vi for vi in new_vts_]
                    P = list(map(lambda x,y: abs(y-x), V[:-1], V[1:]))
                    L = sum(P)
                    lp = len(P)
                    if lp>0:
                        d = L/lp    
                        l_ = list(map(lambda y: d*y/L, list(range(1,lp+1))))
                        l = list(map(lambda x: x/L, P))

                        tmp = 1e-8
                        for i in range(lp):
                            tmp += l[i]
                            m = l_[i]/tmp
                            list_koeff_a[i] = m*list_koeff_a[i]
                        if nkn1==0 or p1==0:
                            new_vts_a = []
                            new_vts_b = list(map(lambda t: pB*(1-t)**2+2*knot_2*t*(1-t)+p2*t**2, list_koeff_b))
                        elif nkn2==p2 or p1==p2:
                            new_vts_a = list(map(lambda t: 2*knot_1*t*(1-t)+pA*t**2, list_koeff_a))
                            new_vts_b = []
                        else:
                            new_vts_a = list(map(lambda t: 2*knot_1*t*(1-t)+pA*t**2, list_koeff_a))
                            new_vts_b = list(map(lambda t: pB*(1-t)**2+2*knot_2*t*(1-t)+p2*t**2, list_koeff_b))
                    
                    
                    
                    new_vts_ = new_vts_b
                    V = [vi for vi in new_vts_]
                    P = list(map(lambda x,y: abs(y-x), V[:-1], V[1:]))
                    L = sum(P)
                    lp = len(P)
                    if lp>0:
                        d = L/lp    
                        l_ = list(map(lambda y: d*y/L, list(range(1,lp+1))))
                        l = list(map(lambda x: x/L, P))

                        tmp = 1e-8
                        for i in range(lp):
                            tmp += l[i]
                            m = l_[i]/tmp
                            list_koeff_b[i] = m*list_koeff_b[i]
                        if nkn1==0 or p1==0:
                            new_vts_a = []
                            new_vts_b = list(map(lambda t: pB*(1-t)**2+2*knot_2*t*(1-t)+p2*t**2, list_koeff_b))
                        elif nkn2==p2 or p1==p2:
                            new_vts_a = list(map(lambda t: 2*knot_1*t*(1-t)+pA*t**2, list_koeff_a))
                            new_vts_b = []
                        else:
                            new_vts_a = list(map(lambda t: 2*knot_1*t*(1-t)+pA*t**2, list_koeff_a))
                            new_vts_b = list(map(lambda t: pB*(1-t)**2+2*knot_2*t*(1-t)+p2*t**2, list_koeff_b))
            
            if new_vts_a:
                for idx in range(pa_sort):
                    me.vertices[sort_list[idx]].co.y += (new_vts_a[idx] + p0 - me.vertices[sort_list[idx]].co.y)*influe
            if new_vts_b:
                for idx in range(cou_vs-pa_sort):
                    me.vertices[sort_list[idx+pa_sort+1]].co.y += (new_vts_b[idx] + p0 - \
                    me.vertices[sort_list[idx+pa_sort+1]].co.y)*influe
            
        if mode[2]:
            p0 = p0_.z
            p1 = p1_.z - p0
            p2 = p2_.z - p0
            knot_1 = kn1_.z - p0
            knot_2 = kn2_.z - p0
            pA = nkn1_.z - p0
            pB = nkn2_.z - p0
            nkn1 = nkn1_.z - p0
            nkn2 = nkn2_.z - p0
            
            if nkn1==0 or p1==0:
                new_vts_a = []
                new_vts_b = list(map(lambda t: pB*(1-t)**2+2*knot_2*t*(1-t)+p2*t**2, list_koeff_b))
            elif nkn2==p2 or p1==p2:
                new_vts_a = list(map(lambda t: 2*knot_1*t*(1-t)+pA*t**2, list_koeff_a))
                new_vts_b = []
            else:
                new_vts_a = list(map(lambda t: 2*knot_1*t*(1-t)+pA*t**2, list_koeff_a))
                new_vts_b = list(map(lambda t: pB*(1-t)**2+2*knot_2*t*(1-t)+p2*t**2, list_koeff_b))
            
            if mode[3]:
                for c in range(steps_smoose):
                    new_vts_ = new_vts_a
                    V = [vi for vi in new_vts_]
                    P = list(map(lambda x,y: abs(y-x), V[:-1], V[1:]))
                    L = sum(P)
                    lp = len(P)
                    if lp>0:
                        d = L/lp    
                        l_ = list(map(lambda y: d*y/L, list(range(1,lp+1))))
                        l = list(map(lambda x: x/L, P))

                        tmp = 1e-8
                        for i in range(lp):
                            tmp += l[i]
                            m = l_[i]/tmp
                            list_koeff_a[i] = m*list_koeff_a[i]
                        if nkn1==0 or p1==0:
                            new_vts_a = []
                            new_vts_b = list(map(lambda t: pB*(1-t)**2+2*knot_2*t*(1-t)+p2*t**2, list_koeff_b))
                        elif nkn2==p2 or p1==p2:
                            new_vts_a = list(map(lambda t: 2*knot_1*t*(1-t)+pA*t**2, list_koeff_a))
                            new_vts_b = []
                        else:
                            new_vts_a = list(map(lambda t: 2*knot_1*t*(1-t)+pA*t**2, list_koeff_a))
                            new_vts_b = list(map(lambda t: pB*(1-t)**2+2*knot_2*t*(1-t)+p2*t**2, list_koeff_b))
                    
                    
                    
                    new_vts_ = new_vts_b
                    V = [vi for vi in new_vts_]
                    P = list(map(lambda x,y: abs(y-x), V[:-1], V[1:]))
                    L = sum(P)
                    lp = len(P)
                    if lp>0:
                        d = L/lp    
                        l_ = list(map(lambda y: d*y/L, list(range(1,lp+1))))
                        l = list(map(lambda x: x/L, P))

                        tmp = 1e-8
                        for i in range(lp):
                            tmp += l[i]
                            m = l_[i]/tmp
                            list_koeff_b[i] = m*list_koeff_b[i]
                        if nkn1==0 or p1==0:
                            new_vts_a = []
                            new_vts_b = list(map(lambda t: pB*(1-t)**2+2*knot_2*t*(1-t)+p2*t**2, list_koeff_b))
                        elif nkn2==p2 or p1==p2:
                            new_vts_a = list(map(lambda t: 2*knot_1*t*(1-t)+pA*t**2, list_koeff_a))
                            new_vts_b = []
                        else:
                            new_vts_a = list(map(lambda t: 2*knot_1*t*(1-t)+pA*t**2, list_koeff_a))
                            new_vts_b = list(map(lambda t: pB*(1-t)**2+2*knot_2*t*(1-t)+p2*t**2, list_koeff_b))
                    
            
            if new_vts_a:
                for idx in range(pa_sort):
                    me.vertices[sort_list[idx]].co.z += (new_vts_a[idx] + p0 - me.vertices[sort_list[idx]].co.z)*influe
            if new_vts_b:
                for idx in range(cou_vs-pa_sort):
                    me.vertices[sort_list[idx+pa_sort+1]].co.z += (new_vts_b[idx] + p0 - \
                    me.vertices[sort_list[idx+pa_sort+1]].co.z)*influe
            
        me.update()   
        bm.free() 
        
        bpy.ops.object.mode_set(mode='EDIT')  
            
    return True


def getMats(context):
    global list_z, mats_idx, list_f, maloe
    
    obj = bpy.context.active_object
    me = obj.data
    
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.mode_set(mode='EDIT') 
    bpy.ops.mesh.select_mode(type='VERT')
    
    list_z = [v.co.z for v in me.vertices if v.select]
    list_z = list(set(list_z))
    list_z.sort()
    tz = list_z[0]
    tmp_z = [tz]
    for lz in list(list_z)[1:]:
        if round(abs(lz-tz),4)>maloe:
            tmp_z.append(lz)
            tz = lz
    list_z = tmp_z
    
    bpy.ops.mesh.select_mode(type='FACE')
    list_f = [p.index for p in me.polygons if p.select]
    black_list = []
    mats_idx = []
    for z in list_z:
        for p in list_f:
            if p not in black_list:
                for v in me.polygons[p].vertices:
                    if abs(me.vertices[v].co.z-z)<maloe:
                        mats_idx.append(me.polygons[p].material_index)
                        black_list.append(p)
                        break
    bpy.ops.mesh.select_mode(type='VERT')



def main_matExtrude(context):
    global list_z, mats_idx, list_f, maloe
    
    obj = bpy.context.active_object
    me = obj.data
    
    bpy.ops.object.mode_set(mode='OBJECT')
    vert = [v.index for v in me.vertices if v.select][0]
    
    
    def find_index_of_selected_vertex(obj):  
        # force 'OBJECT' mode temporarily. [TODO]  
        selected_verts = [i.index for i in obj.data.vertices if i.select]  
        verts_selected = len(selected_verts)  
        if verts_selected <1:                   
            return None  
        else:  
            return selected_verts 
        
        
    def find_connected_verts2(me, found_index, not_list):  
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
        
        
    def find_all_connected_verts2(me, active_v, not_list=[], step=0):
        vlist = [active_v]
        not_list.append(active_v)
        step+=1
        list_v_1 = find_connected_verts2(me, active_v, not_list)
        
        for v in list_v_1:
            list_v_2 = find_all_connected_verts2(me, v, not_list, step) 
            vlist += list_v_2
                     
        return vlist  
    
    
    bm = bmesh.new()
    bm.from_mesh(me)
    check_lukap(bm)
    
    verts = find_all_connected_verts2(me,vert,not_list=[], step=0)
    vert_ = find_extreme_select_verts(me, verts)
    vert = vert_[0]
    verts = find_all_connected_verts2(me,vert,not_list=[], step=0)
    
    vts = [bm.verts[vr] for vr in verts]
    face_build = []
    face_build.extend(verts)
    fl = len(bm.verts)+1
    
    tz = list_z[0]
    tmp_lz = [z-tz for z in list_z]
    list_z = tmp_lz
    
    z_nul = vts[0].co.z
    for zidx,z in enumerate(list_z):
        vts_tmp = []
        matz = mats_idx[min(zidx, len(mats_idx)-1)]
        
        for i,vs in enumerate(vts[:-1]):
            vco1 = vs.co
            vco2 = vts[i+1].co
            vco1.z = z+z_nul
            vco2.z = z+z_nul
            if i==0:
                v1 = bm.verts.new(vco1)
                face_build.append(len(bm.verts)-1)
            else:
                v1=v2
            v2 = bm.verts.new(vco2)
            face_build.append(len(bm.verts)-1)
            f = bm.faces.new([vs,v1,v2,vts[i+1]])
            #check_lukap(bm)
            f.material_index = matz
            
            if i==0:
                vts_tmp.append(v1)
            vts_tmp.append(v2)
        vts = vts_tmp.copy()
        
    bpy.ops.object.mode_set(mode='OBJECT')
    bm.to_mesh(me) 
    bm.free() 
    
    bpy.ops.object.mode_set(mode='EDIT') 
    bpy.ops.mesh.select_mode(type='FACE')
    bpy.ops.mesh.select_all(action = 'DESELECT')
    bpy.ops.object.mode_set(mode='OBJECT')
    for p in face_build:
        me.vertices[p].select = True
    bpy.ops.object.mode_set(mode='EDIT') 
    bpy.ops.mesh.select_mode(type='VERT')
    bpy.ops.mesh.remove_doubles()


def cheredator(step=1):
    bpy.ops.object.mode_set(mode='OBJECT') 
    bpy.ops.object.mode_set(mode='EDIT') 
    
    obj = bpy.context.active_object
    me = obj.data

    verts = find_index_of_selected_vertex(me)
    active = None
    if verts != None:
        extreme_vs = find_extreme_select_verts(me, verts)
        if extreme_vs==[]:
            bm = bmesh.new()
            bm.from_mesh(me) 
            check_lukap(bm)
            active = bm_vert_active_get(bm)[0]
            extreme_vs = [active,active]
            bm.free()
        elif len(extreme_vs) != 2:
            print_error2('Single Loop only', '01 cheredator')
            return False
        
        sort_list = find_all_connected_verts(me,extreme_vs[0],[])
        if len(sort_list) != len(verts) and not active:
            print_error2('Incoherent loop','02 cheredator')
            return False
        
        if len(sort_list) <3:
            print_error2('Should be greater than two vertices','03 cheredator')
            return False
        
        work_list = sort_list[1:-1]
        
        bpy.ops.mesh.select_all(action='DESELECT')
        bpy.ops.mesh.select_mode(type='VERT')
        bpy.ops.object.mode_set(mode='OBJECT')
        
        v_memory = [str(me.vertices[extreme_vs[0]].co)]
        v_memory.append(str(me.vertices[extreme_vs[1]].co))
        most = False
        data_verts = []
        step_tmp = 0
        for i in work_list:
            step_tmp += 1
            if step_tmp>=step:
                most = True
                step_tmp = 0
            if most: 
                me.vertices[i].select = False
                v_memory.append(str(me.vertices[i].co))
                most = False
            else:
                me.vertices[i].select = True
        
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.dissolve_verts()
        bpy.ops.mesh.select_all(action='DESELECT') 
        bpy.ops.object.mode_set(mode='OBJECT')
        
        for v in me.vertices:
            if str(v.co) in v_memory:
                v.select = True
                
        bpy.ops.object.mode_set(mode='EDIT') 
        
        return True


def cheredator_fantom(self):
    step = self.steps
    bpy.ops.object.mode_set(mode='OBJECT') 
    bpy.ops.object.mode_set(mode='EDIT') 
    
    obj = bpy.context.active_object
    me = obj.data

    verts = find_index_of_selected_vertex(me)
    active = None
    if verts != None:
        extreme_vs = find_extreme_select_verts(me, verts)
        if extreme_vs==[]:
            bm = bmesh.new()
            bm.from_mesh(me) 
            check_lukap(bm)
            active = bm_vert_active_get(bm)[0]
            extreme_vs = [active,active]
            bm.free()
        elif len(extreme_vs) != 2:
            print_error2('Single Loop only', '01 cheredator_fantom')
            return False
        sort_list = find_all_connected_verts(me,extreme_vs[0],[])
        
        if len(sort_list) != len(verts) and not active:
            print_error2('Incoherent loop','02 cheredator_fantom')
            return False
        
        if len(sort_list) <3:
            print_error2('Should be greater than two vertices','03 cheredator_fantom')
            return False
        
        work_list = sort_list[1:-1]
        if self.steps>len(work_list):
            self.steps = len(work_list)
        
        bpy.ops.mesh.select_mode(type='VERT')
        
        most = False
        data_verts = []
        step_tmp = 0
        for i in work_list:
            step_tmp += 1
            if step_tmp>=step:
                most = True
                step_tmp = 0
            if most: 
                data_verts.append(me.vertices[i].co)
                most = False
        
        data_verts.insert(0, me.vertices[extreme_vs[0]].co)
        data_verts.append(me.vertices[extreme_vs[1]].co)
        drawPointLineGL(obj.matrix_world, data_verts)
    return True
    

def drawPointLineGL(data_matrix, data_vector):
    from bgl import glVertex3f, glPointSize, glLineStipple, \
                glLineWidth, glBegin, glEnd, GL_POINTS, GL_LINES, \
                glEnable, glDisable, GL_BLEND, glColor4f, GL_LINE_STRIP, GL_LINE_STIPPLE
    
    glLineWidth(1.0)
    glEnable(GL_BLEND)
    # points
    glPointSize(6.0)
    glColor4f(1.0, 0.0, 0.0, 1.0)
    glBegin(GL_POINTS)
    for vert in data_vector:
        vec_corrected = data_matrix*vert
        glVertex3f(*vec_corrected)
    glEnd()
    
    # lines
    glLineWidth(3.0)
    glBegin(GL_LINE_STRIP)
    glColor4f(0.0, 1.0, 0.0, 1.0)
    for i, vector in enumerate(data_vector):
        glVertex3f(*data_matrix*data_vector[i])
    glEnd()
    
    # restore opengl defaults
    glDisable(GL_BLEND)
    glLineWidth(1.0)
    glColor4f(0.0, 0.0, 0.0, 1.0)
    

def DDDLoop():
    obj = bpy.context.active_object
    if obj.type != 'MESH':
        print_error('You need to select object of MESH type', '3DL_00')
        return False
    
    bpy.ops.object.mode_set(mode='OBJECT') 
    bpy.ops.object.mode_set(mode='EDIT')  
    me = obj.data

    selected_verts = [i.index for i in me.vertices if i.select] 
    if not selected_verts:
        print_error('You need to select two loops', '3DL_01')
        return False
        
        
    act_v = selected_verts[0]
    black_list=[]
    loop_1 = find_all_connected_verts(me, act_v, black_list, 0)
    white_list = [i for i in selected_verts if i not in black_list]
    if not white_list:
        print_error('You need to select two loops', '3DL_02')
        return False

    loop_2 = find_all_connected_verts(me, white_list[0], [], 0)

    loop1_x = [me.vertices[idx].co.x for idx in loop_1]
    loop1_y = [me.vertices[idx].co.y for idx in loop_1]

    direct1_x = max(loop1_x)-min(loop1_x)    
    direct1_y = max(loop1_y)-min(loop1_y)
    if direct1_x > direct1_y:
        loop_xz = loop_1
        loop_yz = loop_2
    else:
        loop_xz = loop_2
        loop_yz = loop_1
    
    tmp_x = [(me.vertices[idx].co.z, idx) for idx in loop_xz]
    tmp_y = [(me.vertices[idx].co.z, idx) for idx in loop_yz]
    tmp_x.sort()
    tmp_y.sort()
    loop_xz = [p[1] for p in tmp_x]
    loop_yz = [p[1] for p in tmp_y]
    
    lz_xz = [me.vertices[idx].co.z for idx in loop_xz]
    lz_yz = [me.vertices[idx].co.z for idx in loop_yz]
    lz = lz_xz+lz_yz
    
    lz.sort()
    points = []
    
    plxz = me.vertices[loop_xz[0]].co
    for lxz in loop_xz[1:]:
        co = me.vertices[lxz].co
        for z in lz:
            if z>=plxz.z and z<=co.z:
                delitel = co.z-plxz.z
                if delitel==0: delitel=1e-6
                x = (z-plxz.z)*(co.x-plxz.x)/delitel+plxz.x
                points.append([z, 0, x])
            elif z>co.z:
                continue
        plxz = co
            
    plyz = me.vertices[loop_yz[0]].co
    for lyz in loop_yz[1:]:
        co = me.vertices[lyz].co
        for z in lz:
            if z>=plyz.z and z<=co.z:
                delitel = co.z-plyz.z
                if delitel==0: delitel=1e-6
                y = (z-plyz.z)*(co.y-plyz.y)/delitel+plyz.y
                points.append([z, y, 0])
            elif z>co.z:
                continue
        plyz = co
    
    points.sort()
    plxz = me.vertices[loop_xz[0]].co
    for lxz in loop_xz[1:]:
        co = me.vertices[lxz].co
        for idx, p in enumerate(points):
            if p[0]>=plxz.z and p[0]<=co.z and p[2]==0:
                delitel = co.z-plxz.z
                if delitel==0: delitel=1e-6
                x = (p[0]-plxz.z)*(co.x-plxz.x)/delitel+plxz.x
                points[idx] = (p[0], p[1], x)
            elif p[0]>co.z:
                continue
        plxz = co
                
    plyz = me.vertices[loop_yz[0]].co
    for lyz in loop_yz[1:]:
        co = me.vertices[lyz].co
        for idx, p in enumerate(points):
            if p[0]>=plyz.z and p[0]<=co.z and p[1]==0:
                delitel = co.z-plyz.z
                if delitel==0: delitel=1e-6
                y = (p[0]-plyz.z)*(co.y-plyz.y)/delitel+plyz.y
                points[idx] = (p[0], y, p[2])
            elif p[0]>co.z:
                continue
        plyz = co
    
    points_=[]
    for p in points[1:]:
        points_.append(mathutils.Vector(reversed(p)))
    
    edges = []
    lvs = len(me.vertices)
    for idx, po in enumerate(points_[:-1]):
        edges.append([idx, idx+1])
    
    bpy.ops.mesh.select_all(action='TOGGLE')
    bpy.ops.object.mode_set(mode='OBJECT')
    
    nam = 'slurm_' + str(obj.name)
    mesh = bpy.data.meshes.new(nam+'Mesh') 
    ob = mk_ob(mesh,nam, obj.location)
    
    mesh.from_pydata(points_, edges, [])
    mesh.update(calc_edges=True) 
    ob.select=True

    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.join()
     
    bpy.ops.object.mode_set(mode='EDIT') 
    maloe = bpy.context.scene.tool_settings.double_threshold
    bpy.ops.mesh.remove_doubles(threshold=maloe, use_unselected=False)  


def barc(rad):
    bpy.ops.object.mode_set(mode='OBJECT') 
    bpy.ops.object.mode_set(mode='EDIT') 
    
    obj = bpy.context.active_object
    me = obj.data

    verts = find_index_of_selected_vertex(me)
    cou_vs = len(verts) - 1
    if verts != None and cou_vs>0:
        extreme_vs = find_extreme_select_verts(me, verts)
        if len(extreme_vs) != 2:
            print_error2('Single Loop only', '01 barc')
            return False
        
        sort_list = find_all_connected_verts(me,extreme_vs[0],[])
        if len(sort_list) != len(verts):
            print_error2('Incoherent loop','02 barc')
            return False
        
        bpy.ops.object.mode_set(mode='OBJECT') 
        bm = bmesh.new()
        bm.from_mesh(me)
        check_lukap(bm)
        
        pa_idx = bm_vert_active_get(bm)[0]
        if pa_idx==None:
            print_error2('Active vert is not detected','03 barc')
            return False
        
        p_a_ = me.vertices[extreme_vs[0]].co
        p_b_ = me.vertices[pa_idx].co
        p_c_ = me.vertices[extreme_vs[1]].co
        
        normal_B = getNormalPlane([p_a_, p_b_, p_c_], mathutils.Matrix())
        normal_z = mathutils.Vector((0,0,-1))
        mat_rot_norm = normal_B.rotation_difference(normal_z).to_matrix().to_4x4()
        
        p_a = mat_rot_norm * p_a_
        p_b = mat_rot_norm * p_b_
        p_c = mat_rot_norm * p_c_
        p_ab = (p_a+p_b)/2
        p_bc = (p_b+p_c)/2
        ab = p_b-p_a
        bc = p_c-p_b
        k_ab = -ab.y/(ab.x+1e-7)
        k_bc = -bc.y/(bc.x+1e-7)
        z = p_a.z
        ab_d = mathutils.Vector((k_ab,1,0)).normalized()
        bc_d = mathutils.Vector((k_bc,1,0)).normalized()
        p_d_ = mathutils.geometry.intersect_line_line(p_ab, p_ab+ab_d, p_bc, p_bc+bc_d)
        if p_d_==None:
            print_error2('Impossible to construct the arc radius','04 barc')
            return False
        
        p_d = p_d_[0]
        ad = p_a-p_d
        config = bpy.context.window_manager.paul_manager
        if rad!=None:
            radius = rad
            ac = p_c-p_a
            p_d_ = p_a+ac/2
            ac_div_2_len = ac.length/2
            k_ac = -ac.y/(ac.x+1e-7)
            ac_d = mathutils.Vector((k_ac,1,0)).normalized()
            if rad<ac_div_2_len:
                radius = ac_div_2_len
            
            l1 = (p_b-(p_d_+ac_d)).length
            l2 = (p_b-(p_d_-ac_d)).length
            if l2>l1:
                ac_d = -ac_d
            
            tmp_ld = sqrt(radius**2-ac_div_2_len**2)
            p_d = tmp_ld*ac_d+p_d_
            ad = p_a-p_d
            
        else:
            radius = ad.length
        
        config.barc_rad = radius
        angle = ad.angle(p_c-p_d)
        section_angle = angle/(len(sort_list)-1)
        vector_zero = mathutils.Vector((1,0,0))
        angle_zero = pi/2+ad.angle(vector_zero)
        test_x = sin(section_angle*(len(sort_list)-1)+angle_zero)*radius+p_d.x
        test_y = cos(section_angle*(len(sort_list)-1)+angle_zero)*radius+p_d.y
        test_by_x = abs(test_x-p_c.x)<maloe
        test_by_y = abs(test_y-p_c.y)<maloe
        if not test_by_x or not test_by_y:
            angle_zero = pi/2-ad.angle(vector_zero)
            test_x = sin(angle_zero)*radius+p_d.x
            test_y = cos(angle_zero)*radius+p_d.y
            test_by_x = abs(test_x-p_a.x)<maloe
            test_by_y = abs(test_y-p_a.y)<maloe
            if not test_by_x or not test_by_y:
                angle = 2*pi-angle
                angle_zero = pi/2+ad.angle(vector_zero)
                section_angle = angle/(len(sort_list)-1)
            else:
                test_x = sin(section_angle*(len(sort_list)-1)+angle_zero)*radius+p_d.x
                test_y = cos(section_angle*(len(sort_list)-1)+angle_zero)*radius+p_d.y
                test_by_x = abs(test_x-p_c.x)<maloe
                test_by_y = abs(test_y-p_c.y)<maloe
                if not test_by_x or not test_by_y:
                    angle = 2*pi-angle
                    angle_zero = pi/2-ad.angle(vector_zero)
                    section_angle = angle/(len(sort_list)-1)
        
        mat_rot_norm_inv = mat_rot_norm.inverted()
        for i, v_idx in enumerate(sort_list):
            x = sin(section_angle*i+angle_zero)*radius+p_d.x
            y = cos(section_angle*i+angle_zero)*radius+p_d.y
            me.vertices[v_idx].co = mat_rot_norm_inv * Vector((x,y,z))
        

def ignore_instance():
    names = {}
    for obj in bpy.data.objects:
        if not obj.select \
            or obj.type!='MESH' and obj.type!='CURVE': continue
        
        dataname = obj.type+obj.data.name 
        
        if dataname in names:
            obj.select = False
            bpy.data.objects[names[dataname]].select = False
        else:
            names[dataname]=obj.name


def select_modifiers_objs():
    for obj in bpy.data.objects:
        if obj.type!='MESH' and obj.type!='CURVE': continue
        obj.select = len(obj.modifiers)>0


def switch_matnodes():
    flag = False
    mode = False
    for mat in bpy.data.materials:
        if not flag:
            mode = not mat.use_nodes
            flag = True
        mat.use_nodes = mode
        
        
def all_mats_to_active():
    obj = bpy.context.scene.objects.active
    if obj:
        mats = []
        for mat in obj.material_slots:
            mats.append(mat.name)
        for m in bpy.data.materials:
            if m.name not in mats:
                bpy.ops.object.material_slot_add()
                obj.material_slots[-1].material = bpy.data.materials[m.name]
                

def select_2d_curves():
    for obj in bpy.data.objects:
        if not obj.select: continue
        if obj.type!='CURVE': 
            obj.select = False
        elif obj.data.dimensions!='2D':
            obj.select = False
        

def filter_dubles_origins():
    global maloe
    locs = []
    for obj in bpy.context.scene.objects:
        if obj.type=='CURVE' or obj.type=='MESH':
            loc = obj.location
            flag = False
            for l in locs:
                ll = (l-loc).length
                if ll < maloe:
                    flag = True
                    break
            
            obj.select = flag
            if not flag:
                locs.append(loc)
            

def swap_curve():
    i = None
    dim = {'2D':'3D', '3D':'2D'}
    for obj in bpy.context.selected_objects:
        if obj.type=='CURVE':
            if i==None:
                i = dim[obj.data.dimensions]
            obj.data.dimensions=i


def hue_2matneme():
    for matik in bpy.data.materials:
        H_ = min(round(matik.diffuse_color.h, 2),0.99)
        V_ = min(round(matik.diffuse_color.v, 2),0.99)
        S_ = min(round(matik.diffuse_color.s, 2),0.99)
        
        H = ("%.2f"%(H_))[-2:]
        V = ("%.2f"%(V_))[-2:]
        S = ("%.2f"%(S_))[-2:]
        
        name_ = matik.name.split('---')[-1]
        mat_name = H+V+S+'---'+name_
        matik.name = mat_name


def HVS_from_mathame():
    for matik in bpy.data.materials:
        mat_name = matik.name.split('---')[-1]
        matik.name = mat_name


def supress_materials():
    all_mats = {}
    del_mats = []
    for matik in bpy.data.materials:
        name = matik.name
        diffcolor = str(matik.diffuse_color.r)+str(matik.diffuse_color.g)+ \
                    str(matik.diffuse_color.b)
        if diffcolor in all_mats:
            del_mats.append(name)
        else:
            all_mats[diffcolor] = name

    for obj in bpy.data.objects:
        for m in obj.material_slots:
            mat = m.material
            if mat==None: continue
            diffcolor_ = str(mat.diffuse_color.r)+str(mat.diffuse_color.g)+ \
                         str(mat.diffuse_color.b)
            m.material = bpy.data.materials[all_mats[diffcolor_]]
            
    for del_m in del_mats:
        dm = bpy.data.materials[del_m]
        bpy.data.materials.remove(dm)


def matsUnclone():
    mats = bpy.data.materials
    for obj in bpy.data.objects:
        for slt in obj.material_slots:
            part = slt.name.rpartition('.')
            if part[2].isnumeric() and part[0] in mats:
                slt.material = mats.get(part[0])


def matsPurgeout():
    bpy.ops.object.mode_set(mode='OBJECT', toggle=False)
    for ob in bpy.context.selected_objects:
         bpy.context.scene.objects.active = ob
         if ob.type != 'MESH':
             continue
         mat_slots = {}
         for p in ob.data.polygons:
             mat_slots[p.material_index] = 1
         mat_slots = mat_slots.keys()
         for i in range(len(ob.material_slots)-1, -1, -1):
             if i not in mat_slots:
                 bpy.context.object.active_material_index = i
                 bpy.ops.object.material_slot_remove()


def matchProp():
    cont = bpy.context
    obj = cont.active_object
    pps_ = dir(obj)
    pps = [p for p in pps_ if p.find('show_')==0]
    wpps={}
    typ = obj.type
    for p in pps:
        if hasattr(obj,p):
            wpps[p]=getattr(obj,p)
        
    for o in cont.selected_objects:
        if obj is o: continue
        if o.type != typ: continue
        
        for p in wpps:
            if hasattr(o,p):
                 setattr(o,p,wpps[p])
    

############## PlyCams Render #################
class ExportSomeData(Operator, ExportHelper):
    """This appears in the tooltip of the operator and in the generated docs"""
    bl_idname = "export_test.some_data"  # important since its how bpy.ops.import_test.some_data is constructed
    bl_label = "Export Some Data"

    # ExportHelper mixin class uses this
    filename_ext = ""

    def execute(self, context):
        return render_me(self.filepath)
    

def render_me(filepath):
    sceneName = bpy.context.scene.name    
    glob_res = [bpy.context.scene.render.resolution_x, bpy.context.scene.render.resolution_y] 
    
    bpy.data.scenes[sceneName].render.filepath = filepath
    if not os.path.exists(filepath):
        os.mkdir(filepath)
    outputfile_s = os.path.join(filepath, 'stat.txt')
    with open(outputfile_s, 'w') as w_file:
        w_file.write('Batch stats:\n'+'_________________________________\n')
    
    
    camsi = []
    progress = 0
    sline = ''
    backet_x = bpy.context.scene.render.tile_x
    backet_y = bpy.context.scene.render.tile_y
    sq_backet = backet_x*backet_y
    rp = bpy.context.scene.render.resolution_percentage
    for cam in bpy.data.objects: 
        if ( cam.type =='CAMERA' and not cam.hide_render): 
            flag = False
            res = cam.data.name.split('(')
            res_x = glob_res[0]
            res_y = glob_res[1]
            if len(res)==2:
                res = res[1].split(')')
                if len(res)==2:
                    res = res[0].split('+')
                    if len(res)==2:
                        res_x = int(res[0])
                        res_y = int(res[1])
                        flag = True
                        
            camsi.append((res_x,res_y))
            p_tmp = res_x * res_y
            p_tmp_scale = round(p_tmp*rp/100)
            progress += p_tmp
            if flag:
                sline = sline + cam.name +' | '+str(res_x)+'x'+str(res_y)+' | '+ \
                    str(round(res_x*rp/100))+'x'+str(round(res_y*rp/100))+' | '+ \
                    str(round(p_tmp_scale/sq_backet))+'\n'
            else:
                sline = sline + cam.name +' | default | '+ \
                    str(round(res_x*rp/100))+'x'+str(round(res_y*rp/100))+' | '+ \
                    str(round(p_tmp_scale/sq_backet))+'\n'
            
    
    with open(outputfile_s, 'a') as w_file:
        w_file.write('Total resolution = '+str(round(math.sqrt(progress)))+'x'+str(round(math.sqrt(progress)))+ \
            ' ('+str(round(math.sqrt(progress)*rp/100))+'x'+str(round(math.sqrt(progress)*rp/100))+')'+'\n')
        w_file.write('Default Resolution = '+str(glob_res[0])+'x'+str(glob_res[1])+' ('+str(rp)+'%)'+'\n')
        w_file.write('Tiles = '+str(backet_x)+'x'+ str(backet_y)+'\n')
        w_file.write('Total tiles = '+str(round(progress*rp/(sq_backet*100)))+'\n\n')
        w_file.write('Cameras:\n'+'Name | resolution | scaled ('+str(rp)+'%) | tiles\n'+ \
        '________________________________________\n')
        w_file.write(sline)
        
        
    outputfile = os.path.join(filepath, 'log.txt')
    with open(outputfile, 'w') as w_file:
        w_file.write('Cameras:\n'+'Name | resolution | scaled ('+str(rp)+'%) | progress % | remaining time | elapsed time\n'+ \
        '_____________________________________________________________________________\n')
        
    p_tmp = 0
    time_start = time.time()   
    i = 0
    for cam in bpy.data.objects: 
        if ( cam.type =='CAMERA' and not cam.hide_render): 
            bpy.data.scenes[sceneName].camera = cam 
            bpy.context.scene.render.resolution_x = camsi[i][0]
            bpy.context.scene.render.resolution_y = camsi[i][1]
            bpy.data.scenes[sceneName].render.filepath = filepath+'\\'+cam.name
            bpy.ops.render.render(animation=False, write_still = True) 
            p_tmp += camsi[i][0]*camsi[i][1]
            proc = max(round(p_tmp*100/progress),1)
            r_time = time.time() - time_start
            time_tmp = r_time*(100-proc)/proc
            time_tmp = round(time_tmp)
            
            s_rt = time.strftime('%H:%M:%S',time.gmtime(r_time))
            s_lt = time.strftime('%H:%M:%S',time.gmtime(time_tmp))
            with open(outputfile, 'a') as w_file:
                w_file.write(cam.name +' | '+str(camsi[i][0])+'x'+str(camsi[i][1])+' | '+ \
                    str(round(camsi[i][0]*rp/100))+'x'+str(round(camsi[i][1]*rp/100))+' | '+ \
                    str(proc)+' | '+s_lt+' | '+s_rt+'\n')
            i += 1
    
    bpy.context.scene.render.resolution_x = glob_res[0]
    bpy.context.scene.render.resolution_y = glob_res[1]
    #print('Done!') 
    #print(bpy.data.scenes[sceneName].render.filepath)
    return {'FINISHED'}
    
                                         
class RenderMe(bpy.types.Operator):
   """Cams render"""
   bl_idname = "scene.render_me"
   bl_label = "Render Me"
   bl_options = {'REGISTER', 'UNDO'}

   def execute(self, context):
        bpy.ops.export_test.some_data('INVOKE_DEFAULT')
        return {'FINISHED'}

       
def menu_func(self, context):
    self.layout.operator(RenderMe.bl_idname)
def menu_func_export(self, context):
    self.layout.operator(ExportSomeData.bl_idname, text="Cams Render!")
############# End PolyCams Render ################

    
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
        split = col.split(percentage=0.15)
        if lt.disp_coll:
            split.prop(lt, "disp_coll", text="", icon='DOWNARROW_HLT')
        else:
            split.prop(lt, "disp_coll", text="", icon='RIGHTARROW')
        split.operator("mesh.simple_scale_operator", text='XYcollapse').type_op=0
        if lt.disp_coll:
            box = col.column(align=True).box().column()
            col_top = box.column(align=True)
            row = col_top.row(align=True)
            row.operator("mesh.simple_scale_operator", text='Get Orientation').type_op=1
            
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
            row.prop(lt, 'relation', text = 'Relation')
            box = box.box().column()
            row = box.row(align=True)
            row.prop(lt, 'shape_spline', text = 'Shape spline')
            row = box.row(align=True)
            row.active = lt.shape_spline
            row.prop(lt, 'spline_Bspline2', text = 'Smooth transition')
            row = box.row(align=True)
            
        
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
        if lt.disp_3drotor:
            split.prop(lt, "disp_3drotor", text="3D Rotor", icon='DOWNARROW_HLT')
        else:
            split.prop(lt, "disp_3drotor", text="3D Rotor", icon='RIGHTARROW')
            
        if lt.disp_3drotor:
            box = col.column(align=True).box().column()
            col_top = box.column(align=True)
            row = col_top.row(align=True)
            row.operator("mesh.rotor_operator", text = 'Store key').type_op = 1
            row = col_top.row(align=True)
            split = col_top.split(percentage=0.5)
            left_op = split.operator("mesh.rotor_operator", text="", icon='TRIA_LEFT')
            left_op.type_op = 0
            left_op.sign_op = -1
            right_op = split.operator("mesh.rotor_operator", text="", icon='TRIA_RIGHT')
            right_op.type_op = 0
            right_op.sign_op = 1
            row = col_top.row(align=True)
            if context.mode == 'EDIT_MESH':
                row.prop(lt,"rotor3d_copy", text="Copy")
            else:
                row.prop(lt, "rotor3d_instance", text='Instance')
                row = col_top.row(align=True)
                row.prop(lt,"rotor3d_copy", text="Copy")
            
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
            getlenght_op = split.operator("mesh.offset_operator", text="Restore").type_op = 1
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
            row.operator("mesh.align_operator", text = 'mirrorside').type_op = 7
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
                split.operator("mesh.polycross", text = 'Section').type_op = 0 # section and clear filter
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
                #row.active = lt.filter_edges or lt.filter_verts_bottom or lt.filter_verts_top
                row.operator("mesh.polycross", text = 'to SELECT').type_op = 2 # only filter
                row = col_top.row(align=True)
                row.prop(lt,"filter_edges", text="Filter Edges")
                row = col_top.row(align=True)
                row.prop(lt,"filter_verts_top", text="Filter Top")
                row = col_top.row(align=True)
                row.prop(lt,"filter_verts_bottom", text="Filter Bottom")
        
        split = col.split()
        if lt.disp_matExtrude:
            split.prop(lt, "disp_matExtrude", text="AutoExtrude", icon='DOWNARROW_HLT')
        else:
            split.prop(lt, "disp_matExtrude", text="AutoExtrude", icon='RIGHTARROW')
        
        if lt.disp_matExtrude:
            box = col.column(align=True).box().column()
            col_top = box.column(align=True)
            row = col_top.row(align=True)
            row.operator("mesh.get_mat4extrude", text='Get Mats')
            row = col_top.row(align=True) 
            row.operator("mesh.mat_extrude", text='Template Extrude')
        
        split = col.split()
        if lt.disp_projectloop:
            split.prop(lt, "disp_projectloop", text="3DLoop", icon='DOWNARROW_HLT')
        else:
            split.prop(lt, "disp_projectloop", text="3DLoop", icon='RIGHTARROW')
            
        if lt.disp_projectloop:
            box = col.column(align=True).box().column()
            col_top = box.column(align=True)
            row = col_top.row(align=True)
            row.operator("mesh.projectloop", text='Project 3DLoop')
        
        split = col.split(percentage=0.15)
        if lt.disp_barc:
            split.prop(lt, "disp_barc", text="", icon='DOWNARROW_HLT')
        else:
            split.prop(lt, "disp_barc", text="", icon='RIGHTARROW')
        
        split.operator("mesh.barc", text='Create B-Arc').type_op=0
        if lt.disp_barc:
            box = col.column(align=True).box().column()
            col_top = box.column(align=True)
            row = col_top.row(align=True)
            row.prop(lt,"barc_rad", text="Radius")
            col_top = box.column(align=True)
            row = col_top.row(align=True)
            row.operator("mesh.barc", text="Set radius").type_op=1
        
        split = col.split()
        if lt.disp_fedge:
            split.prop(lt, "disp_fedge", text="Fedge", icon='DOWNARROW_HLT')
        else:
            split.prop(lt, "disp_fedge", text="Fedge", icon='RIGHTARROW')
        
        if lt.disp_fedge:
            box = col.column(align=True).box().column()
            col_top = box.column(align=True)
            row = col_top.row(align=True)
            row.operator("object.fedge", text='fedge')
            row = col_top.row(align=True)
            row.prop(lt, 'fedge_verts', text = 'verts')
            row = col_top.row(align=True)
            row.prop(lt, 'fedge_edges', text = 'edges')
            row = col_top.row(align=True)
            row.prop(lt, 'fedge_three', text = 'nonquad')
            row = col_top.row(align=True)
            split = row.split(0.4, True)
            split.prop(lt, 'fedge_zerop', text = 'area')
            split.prop(lt, 'fedge_WRONG_AREA', text = '')
            
        split = col.split()
        if lt.disp_misc:
            split.prop(lt, "disp_misc", text="Misc", icon='DOWNARROW_HLT')
        else:
            split.prop(lt, "disp_misc", text="Misc", icon='RIGHTARROW')
        
        if lt.disp_misc:
            box = col.column(align=True).box().column()
            col_top = box.column(align=True)
            row = col_top.row(align=True)
            row.operator("object.misc", text='MatchProp').type_op=13
            row = col_top.row(align=True)
            row.operator("object.misc", text='Mats all to active').type_op=8
            row = col_top.row(align=True)
            row.operator("object.misc", text='Mats supress RGB').type_op=4
            row = col_top.row(align=True)
            row.operator("object.misc", text='Mats Unclone').type_op=11
            row = col_top.row(align=True)
            row.operator("object.misc", text='Mats Purgeout').type_op=12
            row = col_top.row(align=True)
            row.operator("object.misc", text='Matname HVS set').type_op=9
            row = col_top.row(align=True)
            row.operator("object.misc", text='Matname HVS del').type_op=10
            row = col_top.row(align=True)
            row.operator("object.misc", text='Matnodes switch').type_op=7
            row = col_top.row(align=True)
            row.operator("object.misc", text='Obj select modified').type_op=6
            row = col_top.row(align=True)
            row.operator("object.misc", text='Obj ignore instances').type_op=0
            row = col_top.row(align=True)
            row.operator("object.misc", text='Obj filter dupes').type_op=2
            row = col_top.row(align=True)
            row.operator("object.misc", text='Curve select 2D').type_op=1
            row = col_top.row(align=True)
            row.operator("object.misc", text='Curve swap 2D/3D').type_op=3
            row = col_top.row(align=True)
            row.operator("mesh.modal_cheredator", text='Ed reduce x2')
        
        if context.mode == 'EDIT_MESH':
            split = col.split()
            if lt.disp_eap:
                split.prop(lt, "disp_eap", text="Extrude Along Path", icon='DOWNARROW_HLT')
            else:
                split.prop(lt, "disp_eap", text="Extrude Along Path", icon='RIGHTARROW')
            
            if lt.disp_eap:
                box = col.column(align=True).box().column()
                col_top = box.column(align=True)
                row = col_top.split(0.60, align = True)
                row.label('Path:')
                row.operator('eap.op0_id', text = 'Store')
                row = col_top.split(0.60, align = True)
                row.label('Start point:')
                row.operator('eap.op1_id', text = 'Store')
                col_top = box.column(align=True)
                row = col_top.row(align=True)
                row.operator('eap.op2_id', text = 'Extrude')
        
        split = col.split()
        split.operator("scene.render_me", text='Render')
        
        split = col.split()
        split.operator("script.paul_update_addon", text = 'Auto update')
            

class D1_fedge(bpy.types.Operator):
    ''' \
    Select loose parts. edges first, vertices second, non-quad polygons third. \
    Выделяет потеряные рёбра, потом вершины и грани, каждый раз вызываясь. \
    '''
    bl_idname = "object.fedge"
    bl_label = "Fffedge"

    selected_show = False
    selected_hide = False
    

    def make_edges(self, edges):
        for e in edges:
            if e.is_loose:
                return True
        return False
    
    # makes indexes set for compare with vertices 
    # in object and find difference
    def make_indeces(self, list, vertices):
        for e in list:
            for i in e.vertices:
                vertices.add(i)

    def make_areas(self, pols):
        config = bpy.context.window_manager.paul_manager
        zerop = config.fedge_zerop
        three = config.fedge_three
        WRONG_AREA = config.fedge_WRONG_AREA
        for p in pols:
            if p.area <= WRONG_AREA and zerop:
                return True
            if len(p.vertices) == 3 and three:
                return True
        return False
    
    def verts(self, obj,selected_hide,selected_show):
        # stage two verts
        config = bpy.context.window_manager.paul_manager
        if not config.fedge_verts:
            return selected_show, selected_hide
        bpy.ops.mesh.select_mode(type='VERT')
        bpy.ops.mesh.select_all(action='DESELECT')
        bpy.ops.object.editmode_toggle()
        vertices = set()
        self.make_indeces(obj.data.edges, vertices)
        self.make_indeces(obj.data.polygons, vertices)
        for i, ver in enumerate(obj.data.vertices):
            if i not in vertices and not ver.hide:
                ver.select = True
                selected_show = True
            elif i not in vertices and ver.hide:
                selected_hide = True
        bpy.ops.object.editmode_toggle()
        return selected_show, selected_hide
    
    def edges(self, obj,selected_hide,selected_show):
        # stage one edges
        config = bpy.context.window_manager.paul_manager
        if not config.fedge_edges:
            return selected_show, selected_hide
        if not selected_show:
            bpy.ops.mesh.select_mode(type='EDGE')
            bpy.ops.mesh.select_all(action='DESELECT')
            bpy.ops.object.editmode_toggle()
            for edg in obj.data.edges:
                if edg.is_loose and not edg.hide:
                    edg.select = True
                    selected_show = True
                elif edg.is_loose and edg.hide:
                    selected_hide = True
            bpy.ops.object.editmode_toggle()
        return selected_show, selected_hide

    def zero(self, obj,selected_hide,selected_show):
        #stage area 0
        config = bpy.context.window_manager.paul_manager
        WRONG_AREA = config.fedge_WRONG_AREA
        if not config.fedge_zerop:
            return selected_show, selected_hide
        if not selected_show:
            bpy.ops.mesh.select_mode(type='FACE')
            bpy.ops.mesh.select_all(action='DESELECT')
            bpy.ops.object.editmode_toggle()
            for pol in obj.data.polygons:
                if pol.area <= WRONG_AREA and not pol.hide:
                    pol.select = True
                    selected_show = True
                elif pol.area <= WRONG_AREA and pol.hide:
                    selected_hide = True
            bpy.ops.object.editmode_toggle()
        return selected_show, selected_hide

    def three(self, obj,selected_hide,selected_show):
        #stage three polygons
        config = bpy.context.window_manager.paul_manager
        if not config.fedge_three:
            return selected_show, selected_hide
        if not selected_show:
            bpy.ops.mesh.select_mode(type='FACE')
            bpy.ops.mesh.select_all(action='DESELECT')
            bpy.ops.object.editmode_toggle()
            for pol in obj.data.polygons:
                if len(pol.vertices) != 4 and not pol.hide:
                    pol.select = True
                    selected_show = True
                elif len(pol.vertices) != 4 and pol.hide:
                    selected_hide = True
            bpy.ops.object.editmode_toggle()
        return selected_show, selected_hide

    def select_loose_objt(self):
        config = bpy.context.window_manager.paul_manager
        objects = bpy.context.selected_objects
        if not objects:
            print_error2('Fedge founds no objects selected. '+ \
                'Select objects or enter edit mode.', '01 fedge')
            return
        bpy.ops.object.select_all(action='DESELECT')

        def dosel(obj, renam):
            obj.select = True
            if obj.name[:9] != '__empty__' and renam:
                obj.name = '__empty__' + obj.name

        for obj in objects:
            if obj.type != 'MESH':
                continue
            data = obj.data
            # zero-verts objs
            if config.fedge_empty:
                if not len(data.vertices):
                    dosel(obj,True)
            # loose verts objs
            if config.fedge_verts:
                vertices = set()
                self.make_indeces(data.edges, vertices)
                self.make_indeces(data.polygons, vertices)
                v = set([i for i in range(len(data.vertices))])
                if v.difference(vertices):
                    dosel(obj,False)
            # zero area pols condition in def
            if config.fedge_zerop:
                if self.make_areas(obj.data.polygons):
                    dosel(obj,False)
            # loose edges
            if config.fedge_edges:
                if self.make_edges(data.edges):
                    dosel(obj,False)
            # triangles
            if config.fedge_three:
                for p in data.polygons:
                    if len(p.vertices) == 3:
                        dosel(obj,False)
            #print(obj.name, obj.select)
                
    def select_loose_edit(self):
        obj = bpy.context.active_object
        selected_show = False
        selected_hide = False


        # stage two verts
        selected_show, selected_hide = self.verts(obj,selected_hide,selected_show)
        # stage one edges
        selected_show, selected_hide = self.edges(obj,selected_hide,selected_show)
        #stage three polygons
        selected_show, selected_hide = self.three(obj,selected_hide,selected_show)
        #stage area 0
        selected_show, selected_hide = self.zero(obj,selected_hide,selected_show)
        # object mode if mesh clean
        if selected_show:
            self.report({'INFO'}, \
                'FEDGE: Found something')
        elif selected_hide:
            self.report({'INFO'}, \
                'FEDGE: Nothing found (but hidden)')
        else:
            bpy.ops.object.editmode_toggle()
            self.report({'INFO'}, \
                'FEDGE: Your object is clean')

    def execute(self, context):
        if bpy.context.mode == 'OBJECT':
            self.select_loose_objt()
        elif bpy.context.mode == 'EDIT_MESH':
            self.select_loose_edit()
        return {'FINISHED'}


class DisableDubleSideOperator(bpy.types.Operator):
    """Disable show duble side all meshes"""
    bl_idname = "mesh.disable_duble_sided"
    bl_label = "DDDisableDoubleSided"
    bl_options = {'REGISTER', 'UNDO'} 

    def execute(self, context):
        for mesh in bpy.data.meshes:
            mesh.show_double_sided=False
        return {'FINISHED'}    

        
class MatExrudeOperator(bpy.types.Operator):
    """Extude with mats"""
    bl_idname = "mesh.mat_extrude"
    bl_label = "Mat Extrude"
    bl_options = {'REGISTER', 'UNDO'} 

    @classmethod
    def poll(cls, context):
        return context.active_object is not None

    def execute(self, context):
        main_matExtrude(context)
        return {'FINISHED'}    


class GetMatsOperator(bpy.types.Operator):
    """Get mats"""
    bl_idname = "mesh.get_mat4extrude"
    bl_label = "Get Mats for extrude"
    bl_options = {'REGISTER', 'UNDO'} 

    @classmethod
    def poll(cls, context):
        return context.active_object is not None

    def execute(self, context):
        getMats(context)
        return {'FINISHED'}    

        
class SSOperator(bpy.types.Operator):
    """Tooltip"""
    bl_idname = "mesh.simple_scale_operator"
    bl_label = "SScale operator"
    bl_options = {'REGISTER', 'UNDO'} 
    
    type_op = bpy.props.IntProperty(name = 'type_op', default = 0, options = {'HIDDEN'})
    
    @classmethod
    def poll(cls, context):
        return context.active_object is not None

    def execute(self, context):
        if self.type_op == 1:
            getorient()
        else:
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
            lt.filter_edges = False
            lt.filter_verts_top = False
            lt.filter_verts_bottom = False
        elif self.type_op == 1:
            lt.SPLIT = True
            lt.filter_edges = False
            lt.filter_verts_top = False
            lt.filter_verts_bottom = False
        else:
            if lt.filter_edges or lt.filter_verts_bottom or lt.filter_verts_top:
                if lt.filter_edges:
                    lt.filter_verts_bottom = False
                    lt.filter_verts_top = False
            else:
                select_v_on_plane()
                return {'FINISHED'}
        
        crosspols()
        return {'FINISHED'}


class Project3DLoopOperator(bpy.types.Operator):
    bl_idname = "mesh.projectloop"
    bl_label = "Project 3D Loop"
    bl_options = {'REGISTER', 'UNDO'} 
    
    @classmethod
    def poll(cls, context):
        return context.active_object is not None

    def execute(self, context):
        
        DDDLoop()
        return {'FINISHED'}


class BarcOperator(bpy.types.Operator):
    bl_idname = "mesh.barc"
    bl_label = "BARC"
    bl_options = {'REGISTER', 'UNDO'} 
    
    type_op = bpy.props.IntProperty(name = 'type_op', default = 0, options = {'HIDDEN'})
    
    @classmethod
    def poll(cls, context):
        return context.active_object is not None

    def execute(self, context):
        if self.type_op==0:
            barc(None)
        else:
            barc(bpy.context.window_manager.paul_manager.barc_rad)
        bpy.ops.object.mode_set(mode='EDIT') 
        return {'FINISHED'}
    

class MiscOperator(bpy.types.Operator):
    bl_idname = "object.misc"
    bl_label = "Misc"
    bl_options = {'REGISTER', 'UNDO'} 
    
    type_op = bpy.props.IntProperty(name = 'type_op', default = 0, options = {'HIDDEN'})
    
    @classmethod
    def poll(cls, context):
        return context.active_object is not None

    def execute(self, context):
        if self.type_op==0:
            ignore_instance()
        elif self.type_op==1:
            select_2d_curves()
        elif self.type_op==2:
            filter_dubles_origins()
        elif self.type_op==3:
            swap_curve()
        elif self.type_op==4:
            supress_materials()
        elif self.type_op==5:
            context.scene['cheredator']=cheredator()
        elif self.type_op==6:
            select_modifiers_objs()
        elif self.type_op==7:
            switch_matnodes()
        elif self.type_op==8:
            all_mats_to_active()
        elif self.type_op==9:
            hue_2matneme()
        elif self.type_op==10:
            HVS_from_mathame()
        elif self.type_op==11:
            matsUnclone()
        elif self.type_op==12:
            matsPurgeout()
        elif self.type_op==13:
            matchProp()
        return {'FINISHED'}


class SpreadOperator(bpy.types.Operator):
    """Tooltip"""
    bl_idname = "mesh.spread_operator"
    bl_label = "Spread operator"
    bl_options = {'REGISTER', 'UNDO'} 
    
    def updateself(self, context):
        bpy.context.window_manager.paul_manager.shape_inf = self.influence * 5
    
    influence = bpy.props.IntProperty(name = "Shape",
        description = "instance -> spline -> spline 2 -> Basier_mid -> Basier -> instance",
        default = 0,
        min = 0,
        max = 50,
        update = updateself)
    
    @classmethod
    def poll(cls, context):
        return context.active_object is not None
    
    def draw(self, context):
        layout = self.layout
        row = layout.row()
        row.prop(self, 'influence')
    
    def execute(self, context):
        config = bpy.context.window_manager.paul_manager
        if main_spread(context, (config.spread_x, config.spread_y, config.spread_z, config.relation), self.influence*5):
            pass
            #print('spread complete')
        return {'FINISHED'}


class AlignOperator(bpy.types.Operator):
    bl_idname = "mesh.align_operator"
    bl_label = "Align operator"
    bl_options = {'REGISTER', 'UNDO'} 
    
    type_op = bpy.props.IntProperty(name = 'type_op', default = 0, options = {'HIDDEN'})
    dist = bpy.props.FloatProperty(name='dist', precision=4)
    dist_x = bpy.props.FloatProperty(name='X', precision=4)
    dist_y = bpy.props.FloatProperty(name='Y', precision=4)
    dist_z = bpy.props.FloatProperty(name='Z', precision=4)
    
    @classmethod
    def poll(cls, context):
        return context.active_object is not None

    def execute(self, context):
        config = bpy.context.window_manager.paul_manager
        if self.type_op==1:
            store_align()
            config.step_len = GetStoreVecLength()
            self.dist_x = Vector(config.vec_store).x
            self.dist_y = Vector(config.vec_store).y
            self.dist_z = Vector(config.vec_store).z
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
        elif self.type_op==6:
            # 3d Match Flip
            match3D(True)
        elif self.type_op==7:
            mirrorside()
        
        self.dist = config.step_len
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
            ao=bpy.context.active_object.name
            l_obj=[]
            for obj_a in bpy.context.selected_objects:
                l_obj.append(obj_a.name)
            if config.shift_copy:
                if bpy.context.mode=='OBJECT':
                    for obj_a in bpy.context.selected_objects:
                        bpy.context.scene.objects.active = obj_a
                        bpy.ops.object.select_all(action='DESELECT')
                        bpy.ops.object.select_pattern(pattern=obj_a.name)
                        bpy.ops.object.duplicate(linked=config.instance)
                        
                    for obj_a_name in l_obj:
                        bpy.context.scene.objects[obj_a_name].select=True
                    bpy.context.scene.objects.active = bpy.data.objects[ao]
                    
                elif bpy.context.mode=='EDIT_MESH':                    
                    bpy.ops.mesh.duplicate()
                
            x = config.step_len * self.sign_op
            if bpy.context.mode=='OBJECT':
                for obj_a_ in l_obj:
                    obj_a = bpy.context.scene.objects[obj_a_]
                    bpy.context.scene.objects.active = obj_a
                    bpy.ops.object.select_all(action='DESELECT')
                    bpy.ops.object.select_pattern(pattern=obj_a.name)
                    main_offset(x)
                    
                for obj_a_name in l_obj:
                    bpy.context.scene.objects[obj_a_name].select=True
                bpy.context.scene.objects.active = bpy.data.objects[ao]
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


class RotorOperator(bpy.types.Operator):
    bl_idname = "mesh.rotor_operator"
    bl_label = "Rotor operator"
    bl_options = {'REGISTER', 'UNDO'} 
    
    type_op = bpy.props.IntProperty(name = 'type_op', default = 0, options = {'HIDDEN'})
    sign_op = bpy.props.IntProperty(name = 'sign_op', default = 1, options = {'HIDDEN'})
    
    
    @classmethod
    def poll(cls, context):
        return context.active_object is not None

    def execute(self, context):
        config = bpy.context.window_manager.paul_manager
        if self.type_op==0:     # move left / right
            ao=bpy.context.active_object.name
            l_obj=[]
            for obj_a in bpy.context.selected_objects:
                l_obj.append(obj_a.name)
            if config.rotor3d_copy:
                if bpy.context.mode=='OBJECT':
                    for obj_a in bpy.context.selected_objects:
                        bpy.context.scene.objects.active = obj_a
                        bpy.ops.object.select_all(action='DESELECT')
                        bpy.ops.object.select_pattern(pattern=obj_a.name)
                        bpy.ops.object.duplicate(linked=config.rotor3d_instance)
                        
                    for obj_a_name in l_obj:
                        bpy.context.scene.objects[obj_a_name].select=True
                    bpy.context.scene.objects.active = bpy.data.objects[ao]
                    
                elif bpy.context.mode=='EDIT_MESH':                    
                    bpy.ops.mesh.duplicate()
                
            x = config.step_angle * self.sign_op
            if bpy.context.mode=='OBJECT':
                for obj_a_ in l_obj:
                    obj_a = bpy.context.scene.objects[obj_a_]
                    bpy.context.scene.objects.active = obj_a
                    bpy.ops.object.select_all(action='DESELECT')
                    bpy.ops.object.select_pattern(pattern=obj_a.name)
                    main_rotor(x)
                    
                for obj_a_name in l_obj:
                    bpy.context.scene.objects[obj_a_name].select=True
                bpy.context.scene.objects.active = bpy.data.objects[ao]
            else:
                main_rotor(x)
        
        elif self.type_op==1:   # get angle
            GetStoreVecAngle()
        
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
    relation = bpy.props.BoolProperty(name = 'relation', default = True)
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
    step_angle = bpy.props.FloatProperty(name="step_angle")
    
    shift_lockX = bpy.props.BoolProperty(name = 'shift_lockX', default = False)
    shift_lockY = bpy.props.BoolProperty(name = 'shift_lockY', default = False)
    shift_lockZ = bpy.props.BoolProperty(name = 'shift_lockZ', default = False)
    shift_copy = bpy.props.BoolProperty(name = 'shift_copy', default = False)
    shift_local = bpy.props.BoolProperty(name = 'shift_local', default = False)
    
    rotor3d_copy = bpy.props.BoolProperty(name = 'rotor3d_copy', default = False)
    rotor3d_instance = bpy.props.BoolProperty(name="rotor3d_instance")
    rotor3d_center = bpy.props.FloatVectorProperty(name="rotor3d_center")
    rotor3d_axis = bpy.props.FloatVectorProperty(name="rotor3d_axis")
    
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
    
    shape_inf = bpy.props.IntProperty(name="shape_inf", min=0, max=200, default = 0)
    shape_spline = bpy.props.BoolProperty(name="shape_spline", default = False)
    spline_Bspline2 = bpy.props.BoolProperty(name="spline_Bspline2", default = True)
    barc_rad = bpy.props.FloatProperty(name="barc_rad")
    
    disp_matExtrude = bpy.props.BoolProperty(name = 'disp_matExtrude', default = False)
    disp_projectloop = bpy.props.BoolProperty(name = 'disp_projectloop', default = False)
    disp_barc = bpy.props.BoolProperty(name = 'disp_barc', default = False)
    disp_misc = bpy.props.BoolProperty(name = 'disp_misc', default = False)
    disp_eap = bpy.props.BoolProperty(name = 'disp_eap', default = False)
    disp_fedge = bpy.props.BoolProperty(name = 'disp_fedge', default = False)
    disp_coll = bpy.props.BoolProperty(name = 'disp_coll', default = False)
    disp_3drotor = bpy.props.BoolProperty(name = 'disp_3drotor', default = False)
    
    fedge_verts = BoolProperty(name='verts', default=True)
    fedge_edges = BoolProperty(name='edges', default=True)
    fedge_zerop = BoolProperty(name='zerop', default=True)
    fedge_empty = BoolProperty(name='empty', default=True)
    fedge_three = BoolProperty(name='three', default=True)
    fedge_WRONG_AREA = bpy.props.FloatProperty(name="WRONG_AREA", default=0.02, precision=4)
    

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
        
def print_error2(message,code_error='None'):
    print('Error:'+code_error)
    bpy.ops.error.message('INVOKE_DEFAULT', 
        type = "Message",
        message = message)   

class CheredatorModalOperator(bpy.types.Operator):
    bl_idname = "mesh.modal_cheredator"
    bl_label = "Cheredator"
    bl_options = {'REGISTER', 'UNDO'}  
    steps = bpy.props.IntProperty(options = {'HIDDEN'}) 
        
    def execute(self, context):
        context.scene['cheredator'] = cheredator_fantom(self)
        return
    
    def modal(self, context, event):
        #context.area.tag_redraw()
        if event.type == 'WHEELDOWNMOUSE':  
            self.steps += 1
            self.execute(context)
            if not context.scene['cheredator']:
                self.steps -= 1
                
        elif event.type == 'WHEELUPMOUSE':  
            self.steps = max(self.steps-1,0)
            self.execute(context)
        elif event.type == 'LEFTMOUSE':  # Confirm
            bpy.types.SpaceView3D.draw_handler_remove(self._handle, 'WINDOW')
            cheredator(self.steps)
            return {'FINISHED'}
        elif event.type in ('RIGHTMOUSE', 'ESC') or not context.scene['cheredator']:  # Cancel
            bpy.types.SpaceView3D.draw_handler_remove(self._handle, 'WINDOW')
            return {'CANCELLED'}
 
        return {'RUNNING_MODAL'}
 
    def invoke(self, context, event):
        self.steps = 2
        
        ars = (self,)
        self._handle = bpy.types.SpaceView3D.draw_handler_add(cheredator_fantom, ars, 'WINDOW', 'POST_VIEW')
        self.execute(context)
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}


# ***********  Autoupdate  ***************
class ThisScriptUpdateAddon(bpy.types.Operator):
    """ Update this addon without any browsing and so on. After - press F8 to reload addons """
    bl_idname = "script.paul_update_addon"
    bl_label = "Update 1D_script addon"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        script_paths = os.path.normpath(os.path.dirname(__file__))
        
        os.curdir = os.path.dirname(os.path.join(script_paths,'addons'))
        os.chdir(os.curdir)

        try:
            url = 'http://dl.dropboxusercontent.com/u/59609328/Blender-Rus/1D_Scripts.py'
            file = urllib.request.urlretrieve(url,os.path.normpath(os.path.join(os.curdir,'1D_Scripts.py')))
            self.report({'INFO'}, "Reload addons with F8 button")
        except:
            self.report({'ERROR'}, "Cannot retrieve file from Internet")
        return {'FINISHED'}   









classes = [eap_op0, eap_op1, eap_op2, \
    RenderMe, ExportSomeData, RotorOperator, DisableDubleSideOperator, \
    MatExrudeOperator, GetMatsOperator, CrossPolsOperator, SSOperator, SpreadOperator, \
    AlignOperator, Project3DLoopOperator, BarcOperator, LayoutSSPanel, MessageOperator, \
    OffsetOperator, MiscOperator, paul_managerProps, ThisScriptUpdateAddon, \
    CheredatorModalOperator, D1_fedge]


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
    bpy.context.window_manager.paul_manager.shape_inf = 0
    
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