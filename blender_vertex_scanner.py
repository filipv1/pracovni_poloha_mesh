#!/usr/bin/env python3
"""
Blender Vertex Scanner - Automatically finds key vertices on spine
"""

import bpy
import bmesh
import numpy as np
from pathlib import Path

print("\n" + "=" * 60)
print("AUTOMATIC SPINE VERTEX FINDER")
print("=" * 60)

# Clear scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# Load mesh
base_dir = Path(r"C:\Users\vaclavik\ruce4\pracovni_poloha_mesh\blender_export_all")
mesh_file = base_dir / "frame_0000.obj"

if mesh_file.exists():
    # Import mesh
    try:
        bpy.ops.wm.obj_import(filepath=str(mesh_file))
    except:
        bpy.ops.import_scene.obj(filepath=str(mesh_file))
    
    if bpy.context.selected_objects:
        mesh_obj = bpy.context.selected_objects[0]
        bpy.context.view_layer.objects.active = mesh_obj
        
        # Get vertex positions
        vertices = mesh_obj.data.vertices
        
        print(f"Total vertices: {len(vertices)}")
        
        # Find vertices along the spine (back center line)
        # Spine vertices should have X close to 0 (center) and Z negative (back)
        
        spine_candidates = []
        
        for v in vertices:
            x, y, z = v.co
            # Check if vertex is near center (x ≈ 0) and on the back (z < 0)
            if abs(x) < 0.05 and z < -0.05:  # Adjust thresholds as needed
                spine_candidates.append({
                    'id': v.index,
                    'x': x,
                    'y': y,
                    'z': z,
                    'height': y  # Y is vertical in SMPL-X
                })
        
        # Sort by height (Y coordinate)
        spine_candidates.sort(key=lambda v: v['y'])
        
        print(f"\nFound {len(spine_candidates)} vertices along spine")
        
        if spine_candidates:
            # Find key positions
            total = len(spine_candidates)
            
            # Lower back (20% from bottom)
            lower_back_idx = int(total * 0.2)
            lower_back = spine_candidates[lower_back_idx]
            
            # Mid back (50%)
            mid_back_idx = int(total * 0.5)
            mid_back = spine_candidates[mid_back_idx]
            
            # Upper back (70%)
            upper_back_idx = int(total * 0.7)
            upper_back = spine_candidates[upper_back_idx]
            
            # Neck area (85%)
            neck_idx = int(total * 0.85)
            neck_back = spine_candidates[neck_idx]
            
            print("\n" + "=" * 60)
            print("KEY SPINE VERTICES FOUND:")
            print("=" * 60)
            print(f"\nLOWER BACK (Lumbar region):")
            print(f"  Vertex ID: {lower_back['id']}")
            print(f"  Position: ({lower_back['x']:.3f}, {lower_back['y']:.3f}, {lower_back['z']:.3f})")
            
            print(f"\nMID BACK:")
            print(f"  Vertex ID: {mid_back['id']}")
            print(f"  Position: ({mid_back['x']:.3f}, {mid_back['y']:.3f}, {mid_back['z']:.3f})")
            
            print(f"\nUPPER BACK (Between shoulders):")
            print(f"  Vertex ID: {upper_back['id']}")
            print(f"  Position: ({upper_back['x']:.3f}, {upper_back['y']:.3f}, {upper_back['z']:.3f})")
            
            print(f"\nNECK BACK (Cervical region):")
            print(f"  Vertex ID: {neck_back['id']}")
            print(f"  Position: ({neck_back['x']:.3f}, {neck_back['y']:.3f}, {neck_back['z']:.3f})")
            
            # Switch to edit mode to highlight
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.select_all(action='DESELECT')
            bpy.context.tool_settings.mesh_select_mode = (True, False, False)
            
            bm = bmesh.from_edit_mesh(mesh_obj.data)
            bm.verts.ensure_lookup_table()
            
            # Select the found vertices
            key_vertices = [lower_back['id'], mid_back['id'], upper_back['id'], neck_back['id']]
            for vid in key_vertices:
                bm.verts[vid].select = True
            
            bmesh.update_edit_mesh(mesh_obj.data)
            
            print("\n" + "=" * 60)
            print("SUGGESTED VERTICES FOR CALCULATIONS:")
            print(f"LUMBAR_VERTEX_ID = {lower_back['id']}  # Lower back on skin")
            print(f"CERVICAL_VERTEX_ID = {neck_back['id']}  # Neck back on skin")
            print("\nThese vertices are now SELECTED (orange) in the viewport")
            print("You can adjust selection manually if needed")
            print("=" * 60)
            
            # Also print some nearby alternatives
            print("\nNEARBY ALTERNATIVES (if needed):")
            for i in range(max(0, lower_back_idx-2), min(total, lower_back_idx+3)):
                v = spine_candidates[i]
                print(f"  Lower back area: ID {v['id']} at height {v['y']:.3f}")
            
        else:
            print("No spine vertices found - check mesh orientation")