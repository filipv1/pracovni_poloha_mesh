#!/usr/bin/env python3
"""
Simple vertex test - manually check specific vertices
"""

import bpy
import bmesh
from pathlib import Path

print("\n" + "=" * 60)
print("MANUAL VERTEX CHECKER")
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
        
        # Switch to Edit Mode
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='DESELECT')
        bpy.context.tool_settings.mesh_select_mode = (True, False, False)
        
        # Get BMesh
        bm = bmesh.from_edit_mesh(mesh_obj.data)
        bm.verts.ensure_lookup_table()
        
        # TEST SPECIFIC VERTICES
        test_vertices = [
            3121,  # Potential lower back
            3576,  # Another lower back candidate
            4234,  # Mid back
            5432,  # Upper back  
            6789,  # Neck area
            7234,  # Higher neck
            8567,  # Base of head
            9002,  # Known head vertex
        ]
        
        print(f"\nTesting {len(test_vertices)} vertices:")
        print("-" * 60)
        
        good_spine_vertices = []
        
        for vid in test_vertices:
            if vid < len(bm.verts):
                v = bm.verts[vid]
                x, y, z = v.co.x, v.co.y, v.co.z
                
                # Check if on spine (center back)
                is_spine = abs(x) < 0.05 and z < -0.02
                
                print(f"\nVertex {vid}:")
                print(f"  Position: X={x:.3f}, Y={y:.3f}, Z={z:.3f}")
                print(f"  Height (Y): {y:.3f}")
                print(f"  Back depth (Z): {z:.3f}")
                
                if y < -0.3:
                    area = "LOWER BACK/PELVIS"
                elif y < 0:
                    area = "MID BACK"
                elif y < 0.3:
                    area = "UPPER BACK"
                else:
                    area = "NECK/HEAD"
                
                print(f"  Area: {area}")
                
                if is_spine:
                    print("  >>> ON SPINE! Good candidate!")
                    good_spine_vertices.append((vid, y, area))
                else:
                    print("  Not on spine center")
        
        # Highlight good candidates
        if good_spine_vertices:
            print("\n" + "=" * 60)
            print("RECOMMENDED SPINE VERTICES:")
            print("=" * 60)
            
            # Clear selection
            for v in bm.verts:
                v.select = False
            
            # Sort by height
            good_spine_vertices.sort(key=lambda x: x[1])
            
            for vid, height, area in good_spine_vertices:
                print(f"Vertex {vid}: {area} (Y={height:.3f})")
                bm.verts[vid].select = True
            
            bmesh.update_edit_mesh(mesh_obj.data)
            
            if len(good_spine_vertices) >= 2:
                lower = good_spine_vertices[0]
                upper = good_spine_vertices[-1]
                print("\n" + "=" * 60)
                print("SUGGESTED FOR TRUNK ANGLE:")
                print(f"LUMBAR_VERTEX_ID = {lower[0]}  # {lower[2]}")
                print(f"CERVICAL_VERTEX_ID = {upper[0]}  # {upper[2]}")
                print("=" * 60)
        
        print("\n" + "=" * 60)
        print("NEXT STEPS:")
        print("1. Selected vertices are shown in ORANGE")
        print("2. Use Numpad 3 for side view")
        print("3. If you want to test other vertices:")
        print("\n# In Blender Python Console, type:")
        print("import bmesh")
        print("obj = bpy.context.active_object")
        print("bm = bmesh.from_edit_mesh(obj.data)")
        print("bm.verts[3576].select = True  # Change number")
        print("bmesh.update_edit_mesh(obj.data)")
        print("=" * 60)
        
else:
    print(f"ERROR: File not found: {mesh_file}")