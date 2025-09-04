#!/usr/bin/env python3
"""
Blender Vertex Info Script
Shows selected vertex IDs in real-time
"""

import bpy
import bmesh

print("\n" + "=" * 60)
print("VERTEX ID FINDER")
print("=" * 60)

# Clear scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# Load first frame mesh
from pathlib import Path
base_dir = Path(r"C:\Users\vaclavik\ruce4\pracovni_poloha_mesh\blender_export_all")
mesh_file = base_dir / "frame_0000.obj"

if not mesh_file.exists():
    print(f"ERROR: Mesh file not found: {mesh_file}")
else:
    # Import mesh
    try:
        bpy.ops.wm.obj_import(filepath=str(mesh_file))
    except:
        bpy.ops.import_scene.obj(filepath=str(mesh_file))
    
    if bpy.context.selected_objects:
        mesh_obj = bpy.context.selected_objects[0]
        mesh_obj.name = "SMPLX_Mesh"
        
        # Make it active
        bpy.context.view_layer.objects.active = mesh_obj
        
        # Switch to Edit Mode
        bpy.ops.object.mode_set(mode='EDIT')
        
        # Get BMesh
        bm = bmesh.from_edit_mesh(mesh_obj.data)
        bm.verts.ensure_lookup_table()
        
        print(f"\nMesh loaded: {len(bm.verts)} vertices")
        print("\n" + "=" * 60)
        print("HOW TO USE:")
        print("1. You are in EDIT MODE")
        print("2. Press 'A' to deselect all")
        print("3. Click on any vertex (orange dot appears)")
        print("4. Run this code snippet in Blender console:")
        print("\n--- COPY THIS CODE TO BLENDER CONSOLE ---")
        print("import bmesh")
        print("obj = bpy.context.active_object")
        print("bm = bmesh.from_edit_mesh(obj.data)")
        print("selected = [v.index for v in bm.verts if v.select]")
        print("if selected:")
        print("    print(f'Selected vertex ID: {selected[0]}')")
        print("else:")
        print("    print('No vertex selected')")
        print("--- END OF CODE ---")
        print("\n5. Or use this to select specific vertex:")
        print("--- SELECT VERTEX BY ID ---")
        print("vertex_id = 5000  # Change this number")
        print("bm.verts[vertex_id].select = True")
        print("bmesh.update_edit_mesh(obj.data)")
        print("--- END OF CODE ---")
        print("\nKEY AREAS TO CHECK:")
        print("- Lower back: around vertex 3000-4000")
        print("- Upper back: around vertex 5000-6000")
        print("- Neck back: around vertex 7000-8000")
        print("- Head top: around vertex 9000-9500")
        print("\nVIEW CONTROLS:")
        print("- Numpad 3: Side view (BEST FOR SPINE)")
        print("- Numpad 7: Top view")
        print("- Numpad 1: Front view")
        print("- Mouse wheel: Zoom")
        print("- Middle mouse: Rotate")
        print("=" * 60)
        
        # Try to select a vertex on the back as example
        try:
            # Deselect all first
            for v in bm.verts:
                v.select = False
            
            # Select example vertex (middle back area)
            test_vertex = 5432  # Example vertex
            if test_vertex < len(bm.verts):
                bm.verts[test_vertex].select = True
                print(f"\nEXAMPLE: Selected vertex {test_vertex} as demonstration")
                print("This vertex should be highlighted in orange")
            
            bmesh.update_edit_mesh(mesh_obj.data)
        except:
            pass
        
        # Set to vertex select mode
        bpy.context.tool_settings.mesh_select_mode = (True, False, False)