#!/usr/bin/env python3
"""
Blender Vertex ID Picker
Load mesh and click on vertices to get their IDs
Perfect for finding vertex positions on skin surface
"""

import bpy
import bmesh
from pathlib import Path

print("\n" + "=" * 60)
print("VERTEX ID PICKER FOR BLENDER")
print("=" * 60)

# Clear scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# Load first frame mesh
base_dir = Path(r"C:\Users\vaclavik\ruce4\pracovni_poloha_mesh\blender_export_all")
mesh_file = base_dir / "frame_0000.obj"

if not mesh_file.exists():
    print(f"ERROR: Mesh file not found: {mesh_file}")
    print("Please run export_all_vectors_to_blender.py first!")
else:
    # Import mesh
    try:
        bpy.ops.wm.obj_import(filepath=str(mesh_file))
    except:
        bpy.ops.import_scene.obj(filepath=str(mesh_file))
    
    if bpy.context.selected_objects:
        mesh_obj = bpy.context.selected_objects[0]
        mesh_obj.name = "SMPLX_Mesh_For_Vertex_Picking"
        
        # Make it active
        bpy.context.view_layer.objects.active = mesh_obj
        
        # Switch to Edit Mode
        bpy.ops.object.mode_set(mode='EDIT')
        
        # Deselect all
        bpy.ops.mesh.select_all(action='DESELECT')
        
        # Switch to Vertex select mode
        bpy.context.tool_settings.mesh_select_mode = (True, False, False)
        
        print("\n" + "=" * 60)
        print("VERTEX PICKER READY!")
        print("=" * 60)
        print("\nINSTRUCTIONS:")
        print("1. You are now in EDIT MODE with VERTEX SELECT")
        print("2. Click on any vertex to select it")
        print("3. Look at the top bar - it shows: 'Vertex: X/10475'")
        print("   where X is the vertex ID (0-based)")
        print("\nUSEFUL VERTICES TO FIND:")
        print("- Lower back (lumbar area on skin)")
        print("- Upper back (between shoulder blades)")
        print("- Base of neck (on skin)")
        print("- Shoulders (on skin)")
        print("\nTIPS:")
        print("- Use mouse scroll to zoom")
        print("- Middle mouse to rotate view")
        print("- Press '3' on numpad for side view")
        print("- Press '1' on numpad for front view")
        print("- Press '7' on numpad for top view")
        print("- Alt+Click to select vertex behind others")
        print("\nWRITE DOWN THE VERTEX IDs YOU FIND!")
        print("Example: HEAD_VERTEX_ID = 9002")
        print("=" * 60)
        
        # Set viewport to solid mode for better visibility
        for area in bpy.context.screen.areas:
            if area.type == 'VIEW_3D':
                for space in area.spaces:
                    if space.type == 'VIEW_3D':
                        space.shading.type = 'SOLID'
                        # Note: show_indices removed - not available in all Blender versions
    else:
        print("ERROR: Could not import mesh!")