#!/usr/bin/env python3
"""
Interactive Vertex Selector for Blender
Real-time vertex ID display using modal operator
"""

import bpy
import bmesh
from bpy.types import Operator
from pathlib import Path

class MESH_OT_vertex_selector(Operator):
    """Select vertices and show their IDs"""
    bl_idname = "mesh.vertex_selector"
    bl_label = "Vertex ID Selector"
    
    def modal(self, context, event):
        if event.type == 'TIMER':
            # Check for selected vertices
            obj = context.active_object
            if obj and obj.mode == 'EDIT':
                bm = bmesh.from_edit_mesh(obj.data)
                selected = [v.index for v in bm.verts if v.select]
                
                if selected and selected != self.last_selected:
                    self.last_selected = selected
                    print(f"\n>>> SELECTED VERTEX IDs: {selected}")
                    if len(selected) == 1:
                        v = bm.verts[selected[0]]
                        print(f"    Position: ({v.co.x:.3f}, {v.co.y:.3f}, {v.co.z:.3f})")
                        
                        # Suggest what this might be based on position
                        if v.co.y < -0.5:
                            print("    Area: LOWER BODY")
                        elif v.co.y < 0:
                            print("    Area: MID BODY")
                        elif v.co.y < 0.5:
                            print("    Area: UPPER BODY")
                        else:
                            print("    Area: HEAD/NECK")
                        
                        if abs(v.co.x) < 0.05 and v.co.z < -0.05:
                            print("    >>> GOOD! This is on the SPINE/BACK")
                            print(f"    >>> You can use: VERTEX_ID = {selected[0]}")
        
        elif event.type == 'ESC':
            return self.cancel(context)
        
        return {'PASS_THROUGH'}
    
    def invoke(self, context, event):
        self.last_selected = []
        self._timer = context.window_manager.event_timer_add(0.1, window=context.window)
        context.window_manager.modal_handler_add(self)
        print("\n" + "=" * 60)
        print("VERTEX SELECTOR ACTIVE!")
        print("Click on vertices to see their IDs")
        print("Press ESC to stop")
        print("=" * 60)
        return {'RUNNING_MODAL'}
    
    def cancel(self, context):
        context.window_manager.event_timer_remove(self._timer)
        print("\nVertex selector stopped")
        return {'CANCELLED'}

# Clear and setup
print("\n" + "=" * 60)
print("INTERACTIVE VERTEX ID SELECTOR")
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
        mesh_obj.name = "SMPLX_Mesh"
        bpy.context.view_layer.objects.active = mesh_obj
        
        # Switch to Edit Mode
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='DESELECT')
        bpy.context.tool_settings.mesh_select_mode = (True, False, False)
        
        # Register and run the operator
        try:
            bpy.utils.register_class(MESH_OT_vertex_selector)
        except:
            bpy.utils.unregister_class(MESH_OT_vertex_selector)
            bpy.utils.register_class(MESH_OT_vertex_selector)
        
        print("\n" + "=" * 60)
        print("INSTRUCTIONS:")
        print("1. Click on any vertex")
        print("2. Look at this console for vertex ID")
        print("3. Press ESC when done")
        print("\nRECOMMENDED VERTICES TO FIND:")
        print("- LOWER BACK (lumbar): Y around -0.5 to -0.3")
        print("- UPPER BACK (cervical): Y around 0.2 to 0.4")
        print("\nPress NUMPAD 3 for SIDE VIEW (best for spine)")
        print("=" * 60)
        print("\nStarting interactive selector...")
        
        # Start the modal operator
        bpy.ops.mesh.vertex_selector('INVOKE_DEFAULT')
        
else:
    print(f"ERROR: Mesh file not found: {mesh_file}")