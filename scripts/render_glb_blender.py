#!/usr/bin/env python3
"""
Blender script to render GLB files matching pyrender behavior.
Usage: blender --background --python render_glb_blender.py -- <glb_path> <output_dir> <resolution> <view_name> <cam_x> <cam_y> <cam_z> <light_intensity> <bg_color> <vertical_offset>
"""
import bpy
import sys
import os
import math
from pathlib import Path
from mathutils import Vector

# Get arguments after '--'
argv = sys.argv
argv = argv[argv.index("--") + 1:]  # Get all args after "--"

if len(argv) < 3:
    print("Usage: blender --background --python render_glb_blender.py -- <glb_path> <output_dir> <resolution> <view_name> <cam_x> <cam_y> <cam_z> <light_intensity> <bg_color> <vertical_offset>")
    sys.exit(1)

glb_path = argv[0]
output_dir = argv[1]
resolution = int(argv[2]) if len(argv) > 2 else 512
view_name = argv[3] if len(argv) > 3 else "view"
cam_x = float(argv[4]) if len(argv) > 4 else 0.0
cam_y = float(argv[5]) if len(argv) > 5 else 0.0
cam_z = float(argv[6]) if len(argv) > 6 else 2.0
light_intensity = float(argv[7]) if len(argv) > 7 else 2.0
bg_color = argv[8] if len(argv) > 8 else "white"
vertical_offset = float(argv[9]) if len(argv) > 9 else 0.0

print(f"Rendering {glb_path}")
print(f"Output: {output_dir}")
print(f"Resolution: {resolution}")
print(f"View: {view_name}")
print(f"Camera position: ({cam_x}, {cam_y}, {cam_z})")
print(f"Light intensity: {light_intensity}")
print(f"Background: {bg_color}")
print(f"Vertical offset (look-at): {vertical_offset}")

# Clear default scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Import GLB
bpy.ops.import_scene.gltf(filepath=glb_path)

# Get imported objects
imported_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']

print(f"Imported {len(imported_objects)} mesh objects")

if not imported_objects:
    print("ERROR: No mesh objects imported")
    sys.exit(1)

# Join all meshes into a single object (matching trimesh.Scene.dump(concatenate=True))
if len(imported_objects) > 1:
    bpy.context.view_layer.objects.active = imported_objects[0]
    for obj in imported_objects:
        obj.select_set(True)
    bpy.ops.object.join()
    mesh_obj = bpy.context.active_object
else:
    mesh_obj = imported_objects[0]

# Apply all transformations to mesh data
bpy.context.view_layer.objects.active = mesh_obj
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

# Get all vertices in world space
mesh = mesh_obj.data
verts = [mesh_obj.matrix_world @ v.co for v in mesh.vertices]

# Calculate centroid (average of all vertices - exact trimesh behavior)
centroid = Vector((
    sum(v.x for v in verts) / len(verts),
    sum(v.y for v in verts) / len(verts),
    sum(v.z for v in verts) / len(verts)
))

print(f"Centroid: ({centroid.x:.6f}, {centroid.y:.6f}, {centroid.z:.6f})")

# Center mesh by translating all vertices (matching: mesh.vertices -= mesh.centroid)
for v in mesh.vertices:
    v.co -= centroid

bpy.context.view_layer.update()
mesh.update()

# Calculate extents after centering
verts_centered = [mesh_obj.matrix_world @ v.co for v in mesh.vertices]
xs = [v.x for v in verts_centered]
ys = [v.y for v in verts_centered]
zs = [v.z for v in verts_centered]

extent_x = max(xs) - min(xs)
extent_y = max(ys) - min(ys)
extent_z = max(zs) - min(zs)
max_extent = max(extent_x, extent_y, extent_z)

# Calculate scale (matching: scale = 1.5 / np.max(mesh.extents))
scale_factor = 1.5 / max_extent if max_extent > 0 else 1.0

print(f"Extents: ({extent_x:.6f}, {extent_y:.6f}, {extent_z:.6f})")
print(f"Max extent: {max_extent:.6f}")
print(f"Scale factor: {scale_factor:.6f}")

# Scale all vertices (matching: mesh.vertices *= scale)
for v in mesh.vertices:
    v.co *= scale_factor

bpy.context.view_layer.update()
mesh.update()

# Verify final centering
verts_final = [mesh_obj.matrix_world @ v.co for v in mesh.vertices]
final_centroid = Vector((
    sum(v.x for v in verts_final) / len(verts_final),
    sum(v.y for v in verts_final) / len(verts_final),
    sum(v.z for v in verts_final) / len(verts_final)
))
print(f"Final centroid: ({final_centroid.x:.6f}, {final_centroid.y:.6f}, {final_centroid.z:.6f})")

# Verify final extents
xs_f = [v.x for v in verts_final]
ys_f = [v.y for v in verts_final]
zs_f = [v.z for v in verts_final]
print(f"Final extents: ({max(xs_f)-min(xs_f):.6f}, {max(ys_f)-min(ys_f):.6f}, {max(zs_f)-min(zs_f):.6f})")

# Setup render settings
scene = bpy.context.scene
scene.render.engine = 'CYCLES'  # Use Cycles for better texture rendering
scene.cycles.samples = 32  # Lower samples for faster render
scene.render.resolution_x = resolution
scene.render.resolution_y = resolution

# Set output format to PNG RGB
scene.render.image_settings.file_format = 'PNG'
scene.render.image_settings.color_depth = '8'
scene.render.image_settings.color_mode = 'RGB'
scene.render.film_transparent = False

# Set world background color (constant, not affected by light_intensity)
if not scene.world:
    scene.world = bpy.data.worlds.new("World")
scene.world.use_nodes = True
bg_node = scene.world.node_tree.nodes.get("Background")
if bg_node:
    if bg_color == "white":
        bg_node.inputs[0].default_value = (1, 1, 1, 1)
    elif bg_color == "black":
        bg_node.inputs[0].default_value = (0, 0, 0, 1)
    elif bg_color == "gray" or bg_color == "grey":
        # Dark gray background
        bg_node.inputs[0].default_value = (0.25, 0.25, 0.25, 1)
    else:  # default to dark gray
        bg_node.inputs[0].default_value = (0.25, 0.25, 0.25, 1)
    # Keep background strength constant at 1.0 (background color stays fixed)
    bg_node.inputs[1].default_value = 1.0

# Add area light for object illumination (controlled by light_intensity)
light_data = bpy.data.lights.new(name="AreaLight", type='AREA')
light_data.energy = light_intensity * 100  # Scale up for area light
light_data.size = 5.0  # Large soft light
area_light = bpy.data.objects.new(name="AreaLight", object_data=light_data)
scene.collection.objects.link(area_light)
area_light.location = (2, 2, 3)  # Position above and to the side
area_light.rotation_euler = (0.5, 0.3, 0)  # Angle toward object

# Setup camera matching pyrender's PerspectiveCamera(yfov=np.pi / 3.0)
camera_data = bpy.data.cameras.new(name='Camera')
camera = bpy.data.objects.new('Camera', camera_data)
scene.collection.objects.link(camera)
scene.camera = camera
# Convert yfov to lens focal length
# yfov = np.pi / 3.0 = 60 degrees (vertical field of view)
# For 36mm sensor: focal_length = sensor_height / (2 * tan(yfov/2))
yfov = math.pi / 3.0  # 60 degrees
sensor_height = 24  # Standard 36mm sensor has 24mm height for 3:2 ratio
focal_length = sensor_height / (2 * math.tan(yfov / 2))
camera_data.lens = focal_length
camera_data.sensor_height = sensor_height

# Global ambient lighting only (no directional light)
# Uniform illumination from all directions - ideal for data preprocessing

# Position camera at specified location
camera.location = (cam_x, cam_y, cam_z)

# Point camera at target with vertical offset (avoid gimbal lock)
# Camera looks down -Z axis, Y is up
# vertical_offset shifts the look-at target up (+) or down (-) to adjust framing
look_at_target = Vector((0, 0, vertical_offset))
direction = look_at_target - Vector((cam_x, cam_y, cam_z))
rot_quat = direction.to_track_quat('-Z', 'Y')
camera.rotation_mode = 'QUATERNION'
camera.rotation_quaternion = rot_quat

print(f"View: {view_name}")
print(f"Camera location: {camera.location}")
print(f"Look-at target: {look_at_target}")
print(f"Camera rotation (euler degrees): ({math.degrees(camera.rotation_euler.x):.1f}, {math.degrees(camera.rotation_euler.y):.1f}, {math.degrees(camera.rotation_euler.z):.1f})")


# Create output directory
os.makedirs(output_dir, exist_ok=True)

uid = Path(glb_path).stem

# Render
output_path = os.path.join(output_dir, f"{uid}_{view_name}.png")
scene.render.filepath = output_path
bpy.ops.render.render(write_still=True)

print(f"✓ Saved to {output_path}")
print("✓ Rendering complete!")
