"""
3D GLB Mesh Segmentation using DINOv2 Feature Clustering

This script automatically segments a 3D GLB mesh into distinct parts:
1. Renders the mesh from multiple views
2. Extracts DINOv2 features per pixel (semantic embeddings)
3. Projects features to mesh faces
4. Clusters faces by feature similarity
5. Outputs separate GLB files for each cluster

No text prompts needed - automatically discovers all distinct parts.

Usage:
    python segment_glb_dinov2.py --input mesh.glb --num_clusters 3 --output_dir ./output
"""

import os
import sys

# Set headless rendering backend before importing pyrender
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import argparse
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import trimesh
import pyrender
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class DINOv2MeshSegmenter:
    def __init__(self, model_name: str = "dinov2_vits14", device: str = "cuda"):
        self.device = device
        self.patch_size = 14  # DINOv2 uses 14x14 patches

        print(f"Loading DINOv2 ({model_name})...")
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        self.model = self.model.to(device)
        self.model.eval()
        print("DINOv2 loaded!")

    def load_mesh(self, glb_path: str) -> trimesh.Trimesh:
        """Load GLB file and convert to single mesh."""
        scene = trimesh.load(glb_path)
        if isinstance(scene, trimesh.Scene):
            mesh = scene.to_geometry()
        else:
            mesh = scene
        return mesh

    def create_face_id_colors(self, num_faces: int) -> np.ndarray:
        """Create unique colors for each face to encode face IDs."""
        colors = np.zeros((num_faces, 4), dtype=np.uint8)
        for i in range(num_faces):
            colors[i, 0] = (i >> 16) & 0xFF
            colors[i, 1] = (i >> 8) & 0xFF
            colors[i, 2] = i & 0xFF
            colors[i, 3] = 255
        return colors

    def decode_face_ids(self, image: np.ndarray) -> np.ndarray:
        """Decode face IDs from rendered image."""
        r = image[:, :, 0].astype(np.int32)
        g = image[:, :, 1].astype(np.int32)
        b = image[:, :, 2].astype(np.int32)
        face_ids = (r << 16) | (g << 8) | b
        return face_ids

    def _look_at(self, eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
        """Create a look-at camera matrix."""
        forward = target - eye
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        mat = np.eye(4)
        mat[:3, 0] = right
        mat[:3, 1] = up
        mat[:3, 2] = -forward
        mat[:3, 3] = eye
        return mat

    def render_views(self, mesh: trimesh.Trimesh, num_views: int = 8,
                     resolution: Tuple[int, int] = (518, 518)) -> List[Dict]:
        """Render mesh from multiple views."""
        views = []

        # Create pyrender scene for color rendering
        scene_color = pyrender.Scene(bg_color=[255, 255, 255, 255])
        pr_mesh_color = pyrender.Mesh.from_trimesh(mesh)
        scene_color.add(pr_mesh_color)

        # Create face-colored mesh for ID rendering
        face_colors = self.create_face_id_colors(len(mesh.faces))
        mesh_id = mesh.copy()
        mesh_id.visual = trimesh.visual.ColorVisuals(mesh_id, face_colors=face_colors)
        scene_id = pyrender.Scene(bg_color=[0, 0, 0, 255])
        pr_mesh_id = pyrender.Mesh.from_trimesh(mesh_id, smooth=False)
        scene_id.add(pr_mesh_id)

        # Camera setup
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
        bounds = mesh.bounds
        center = mesh.centroid
        scale = np.max(bounds[1] - bounds[0])
        distance = scale * 2.0

        renderer = pyrender.OffscreenRenderer(resolution[0], resolution[1])

        # Generate camera positions with multiple elevations for better coverage
        camera_positions = []

        # Ring at low elevation (0.2 rad)
        for i in range(num_views):
            angle = 2 * np.pi * i / num_views
            camera_positions.append((angle, 0.2))

        # Ring at medium elevation (0.6 rad)
        for i in range(num_views // 2):
            angle = 2 * np.pi * i / (num_views // 2) + np.pi / num_views
            camera_positions.append((angle, 0.6))

        # Top view
        camera_positions.append((0, 1.4))
        # Bottom view
        camera_positions.append((0, -1.0))

        for angle, elevation in camera_positions:
            cam_x = distance * np.cos(angle) * np.cos(elevation)
            cam_y = distance * np.sin(elevation)
            cam_z = distance * np.sin(angle) * np.cos(elevation)

            camera_pose = self._look_at(
                eye=np.array([cam_x, cam_y, cam_z]) + center,
                target=center,
                up=np.array([0, 1, 0])
            )

            cam_node_color = scene_color.add(camera, pose=camera_pose)
            cam_node_id = scene_id.add(camera, pose=camera_pose)

            light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
            light_node = scene_color.add(light, pose=camera_pose)

            color_img, _ = renderer.render(scene_color)
            scene_id_flags = pyrender.RenderFlags.FLAT | pyrender.RenderFlags.SKIP_CULL_FACES
            id_img, _ = renderer.render(scene_id, flags=scene_id_flags)

            views.append({
                'color': color_img,
                'face_ids': self.decode_face_ids(id_img),
                'angle': np.degrees(angle)
            })

            scene_color.remove_node(cam_node_color)
            scene_color.remove_node(light_node)
            scene_id.remove_node(cam_node_id)

        renderer.delete()
        return views

    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """Extract DINOv2 features from an image."""
        # Preprocess image
        img = Image.fromarray(image)

        # Resize to be divisible by patch size
        h, w = image.shape[:2]
        new_h = (h // self.patch_size) * self.patch_size
        new_w = (w // self.patch_size) * self.patch_size
        img = img.resize((new_w, new_h), Image.BILINEAR)

        # Convert to tensor and normalize
        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        # Normalize with ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        img_tensor = (img_tensor - mean) / std

        # Extract features
        with torch.no_grad():
            features = self.model.forward_features(img_tensor)
            patch_tokens = features['x_norm_patchtokens']  # [1, num_patches, dim]

        # Reshape to spatial grid
        num_patches_h = new_h // self.patch_size
        num_patches_w = new_w // self.patch_size
        feature_map = patch_tokens.reshape(1, num_patches_h, num_patches_w, -1)
        feature_map = feature_map.squeeze(0).cpu().numpy()  # [H, W, dim]

        return feature_map, (new_h, new_w)

    def project_features_to_faces(self, views: List[Dict], num_faces: int,
                                   feature_dim: int) -> np.ndarray:
        """Project DINOv2 features to mesh faces."""
        face_features = np.zeros((num_faces, feature_dim), dtype=np.float32)
        face_counts = np.zeros(num_faces, dtype=np.float32)

        for view in views:
            color_img = view['color']
            face_ids = view['face_ids']

            # Extract features
            feature_map, (new_h, new_w) = self.extract_features(color_img)
            feat_h, feat_w, feat_dim = feature_map.shape

            # Resize face_ids to match feature resolution
            face_ids_resized = np.array(
                Image.fromarray(face_ids.astype(np.int32)).resize(
                    (feat_w, feat_h), Image.NEAREST
                )
            )

            # Accumulate features for each face
            for y in range(feat_h):
                for x in range(feat_w):
                    fid = face_ids_resized[y, x]
                    if 0 < fid < num_faces:
                        face_features[fid] += feature_map[y, x]
                        face_counts[fid] += 1

        # Average features
        valid_mask = face_counts > 0
        face_features[valid_mask] /= face_counts[valid_mask, np.newaxis]

        return face_features, valid_mask

    def propagate_features(self, mesh: trimesh.Trimesh, features: np.ndarray,
                           valid_mask: np.ndarray, iterations: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Propagate features from visible faces to neighboring invisible faces."""
        print(f"Propagating features to neighboring faces...")

        # Build face adjacency from mesh
        # Two faces are adjacent if they share an edge (2 vertices)
        from collections import defaultdict

        edge_to_faces = defaultdict(list)
        for face_idx, face in enumerate(mesh.faces):
            edges = [(min(face[0], face[1]), max(face[0], face[1])),
                     (min(face[1], face[2]), max(face[1], face[2])),
                     (min(face[2], face[0]), max(face[2], face[0]))]
            for edge in edges:
                edge_to_faces[edge].append(face_idx)

        # Build adjacency list
        face_neighbors = defaultdict(set)
        for faces in edge_to_faces.values():
            for i in range(len(faces)):
                for j in range(i + 1, len(faces)):
                    face_neighbors[faces[i]].add(faces[j])
                    face_neighbors[faces[j]].add(faces[i])

        new_features = features.copy()
        new_valid_mask = valid_mask.copy()

        for iteration in range(iterations):
            updated = 0
            for face_idx in range(len(mesh.faces)):
                if new_valid_mask[face_idx]:
                    continue  # Already has features

                # Get features from valid neighbors
                neighbor_features = []
                for neighbor in face_neighbors[face_idx]:
                    if new_valid_mask[neighbor]:
                        neighbor_features.append(new_features[neighbor])

                if neighbor_features:
                    new_features[face_idx] = np.mean(neighbor_features, axis=0)
                    new_valid_mask[face_idx] = True
                    updated += 1

            if updated == 0:
                break
            print(f"  Iteration {iteration + 1}: propagated to {updated} faces")

        return new_features, new_valid_mask

    def cluster_faces(self, features: np.ndarray, valid_mask: np.ndarray,
                      num_clusters: int, method: str = "kmeans") -> np.ndarray:
        """Cluster faces based on their features."""
        # Get valid features
        valid_features = features[valid_mask]

        # Normalize features
        scaler = StandardScaler()
        valid_features_norm = scaler.fit_transform(valid_features)

        print(f"Clustering {len(valid_features)} faces into {num_clusters} clusters...")

        if method == "kmeans":
            clusterer = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        elif method == "spectral":
            clusterer = SpectralClustering(n_clusters=num_clusters, random_state=42,
                                           affinity='nearest_neighbors', n_neighbors=10)
        else:
            raise ValueError(f"Unknown clustering method: {method}")

        valid_labels = clusterer.fit_predict(valid_features_norm)

        # Map back to all faces
        labels = np.full(len(features), -1, dtype=np.int32)
        labels[valid_mask] = valid_labels

        return labels

    def find_largest_connected_component(self, mesh: trimesh.Trimesh,
                                          face_indices: np.ndarray) -> np.ndarray:
        """Find the largest connected component among selected faces."""
        from collections import defaultdict, deque

        if len(face_indices) == 0:
            return face_indices

        # Build adjacency among selected faces only
        face_set = set(face_indices)

        # Map edges to faces
        edge_to_faces = defaultdict(list)
        for face_idx in face_indices:
            face = mesh.faces[face_idx]
            edges = [(min(face[0], face[1]), max(face[0], face[1])),
                     (min(face[1], face[2]), max(face[1], face[2])),
                     (min(face[2], face[0]), max(face[2], face[0]))]
            for edge in edges:
                edge_to_faces[edge].append(face_idx)

        # Build adjacency list
        face_neighbors = defaultdict(set)
        for faces in edge_to_faces.values():
            for i in range(len(faces)):
                for j in range(i + 1, len(faces)):
                    face_neighbors[faces[i]].add(faces[j])
                    face_neighbors[faces[j]].add(faces[i])

        # Find connected components using BFS
        visited = set()
        components = []

        for start_face in face_indices:
            if start_face in visited:
                continue

            # BFS to find connected component
            component = []
            queue = deque([start_face])
            visited.add(start_face)

            while queue:
                face = queue.popleft()
                component.append(face)

                for neighbor in face_neighbors[face]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

            components.append(component)

        # Return largest component
        largest = max(components, key=len)
        print(f"    Found {len(components)} components, largest has {len(largest)} faces ({100*len(largest)/len(face_indices):.1f}% of cluster)")

        return np.array(largest)

    def expand_face_selection(self, mesh: trimesh.Trimesh, face_indices: np.ndarray,
                               offset: int = 1) -> np.ndarray:
        """Expand face selection by including neighboring faces (dilation)."""
        from collections import defaultdict

        if offset <= 0:
            return face_indices

        # Build face adjacency
        edge_to_faces = defaultdict(list)
        for face_idx, face in enumerate(mesh.faces):
            edges = [(min(face[0], face[1]), max(face[0], face[1])),
                     (min(face[1], face[2]), max(face[1], face[2])),
                     (min(face[2], face[0]), max(face[2], face[0]))]
            for edge in edges:
                edge_to_faces[edge].append(face_idx)

        face_neighbors = defaultdict(set)
        for faces in edge_to_faces.values():
            for i in range(len(faces)):
                for j in range(i + 1, len(faces)):
                    face_neighbors[faces[i]].add(faces[j])
                    face_neighbors[faces[j]].add(faces[i])

        # Expand selection
        selected = set(face_indices)
        for _ in range(offset):
            new_selected = set()
            for face_idx in selected:
                new_selected.update(face_neighbors[face_idx])
            selected.update(new_selected)

        expanded = np.array(list(selected))
        print(f"    Expanded selection: {len(face_indices)} -> {len(expanded)} faces (offset={offset})")
        return expanded

    def extract_from_original(self, original_mesh: trimesh.Trimesh,
                               face_indices: np.ndarray,
                               center: bool = True) -> Optional[trimesh.Trimesh]:
        """Extract faces from original mesh preserving textures/materials."""
        if len(face_indices) == 0:
            return None

        # Use trimesh's submesh method which properly preserves textures/UVs
        new_mesh = original_mesh.submesh([face_indices], append=True)

        # Center the mesh
        if center:
            new_mesh.vertices -= new_mesh.centroid

        return new_mesh

    def extract_mesh_by_label(self, mesh: trimesh.Trimesh, labels: np.ndarray,
                               target_label: int, name: str = "part") -> Optional[trimesh.Trimesh]:
        """Extract submesh for faces with specific label."""
        face_mask = labels == target_label
        selected_face_indices = np.where(face_mask)[0]

        if len(selected_face_indices) == 0:
            return None

        selected_faces = mesh.faces[selected_face_indices]
        unique_verts = np.unique(selected_faces.flatten())
        vert_mapping = {old: new for new, old in enumerate(unique_verts)}
        new_faces = np.array([[vert_mapping[v] for v in face] for face in selected_faces])
        new_vertices = mesh.vertices[unique_verts]

        new_mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces)

        if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
            new_mesh.visual.vertex_colors = mesh.visual.vertex_colors[unique_verts]

        new_mesh.vertices -= new_mesh.centroid

        print(f"  Cluster {target_label} ('{name}'): {len(new_mesh.faces)} faces ({100*len(new_mesh.faces)/len(mesh.faces):.1f}%)")

        return new_mesh

    def segment_mesh(self, glb_path: str, output_dir: str, num_clusters: int = 3,
                     num_views: int = 8, cluster_method: str = "kmeans",
                     save_debug: bool = True) -> Dict[int, trimesh.Trimesh]:
        """
        Main function to segment a GLB mesh using DINOv2 feature clustering.

        Args:
            glb_path: Path to input GLB file
            output_dir: Directory to save output GLB files
            num_clusters: Number of clusters/parts to segment into
            num_views: Number of views to render
            cluster_method: Clustering method ("kmeans" or "spectral")
            save_debug: Save debug visualizations

        Returns:
            Dictionary mapping cluster IDs to extracted meshes
        """
        os.makedirs(output_dir, exist_ok=True)

        # Load mesh
        print(f"Loading mesh: {glb_path}")
        mesh = self.load_mesh(glb_path)
        print(f"Mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

        # Render views
        print(f"Rendering {num_views} views...")
        views = self.render_views(mesh, num_views=num_views)

        # Extract DINOv2 features
        print("Extracting DINOv2 features...")
        # Get feature dimension from first view
        test_features, _ = self.extract_features(views[0]['color'])
        feature_dim = test_features.shape[-1]
        print(f"Feature dimension: {feature_dim}")

        # Project features to faces
        print("Projecting features to mesh faces...")
        face_features, valid_mask = self.project_features_to_faces(views, len(mesh.faces), feature_dim)
        print(f"Valid faces with direct features: {np.sum(valid_mask)}/{len(mesh.faces)}")

        # Propagate features to neighboring faces without visibility
        face_features, valid_mask = self.propagate_features(mesh, face_features, valid_mask)
        print(f"Valid faces after propagation: {np.sum(valid_mask)}/{len(mesh.faces)}")

        # Cluster faces
        labels = self.cluster_faces(face_features, valid_mask, num_clusters, cluster_method)

        # Save debug visualization
        if save_debug:
            debug_dir = os.path.join(output_dir, "debug")
            os.makedirs(debug_dir, exist_ok=True)

            # Save rendered views
            for i, view in enumerate(views):
                Image.fromarray(view['color']).save(
                    os.path.join(debug_dir, f"view_{i:02d}.png")
                )

            # Create colored mesh showing clusters
            cluster_colors = [
                [255, 0, 0, 255],    # Red
                [0, 255, 0, 255],    # Green
                [0, 0, 255, 255],    # Blue
                [255, 255, 0, 255],  # Yellow
                [255, 0, 255, 255],  # Magenta
                [0, 255, 255, 255],  # Cyan
                [255, 128, 0, 255],  # Orange
                [128, 0, 255, 255],  # Purple
            ]

            face_colors = np.zeros((len(mesh.faces), 4), dtype=np.uint8)
            for i, label in enumerate(labels):
                if label >= 0:
                    face_colors[i] = cluster_colors[label % len(cluster_colors)]
                else:
                    face_colors[i] = [128, 128, 128, 255]  # Gray for unassigned

            colored_mesh = mesh.copy()
            colored_mesh.visual = trimesh.visual.ColorVisuals(colored_mesh, face_colors=face_colors)
            colored_mesh.export(os.path.join(debug_dir, "clustered_mesh.glb"))
            print(f"  Saved debug visualization to {debug_dir}/")

        # Save labels for later use
        labels_path = os.path.join(output_dir, "cluster_labels.npy")
        np.save(labels_path, labels)
        print(f"  Saved cluster labels to {labels_path}")

        # Extract meshes for each cluster
        results = {}
        print(f"\nExtracting {num_clusters} clusters:")

        for cluster_id in range(num_clusters):
            extracted = self.extract_mesh_by_label(mesh, labels, cluster_id, f"part_{cluster_id}")

            if extracted is not None:
                output_path = os.path.join(output_dir, f"cluster_{cluster_id}.glb")
                extracted.export(output_path)
                results[cluster_id] = extracted

        return results, labels, mesh

    def extract_clean_part(self, original_glb: str, labels: np.ndarray,
                           cluster_ids: List[int], output_path: str,
                           use_largest_component: bool = True,
                           offset: int = 0,
                           height_min: float = None,
                           height_max: float = None,
                           fill_holes: bool = False) -> trimesh.Trimesh:
        """
        Extract clean part from original mesh with texture preservation.

        Args:
            original_glb: Path to original GLB file
            labels: Cluster labels for each face
            cluster_ids: List of cluster IDs to include (e.g., [0] for head only)
            output_path: Path to save extracted GLB
            use_largest_component: If True, only keep largest connected component (remove noise)
            offset: Number of face layers to expand selection (for clean cuts)
            height_min: Optional min Y position filter (normalized 0-1)
            height_max: Optional max Y position filter (normalized 0-1)
            fill_holes: If True, clean first (largest component), then expand, no final filtering

        Returns:
            Extracted mesh with original textures
        """
        # Load original mesh
        print(f"Loading original mesh: {original_glb}")
        original_mesh = self.load_mesh(original_glb)

        # Get face indices for selected clusters
        face_mask = np.isin(labels, cluster_ids)
        face_indices = np.where(face_mask)[0]
        print(f"Selected clusters {cluster_ids}: {len(face_indices)} faces")

        # Apply height filter if specified
        if (height_min is not None or height_max is not None) and len(face_indices) > 0:
            face_centers = original_mesh.triangles_center[face_indices]
            y_positions = face_centers[:, 1]

            # Normalize to 0-1
            y_min_mesh, y_max_mesh = original_mesh.vertices[:, 1].min(), original_mesh.vertices[:, 1].max()
            y_normalized = (y_positions - y_min_mesh) / (y_max_mesh - y_min_mesh)

            height_mask = np.ones(len(face_indices), dtype=bool)
            if height_min is not None:
                height_mask &= (y_normalized >= height_min)
            if height_max is not None:
                height_mask &= (y_normalized <= height_max)

            face_indices = face_indices[height_mask]
            print(f"After height filter [{height_min}, {height_max}]: {len(face_indices)} faces")

        if fill_holes:
            # FILL HOLES MODE:
            # 1. First clean (largest component) to get core shape
            # 2. Then expand with offset to fill holes
            # 3. No final filtering - keep all faces
            if len(face_indices) > 0:
                print("Cleaning first (largest connected component)...")
                face_indices = self.find_largest_connected_component(original_mesh, face_indices)

            if offset > 0 and len(face_indices) > 0:
                print(f"Expanding to fill holes with offset={offset}...")
                face_indices = self.expand_face_selection(original_mesh, face_indices, offset)

            print(f"Final selection (no filtering): {len(face_indices)} faces")
        else:
            # ORIGINAL MODE:
            # 1. Expand selection with offset (before filtering)
            # 2. Then find largest connected component
            if offset > 0 and len(face_indices) > 0:
                print(f"Expanding selection with offset={offset}...")
                face_indices = self.expand_face_selection(original_mesh, face_indices, offset)

            if use_largest_component and len(face_indices) > 0:
                print("Finding largest connected component...")
                face_indices = self.find_largest_connected_component(original_mesh, face_indices)

        # Extract from original mesh with textures
        print("Extracting from original mesh with textures...")
        extracted = self.extract_from_original(original_mesh, face_indices, center=True)

        if extracted is not None:
            # Export directly as GLB to preserve textures
            extracted.export(output_path, file_type='glb')
            print(f"Saved: {output_path}")
            print(f"  Vertices: {len(extracted.vertices)}, Faces: {len(extracted.faces)}")

        return extracted


def extract_from_mesh_mask(mask_glb: str, original_glb: str, output_path: str, offset: int = 0):
    """
    Use existing GLB mesh as mask to extract matching faces from original.

    Args:
        mask_glb: Path to mask GLB (e.g., cleaned cluster mesh)
        original_glb: Path to original GLB with textures
        output_path: Path to save result
        offset: Expand selection by N faces to fill holes
    """
    from collections import defaultdict

    # Load meshes
    mask_scene = trimesh.load(mask_glb)
    if isinstance(mask_scene, trimesh.Scene):
        mask = mask_scene.to_geometry()
    else:
        mask = mask_scene

    orig_scene = trimesh.load(original_glb)
    if isinstance(orig_scene, trimesh.Scene):
        original = orig_scene.to_geometry()
    else:
        original = orig_scene

    print(f"Mask mesh: {len(mask.faces)} faces")
    print(f"Original mesh: {len(original.faces)} faces")

    # Match faces by geometry signature (translation-invariant)
    def face_signature(mesh, face_idx):
        verts = mesh.vertices[mesh.faces[face_idx]]
        e1 = np.linalg.norm(verts[1] - verts[0])
        e2 = np.linalg.norm(verts[2] - verts[1])
        e3 = np.linalg.norm(verts[0] - verts[2])
        normal = np.cross(verts[1] - verts[0], verts[2] - verts[0])
        norm_len = np.linalg.norm(normal)
        if norm_len > 1e-10:
            normal = normal / norm_len
        return (round(e1, 6), round(e2, 6), round(e3, 6),
                round(normal[0], 4), round(normal[1], 4), round(normal[2], 4))

    # Build signature map for original
    print("Building face signatures...")
    orig_signatures = {}
    for i in range(len(original.faces)):
        sig = face_signature(original, i)
        if sig not in orig_signatures:
            orig_signatures[sig] = []
        orig_signatures[sig].append(i)

    # Find matching faces
    print("Matching faces...")
    matched_faces = set()
    for i in range(len(mask.faces)):
        sig = face_signature(mask, i)
        if sig in orig_signatures:
            matched_faces.update(orig_signatures[sig][:1])

    print(f"Matched: {len(matched_faces)} faces")

    # Build face adjacency for offset expansion
    if offset > 0:
        edge_to_faces = defaultdict(list)
        for idx in range(len(original.faces)):
            face = original.faces[idx]
            edges = [(min(face[0], face[1]), max(face[0], face[1])),
                     (min(face[1], face[2]), max(face[1], face[2])),
                     (min(face[2], face[0]), max(face[2], face[0]))]
            for edge in edges:
                edge_to_faces[edge].append(idx)

        face_neighbors = defaultdict(set)
        for faces in edge_to_faces.values():
            for i in range(len(faces)):
                for j in range(i + 1, len(faces)):
                    face_neighbors[faces[i]].add(faces[j])
                    face_neighbors[faces[j]].add(faces[i])

        # Expand
        selected = matched_faces
        for i in range(offset):
            new_faces = set()
            for f in selected:
                for neighbor in face_neighbors[f]:
                    if neighbor not in selected:
                        new_faces.add(neighbor)
            selected.update(new_faces)
            print(f"  Offset {i+1}: {len(selected)} faces")

        matched_faces = selected

    # Extract from original (no filtering, keep all)
    result = original.submesh([list(matched_faces)], append=True)
    result.vertices -= result.centroid  # center

    result.export(output_path, file_type='glb')
    print(f"Saved: {output_path} ({len(result.faces)} faces, with textures, centered)")

    return result


def main():
    parser = argparse.ArgumentParser(description="Segment 3D GLB mesh using DINOv2 feature clustering")
    parser.add_argument("--input", "-i", required=True, help="Input GLB file path")
    parser.add_argument("--output_dir", "-o", default="./segmented_dinov2", help="Output directory")
    parser.add_argument("--num_clusters", "-k", type=int, default=3, help="Number of clusters/parts")
    parser.add_argument("--num_views", type=int, default=8, help="Number of views to render")
    parser.add_argument("--method", choices=["kmeans", "spectral"], default="kmeans", help="Clustering method")
    parser.add_argument("--no_debug", action="store_true", help="Disable debug output")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")

    # Options for extracting clean parts
    parser.add_argument("--extract", "-e", type=int, nargs="+", help="Extract specific cluster(s) with noise removal (e.g., --extract 0 or --extract 0 1)")
    parser.add_argument("--extract_output", type=str, default=None, help="Output path for extracted part (default: output_dir/extracted.glb)")
    parser.add_argument("--no_largest_component", action="store_true", help="Don't filter to largest connected component")
    parser.add_argument("--offset", type=int, default=0, help="Expand selection by N face layers for cleaner cuts (e.g., --offset 2)")
    parser.add_argument("--fill_holes", action="store_true", help="Clean first (largest component), then expand to fill holes, no final filtering")
    parser.add_argument("--from_mesh", type=str, help="Use existing GLB mesh as mask to extract from original (match faces by geometry)")

    args = parser.parse_args()

    # If using existing mesh as mask, skip segmentation
    if args.from_mesh:
        extract_output = args.extract_output or "./extracted_from_mask.glb"
        print(f"--- Extracting from original using mesh mask ---")
        extract_from_mesh_mask(
            mask_glb=args.from_mesh,
            original_glb=args.input,
            output_path=extract_output,
            offset=args.offset
        )
        print(f"\nDone!")
        return

    segmenter = DINOv2MeshSegmenter(device=args.device)

    results, labels, mesh = segmenter.segment_mesh(
        glb_path=args.input,
        output_dir=args.output_dir,
        num_clusters=args.num_clusters,
        num_views=args.num_views,
        cluster_method=args.method,
        save_debug=not args.no_debug
    )

    print(f"\nSegmented into {len(results)} parts.")

    # Extract clean part if requested
    if args.extract is not None:
        extract_output = args.extract_output or os.path.join(args.output_dir, "extracted.glb")
        print(f"\n--- Extracting clean part from original ---")
        segmenter.extract_clean_part(
            original_glb=args.input,
            labels=labels,
            cluster_ids=args.extract,
            output_path=extract_output,
            use_largest_component=not args.no_largest_component,
            offset=args.offset,
            fill_holes=args.fill_holes
        )

    print(f"\nDone! Output: {args.output_dir}/")


if __name__ == "__main__":
    main()
