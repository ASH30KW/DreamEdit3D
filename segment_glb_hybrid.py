"""
Hybrid 3D GLB Mesh Segmentation

Combines multiple approaches for robust automatic segmentation:
1. DINOv2 features (semantic/visual understanding)
2. Geometric features (curvature, SDF, normals)
3. Auto-K detection (silhouette score)
4. Graph-cut boundary refinement

No manual tuning needed - fully automatic.

Usage:
    python segment_glb_hybrid.py --input mesh.glb --output_dir ./output
"""

import os
import sys

os.environ['PYOPENGL_PLATFORM'] = 'egl'

import argparse
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import trimesh
import pyrender
from scipy import sparse
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from collections import defaultdict, deque
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class HybridMeshSegmenter:
    def __init__(self, model_name: str = "dinov2_vits14", device: str = "cuda"):
        self.device = device
        self.patch_size = 14

        print("Loading DINOv2...")
        self.dino_model = torch.hub.load('facebookresearch/dinov2', model_name)
        self.dino_model = self.dino_model.to(device)
        self.dino_model.eval()
        print("Models loaded!")

    # ==================== Mesh Loading ====================

    def load_mesh(self, glb_path: str) -> trimesh.Trimesh:
        """Load GLB file and convert to single mesh."""
        scene = trimesh.load(glb_path)
        if isinstance(scene, trimesh.Scene):
            mesh = scene.to_geometry()
        else:
            mesh = scene
        return mesh

    # ==================== Geometric Features ====================

    def compute_face_normals(self, mesh: trimesh.Trimesh) -> np.ndarray:
        """Compute face normals."""
        return mesh.face_normals

    def compute_face_curvature(self, mesh: trimesh.Trimesh) -> np.ndarray:
        """Estimate face curvature using dihedral angles with neighbors."""
        face_adjacency = mesh.face_adjacency
        face_adjacency_angles = mesh.face_adjacency_angles

        # Average curvature per face
        curvature = np.zeros(len(mesh.faces))
        counts = np.zeros(len(mesh.faces))

        for (f1, f2), angle in zip(face_adjacency, face_adjacency_angles):
            curvature[f1] += angle
            curvature[f2] += angle
            counts[f1] += 1
            counts[f2] += 1

        counts[counts == 0] = 1
        curvature = curvature / counts

        return curvature.reshape(-1, 1)

    def compute_shape_diameter(self, mesh: trimesh.Trimesh, num_rays: int = 10) -> np.ndarray:
        """Compute Shape Diameter Function (SDF) - local thickness estimation."""
        face_centers = mesh.triangles_center
        face_normals = mesh.face_normals

        sdf_values = np.zeros(len(mesh.faces))

        # For each face, cast rays inward and measure distance to opposite side
        for i, (center, normal) in enumerate(zip(face_centers, face_normals)):
            # Cast ray inward (opposite to normal)
            ray_direction = -normal

            # Add some random cone variation for robustness
            distances = []
            for _ in range(num_rays):
                # Perturb direction slightly
                perturb = np.random.randn(3) * 0.2
                direction = ray_direction + perturb
                direction = direction / np.linalg.norm(direction)

                # Ray-mesh intersection
                locations, index_ray, index_tri = mesh.ray.intersects_location(
                    ray_origins=[center + normal * 0.001],  # Offset to avoid self-intersection
                    ray_directions=[direction]
                )

                if len(locations) > 0:
                    dist = np.linalg.norm(locations[0] - center)
                    distances.append(dist)

            if distances:
                sdf_values[i] = np.median(distances)
            else:
                sdf_values[i] = 0

        # Normalize
        if sdf_values.max() > 0:
            sdf_values = sdf_values / sdf_values.max()

        return sdf_values.reshape(-1, 1)

    def compute_face_positions(self, mesh: trimesh.Trimesh) -> np.ndarray:
        """Compute normalized face center positions."""
        centers = mesh.triangles_center

        # Normalize to [0, 1]
        mins = centers.min(axis=0)
        maxs = centers.max(axis=0)
        ranges = maxs - mins
        ranges[ranges == 0] = 1

        normalized = (centers - mins) / ranges
        return normalized

    def compute_geometric_features(self, mesh: trimesh.Trimesh,
                                    use_sdf: bool = True) -> np.ndarray:
        """Compute all geometric features for each face."""
        print("  Computing geometric features...")

        features = []

        # Face normals (3D)
        normals = self.compute_face_normals(mesh)
        features.append(normals)
        print(f"    Normals: {normals.shape}")

        # Face curvature (1D)
        curvature = self.compute_face_curvature(mesh)
        features.append(curvature)
        print(f"    Curvature: {curvature.shape}")

        # Normalized positions (3D)
        positions = self.compute_face_positions(mesh)
        features.append(positions)
        print(f"    Positions: {positions.shape}")

        # Shape Diameter Function (1D) - slower but useful
        if use_sdf:
            print("    Computing SDF (this may take a moment)...")
            sdf = self.compute_shape_diameter(mesh, num_rays=5)
            features.append(sdf)
            print(f"    SDF: {sdf.shape}")

        # Concatenate all features
        geometric_features = np.hstack(features)
        print(f"  Total geometric features: {geometric_features.shape}")

        return geometric_features

    # ==================== DINOv2 Features ====================

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
        return (r << 16) | (g << 8) | b

    def _look_at(self, eye, target, up):
        """Create look-at camera matrix."""
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

    def render_views(self, mesh: trimesh.Trimesh, num_views: int = 12,
                     resolution: Tuple[int, int] = (518, 518)) -> List[Dict]:
        """Render mesh from multiple views with multiple elevations."""
        views = []

        scene_color = pyrender.Scene(bg_color=[255, 255, 255, 255])
        pr_mesh_color = pyrender.Mesh.from_trimesh(mesh)
        scene_color.add(pr_mesh_color)

        face_colors = self.create_face_id_colors(len(mesh.faces))
        mesh_id = mesh.copy()
        mesh_id.visual = trimesh.visual.ColorVisuals(mesh_id, face_colors=face_colors)
        scene_id = pyrender.Scene(bg_color=[0, 0, 0, 255])
        pr_mesh_id = pyrender.Mesh.from_trimesh(mesh_id, smooth=False)
        scene_id.add(pr_mesh_id)

        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
        bounds = mesh.bounds
        center = mesh.centroid
        scale = np.max(bounds[1] - bounds[0])
        distance = scale * 2.0

        renderer = pyrender.OffscreenRenderer(resolution[0], resolution[1])

        # Multiple elevations for better coverage
        camera_positions = []
        for i in range(num_views):
            angle = 2 * np.pi * i / num_views
            camera_positions.append((angle, 0.2))
        for i in range(num_views // 2):
            angle = 2 * np.pi * i / (num_views // 2) + np.pi / num_views
            camera_positions.append((angle, 0.6))
        camera_positions.append((0, 1.4))  # Top
        camera_positions.append((0, -1.0))  # Bottom

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
            flags = pyrender.RenderFlags.FLAT | pyrender.RenderFlags.SKIP_CULL_FACES
            id_img, _ = renderer.render(scene_id, flags=flags)

            views.append({
                'color': color_img,
                'face_ids': self.decode_face_ids(id_img),
            })

            scene_color.remove_node(cam_node_color)
            scene_color.remove_node(light_node)
            scene_id.remove_node(cam_node_id)

        renderer.delete()
        return views

    def extract_dino_features(self, image: np.ndarray) -> np.ndarray:
        """Extract DINOv2 features from an image."""
        img = Image.fromarray(image)
        h, w = image.shape[:2]
        new_h = (h // self.patch_size) * self.patch_size
        new_w = (w // self.patch_size) * self.patch_size
        img = img.resize((new_w, new_h), Image.BILINEAR)

        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        img_tensor = (img_tensor - mean) / std

        with torch.no_grad():
            features = self.dino_model.forward_features(img_tensor)
            patch_tokens = features['x_norm_patchtokens']

        num_patches_h = new_h // self.patch_size
        num_patches_w = new_w // self.patch_size
        feature_map = patch_tokens.reshape(1, num_patches_h, num_patches_w, -1)
        feature_map = feature_map.squeeze(0).cpu().numpy()

        return feature_map, (new_h, new_w)

    def compute_dino_features(self, mesh: trimesh.Trimesh, num_views: int = 12) -> np.ndarray:
        """Compute DINOv2 features for each face."""
        print("  Rendering views...")
        views = self.render_views(mesh, num_views=num_views)

        print("  Extracting DINOv2 features...")
        test_features, _ = self.extract_dino_features(views[0]['color'])
        feature_dim = test_features.shape[-1]

        face_features = np.zeros((len(mesh.faces), feature_dim), dtype=np.float32)
        face_counts = np.zeros(len(mesh.faces), dtype=np.float32)

        for view in views:
            feature_map, (new_h, new_w) = self.extract_dino_features(view['color'])
            feat_h, feat_w, _ = feature_map.shape

            face_ids_resized = np.array(
                Image.fromarray(view['face_ids'].astype(np.int32)).resize(
                    (feat_w, feat_h), Image.NEAREST
                )
            )

            for y in range(feat_h):
                for x in range(feat_w):
                    fid = face_ids_resized[y, x]
                    if 0 < fid < len(mesh.faces):
                        face_features[fid] += feature_map[y, x]
                        face_counts[fid] += 1

        # Propagate features to faces without direct visibility
        face_features, face_counts = self._propagate_features(mesh, face_features, face_counts)

        valid_mask = face_counts > 0
        face_features[valid_mask] /= face_counts[valid_mask, np.newaxis]

        print(f"  DINOv2 features: {face_features.shape}, coverage: {valid_mask.sum()}/{len(mesh.faces)}")

        return face_features, valid_mask

    def _propagate_features(self, mesh, features, counts, iterations=5):
        """Propagate features to neighboring faces."""
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

        for _ in range(iterations):
            updated = 0
            for face_idx in range(len(mesh.faces)):
                if counts[face_idx] > 0:
                    continue
                neighbor_features = []
                for neighbor in face_neighbors[face_idx]:
                    if counts[neighbor] > 0:
                        neighbor_features.append(features[neighbor] / counts[neighbor])
                if neighbor_features:
                    features[face_idx] = np.mean(neighbor_features, axis=0)
                    counts[face_idx] = 1
                    updated += 1
            if updated == 0:
                break

        return features, counts

    # ==================== Auto-K Detection ====================

    def find_optimal_k(self, features: np.ndarray, k_range: Tuple[int, int] = (2, 8)) -> int:
        """Find optimal number of clusters using silhouette score."""
        print("  Finding optimal K...")

        scaler = StandardScaler()
        features_norm = scaler.fit_transform(features)

        # Subsample for speed if too many faces
        if len(features_norm) > 5000:
            indices = np.random.choice(len(features_norm), 5000, replace=False)
            features_sample = features_norm[indices]
        else:
            features_sample = features_norm

        best_k = k_range[0]
        best_score = -1

        for k in range(k_range[0], k_range[1] + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=5, max_iter=100)
            labels = kmeans.fit_predict(features_sample)
            score = silhouette_score(features_sample, labels)
            print(f"    K={k}: silhouette={score:.3f}")
            if score > best_score:
                best_score = score
                best_k = k

        print(f"  Optimal K={best_k} (score={best_score:.3f})")
        return best_k

    # ==================== Graph-Cut Boundary Refinement ====================

    def build_face_adjacency_matrix(self, mesh: trimesh.Trimesh) -> sparse.csr_matrix:
        """Build sparse adjacency matrix for faces."""
        n_faces = len(mesh.faces)
        rows, cols, data = [], [], []

        edge_to_faces = defaultdict(list)
        for face_idx, face in enumerate(mesh.faces):
            edges = [(min(face[0], face[1]), max(face[0], face[1])),
                     (min(face[1], face[2]), max(face[1], face[2])),
                     (min(face[2], face[0]), max(face[2], face[0]))]
            for edge in edges:
                edge_to_faces[edge].append(face_idx)

        for faces in edge_to_faces.values():
            for i in range(len(faces)):
                for j in range(i + 1, len(faces)):
                    rows.extend([faces[i], faces[j]])
                    cols.extend([faces[j], faces[i]])
                    data.extend([1.0, 1.0])

        return sparse.csr_matrix((data, (rows, cols)), shape=(n_faces, n_faces))

    def refine_boundaries_graphcut(self, mesh: trimesh.Trimesh, labels: np.ndarray,
                                    features: np.ndarray, iterations: int = 3) -> np.ndarray:
        """Refine cluster boundaries using graph-cut style optimization."""
        print("  Refining boundaries...")

        adjacency = self.build_face_adjacency_matrix(mesh)
        refined_labels = labels.copy()

        # Compute cluster centers
        n_clusters = len(np.unique(labels[labels >= 0]))
        scaler = StandardScaler()
        features_norm = scaler.fit_transform(features)

        for iteration in range(iterations):
            changed = 0
            cluster_centers = np.zeros((n_clusters, features_norm.shape[1]))
            cluster_counts = np.zeros(n_clusters)

            for i, label in enumerate(refined_labels):
                if label >= 0:
                    cluster_centers[label] += features_norm[i]
                    cluster_counts[label] += 1

            cluster_counts[cluster_counts == 0] = 1
            cluster_centers /= cluster_counts[:, np.newaxis]

            # For each face, consider switching to neighbor's cluster if it reduces energy
            for face_idx in range(len(mesh.faces)):
                if refined_labels[face_idx] < 0:
                    continue

                current_label = refined_labels[face_idx]

                # Get neighbor labels
                neighbors = adjacency[face_idx].nonzero()[1]
                neighbor_labels = refined_labels[neighbors]
                neighbor_labels = neighbor_labels[neighbor_labels >= 0]

                if len(neighbor_labels) == 0:
                    continue

                # Count neighbor labels
                unique_labels, counts = np.unique(neighbor_labels, return_counts=True)

                # Feature distance to each cluster center
                feature = features_norm[face_idx]
                current_dist = np.linalg.norm(feature - cluster_centers[current_label])

                best_label = current_label
                best_score = current_dist - 0.5 * (counts[unique_labels == current_label].sum() if current_label in unique_labels else 0)

                for label, count in zip(unique_labels, counts):
                    if label != current_label:
                        dist = np.linalg.norm(feature - cluster_centers[label])
                        # Prefer labels with more neighbors (smoothness) and closer features
                        score = dist - 0.5 * count
                        if score < best_score:
                            best_score = score
                            best_label = label

                if best_label != current_label:
                    refined_labels[face_idx] = best_label
                    changed += 1

            print(f"    Iteration {iteration + 1}: {changed} faces changed")
            if changed == 0:
                break

        return refined_labels

    # ==================== Extraction ====================

    def find_largest_component(self, mesh: trimesh.Trimesh, face_indices: np.ndarray) -> np.ndarray:
        """Find largest connected component."""
        if len(face_indices) == 0:
            return face_indices

        edge_to_faces = defaultdict(list)
        for face_idx in face_indices:
            face = mesh.faces[face_idx]
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

        visited = set()
        components = []

        for start_face in face_indices:
            if start_face in visited:
                continue
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

        largest = max(components, key=len)
        return np.array(largest)

    def extract_part(self, mesh: trimesh.Trimesh, face_indices: np.ndarray,
                     center: bool = True) -> Optional[trimesh.Trimesh]:
        """Extract submesh preserving textures."""
        if len(face_indices) == 0:
            return None
        new_mesh = mesh.submesh([face_indices], append=True)
        if center:
            new_mesh.vertices -= new_mesh.centroid
        return new_mesh

    def expand_selection(self, mesh: trimesh.Trimesh, face_indices: np.ndarray,
                         iterations: int = 1) -> np.ndarray:
        """Expand face selection by N iterations (adds neighboring faces)."""
        if len(face_indices) == 0 or iterations <= 0:
            return face_indices

        # Build face adjacency
        edge_to_faces = defaultdict(list)
        for face_idx in range(len(mesh.faces)):
            face = mesh.faces[face_idx]
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

        selected = set(face_indices)
        for _ in range(iterations):
            new_faces = set()
            for face_idx in selected:
                for neighbor in face_neighbors[face_idx]:
                    if neighbor not in selected:
                        new_faces.add(neighbor)
            selected.update(new_faces)

        return np.array(list(selected))

    # ==================== Main Pipeline ====================

    def segment(self, glb_path: str, output_dir: str,
                use_sdf: bool = False,
                auto_k: bool = True,
                k: int = 3,
                refine_boundaries: bool = True,
                min_component_ratio: float = 0.05,
                filter_components: bool = True,
                offset: int = 0) -> Dict:
        """
        Main segmentation pipeline.

        Args:
            glb_path: Path to input GLB
            output_dir: Output directory
            use_sdf: Use Shape Diameter Function (slower but better for organic shapes)
            auto_k: Automatically detect number of clusters
            k: Number of clusters (if auto_k=False)
            refine_boundaries: Apply graph-cut boundary refinement
            min_component_ratio: Minimum component size as ratio of total faces
            filter_components: Filter to largest connected component per cluster
            offset: Expand each part by N faces

        Returns:
            Dictionary with results
        """
        os.makedirs(output_dir, exist_ok=True)

        # Load mesh
        print(f"\n{'='*50}")
        print(f"Loading: {glb_path}")
        mesh = self.load_mesh(glb_path)
        print(f"Mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

        # Compute features
        print(f"\n{'='*50}")
        print("Computing features...")

        # Geometric features
        geometric_features = self.compute_geometric_features(mesh, use_sdf=use_sdf)

        # DINOv2 features
        dino_features, valid_mask = self.compute_dino_features(mesh, num_views=12)

        # Combine features (weight DINOv2 higher for semantic understanding)
        print("\n  Combining features...")
        dino_weight = 2.0
        geo_weight = 1.0

        # Normalize each feature set
        scaler_geo = StandardScaler()
        scaler_dino = StandardScaler()

        geo_norm = scaler_geo.fit_transform(geometric_features)
        dino_norm = scaler_dino.fit_transform(dino_features)

        # Combine
        combined_features = np.hstack([
            geo_norm * geo_weight,
            dino_norm * dino_weight
        ])
        print(f"  Combined features: {combined_features.shape}")

        # Auto-detect K
        print(f"\n{'='*50}")
        if auto_k:
            optimal_k = self.find_optimal_k(combined_features[valid_mask], k_range=(2, 6))
        else:
            optimal_k = k
            print(f"  Using K={k}")

        # Clustering
        print(f"\n{'='*50}")
        print(f"Clustering with K={optimal_k}...")

        scaler = StandardScaler()
        features_norm = scaler.fit_transform(combined_features)

        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        labels = np.full(len(mesh.faces), -1, dtype=np.int32)
        labels[valid_mask] = kmeans.fit_predict(features_norm[valid_mask])

        # Boundary refinement
        if refine_boundaries:
            print(f"\n{'='*50}")
            labels = self.refine_boundaries_graphcut(mesh, labels, combined_features)

        # Save cluster labels
        np.save(os.path.join(output_dir, "cluster_labels.npy"), labels)

        # Extract and save parts
        print(f"\n{'='*50}")
        print("Extracting parts...")

        results = {}
        min_faces = int(len(mesh.faces) * min_component_ratio)

        # Analyze clusters by position
        cluster_info = []
        for cluster_id in range(optimal_k):
            face_indices = np.where(labels == cluster_id)[0]
            if len(face_indices) > 0:
                centers = mesh.triangles_center[face_indices]
                avg_y = centers[:, 1].mean()
                cluster_info.append({
                    'id': cluster_id,
                    'faces': len(face_indices),
                    'avg_y': avg_y
                })

        # Sort by Y position (top to bottom)
        cluster_info.sort(key=lambda x: -x['avg_y'])

        # Name clusters based on position
        part_names = ['top', 'middle', 'bottom', 'part_3', 'part_4', 'part_5']

        for i, info in enumerate(cluster_info):
            cluster_id = info['id']
            part_name = part_names[i] if i < len(part_names) else f"part_{i}"

            face_indices = np.where(labels == cluster_id)[0]

            # Optional: Find largest connected component
            if filter_components:
                face_indices = self.find_largest_component(mesh, face_indices)

            # Optional: Expand selection
            if offset > 0:
                face_indices = self.expand_selection(mesh, face_indices, offset)

            if len(face_indices) < min_faces:
                print(f"  Skipping cluster {cluster_id} ({part_name}): too small ({len(face_indices)} faces)")
                continue

            # Extract with textures
            extracted = self.extract_part(mesh, face_indices)

            if extracted is not None:
                output_path = os.path.join(output_dir, f"{part_name}.glb")
                extracted.export(output_path, file_type='glb')
                print(f"  {part_name}: {len(extracted.faces)} faces ({100*len(extracted.faces)/len(mesh.faces):.1f}%)")
                results[part_name] = {
                    'mesh': extracted,
                    'faces': len(extracted.faces),
                    'cluster_id': cluster_id
                }

        # Save debug visualization
        print(f"\n{'='*50}")
        print("Saving debug visualization...")
        debug_dir = os.path.join(output_dir, "debug")
        os.makedirs(debug_dir, exist_ok=True)

        cluster_colors = [
            [255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255],
            [255, 255, 0, 255], [255, 0, 255, 255], [0, 255, 255, 255],
        ]

        face_colors = np.zeros((len(mesh.faces), 4), dtype=np.uint8)
        for i, label in enumerate(labels):
            if label >= 0:
                face_colors[i] = cluster_colors[label % len(cluster_colors)]
            else:
                face_colors[i] = [128, 128, 128, 255]

        colored_mesh = mesh.copy()
        colored_mesh.visual = trimesh.visual.ColorVisuals(colored_mesh, face_colors=face_colors)
        colored_mesh.export(os.path.join(debug_dir, "clustered_mesh.glb"))

        print(f"\n{'='*50}")
        print(f"Done! Output: {output_dir}/")
        print(f"  Parts: {list(results.keys())}")

        return results


def main():
    parser = argparse.ArgumentParser(description="Hybrid 3D mesh segmentation")
    parser.add_argument("--input", "-i", required=True, help="Input GLB file")
    parser.add_argument("--output_dir", "-o", default="./segmented_hybrid", help="Output directory")
    parser.add_argument("--use_sdf", action="store_true", help="Use Shape Diameter Function (slower)")
    parser.add_argument("--no_auto_k", action="store_true", help="Disable auto-K detection")
    parser.add_argument("--k", type=int, default=3, help="Number of clusters (if --no_auto_k)")
    parser.add_argument("--no_refine", action="store_true", help="Disable boundary refinement")
    parser.add_argument("--keep_all", action="store_true", help="Keep all faces (disable largest component filtering)")
    parser.add_argument("--offset", type=int, default=0, help="Expand each part by N faces")
    parser.add_argument("--device", default="cuda", help="Device")

    args = parser.parse_args()

    segmenter = HybridMeshSegmenter(device=args.device)

    results = segmenter.segment(
        glb_path=args.input,
        output_dir=args.output_dir,
        use_sdf=args.use_sdf,
        auto_k=not args.no_auto_k,
        k=args.k,
        refine_boundaries=not args.no_refine,
        filter_components=not args.keep_all,
        offset=args.offset
    )


if __name__ == "__main__":
    main()
