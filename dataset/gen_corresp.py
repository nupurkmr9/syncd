import argparse
import glob
import itertools
import json
import multiprocessing
import os
import subprocess
from pathlib import Path

import numpy as np
import objaverse
import torch
import trimesh
from pytorch3d.renderer import MeshRasterizer, PerspectiveCameras, RasterizationSettings
from pytorch3d.structures import Meshes
from pytorch3d.transforms import Rotate
from tqdm import tqdm


def load_mesh_glb(mesh_filename, json_file=None):
    mesh = trimesh.load(mesh_filename, force='mesh', skip_texture=True)
    mesh.merge_vertices(merge_tex=True, merge_norm=True)
    all_verts = torch.tensor(mesh.vertices).unsqueeze(0).float()
    all_faces = torch.tensor(mesh.faces).unsqueeze(0)

    with open(json_file, 'r') as f:
        transforms = json.load(f)
        bbox = transforms['bbox']

    rotate = torch.from_numpy(np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])).float()
    bbox = rotate@(torch.from_numpy(np.array(bbox)).float()).T
    bbox = bbox.T

    # Center verts
    all_verts_min = mesh.bounding_box.bounds[0]
    all_verts_max = mesh.bounding_box.bounds[1]
    scale = 1 / max((all_verts_max - all_verts_min).reshape(-1))
    all_verts_center = all_verts_min + ((all_verts_max - all_verts_min) * 0.5)
    all_verts_normalized = all_verts - all_verts_center
    all_verts_normalized = all_verts_normalized * scale

    # blender and pytorch rendering matches with the mesh bounding box as min and max
    if (torch.stack([all_verts_normalized[0].min(0).values, all_verts_normalized[0].max(0).values], dim=0) - bbox).abs().sum() < 0.01:
        mesh = Meshes(verts=all_verts_normalized.float(), faces=all_faces)
        return mesh

    # Center verts again with mesh.bounding_box_oriented.bounds
    all_verts_min = mesh.bounding_box_oriented.bounds[0]
    all_verts_max = mesh.bounding_box_oriented.bounds[1]
    scale = 1 / max((all_verts_max - all_verts_min).reshape(-1))
    all_verts_center = all_verts_min + ((all_verts_max - all_verts_min) * 0.5)
    all_verts_normalized = all_verts - all_verts_center
    all_verts_normalized = all_verts_normalized * scale
    mesh = Meshes(verts=all_verts_normalized.float(), faces=all_faces)
    return mesh


def transform_mesh(mesh, transform, scale=1.0):
    mesh = mesh.clone()
    verts = transform.transform_points(mesh.verts_packed())
    verts = verts * scale
    mesh.offset_verts_(verts - mesh.verts_packed())
    return mesh


def cartesian_to_spherical(xyz):
    xy = xyz[:, 0]**2 + xyz[:, 2]**2
    z = np.sqrt(xy + xyz[:, 1]**2)
    theta = np.arctan2(xyz[:, 1], np.sqrt(xy))
    azimuth = np.arctan2(xyz[:, 2], xyz[:, 0])
    return np.rad2deg(theta), np.rad2deg(azimuth), z


def getw2cpy(transforms):
    x_vector = transforms['x']
    y_vector = transforms['y']
    z_vector = transforms['z']
    origin = transforms['origin']

    rotation_matrix = np.array([x_vector, y_vector, z_vector]).T

    translation_vector = np.array(origin)

    rt_matrix = np.eye(4)
    rt_matrix[:3, :3] = rotation_matrix
    rt_matrix[:3, 3] = translation_vector

    R, T = rotation_matrix, translation_vector

    st = np.array([[-1., 0., 0.], [0., 0., 1.], [0., -1, 0.]])
    st1 = np.array([[-1., 0., 0.], [0., -1., 0.], [0., 0, 1.]])

    R = (st@R@st1.T)
    T = st@T
    rt_matrix_py = np.eye(4)
    rt_matrix_py[:3, :3] = R
    rt_matrix_py[:3, 3] = T
    w2c_py = np.linalg.inv(rt_matrix_py)

    return w2c_py


def getcorresp(zbuffer, cameras, resolution, device):
    # print(resolution)
    depth_sample = zbuffer.permute(0, 3, 1, 2)
    depth_sample = torch.nn.functional.interpolate(depth_sample, (resolution, resolution), align_corners=False, antialias=True, mode="bilinear")

    # create a grid
    num_patches_x = num_patches_y = resolution
    horizontal_patch_edges = 2*torch.linspace(1, 0, num_patches_x+1) - 1.
    vertical_patch_edges = 2*torch.linspace(1, 0, num_patches_y+1) - 1.
    horizontal_patch_edges = (
            horizontal_patch_edges[:-1] + horizontal_patch_edges[1:]
        ) / 2
    vertical_patch_edges = (
            vertical_patch_edges[:-1] + vertical_patch_edges[1:]
        ) / 2
    h_pos, v_pos = torch.meshgrid(
            horizontal_patch_edges, vertical_patch_edges, indexing='xy'
        )
    grid = torch.stack([h_pos, v_pos], -1).unsqueeze(0).expand(2, -1, -1, -1)
    grid = grid.to(device)
    grid1 = torch.cat([h_pos.reshape(-1, 1).to(device), v_pos.reshape(-1, 1).to(device), depth_sample[0, 0].reshape(-1, 1)], dim=-1)

    # unproject the NDC grid to world coordinate from first camera and reproject in the second camera
    point1 = cameras[0].unproject_points(grid1.to(device), from_ndc=True, world_coordinates=True)
    point2 = cameras[1].transform_points(point1)
    point2_depth_reprojected = (1./point2[:, 2]).reshape(-1, 1)
    point2_depth_original = torch.nn.functional.grid_sample(depth_sample[1:, :, :, :], -1.*point2[:, :2].reshape(1, resolution, resolution, 2), align_corners=False, mode="nearest")

    # check for same depth
    common_pixels = (((point2_depth_original.reshape(-1, 1) - point2_depth_reprojected).abs() < 0.01)*(point2_depth_original.reshape(-1, 1) > 0.)*(point2_depth_reprojected > 0.)).nonzero()[:, 0]

    corresp2 = point2[:, :2][common_pixels].contiguous()
    corresp1 = grid1[:, :2][common_pixels].contiguous()

    correspondence = torch.cat([corresp1[:, [1, 0]], corresp2[:, [1, 0]]], -1)
    del grid, grid1
    return correspondence


def all_pairs_correspondece(uid, mesh_path, rendered_path, num=3, device='cuda', resolution=128):
    if os.path.exists(f'{rendered_path}/{uid}.zip') and not os.path.exists(f'{rendered_path}/{uid}'):
        pathToZip = f'{rendered_path}/{uid}.zip'
        pathToOut = f'{rendered_path}/{uid}'
        unzip = ['unzip', '-o', pathToZip, '-d', pathToOut]
        _ = subprocess.call(unzip)
    json_files = glob.glob(f'{rendered_path}/{uid}/0*.json')
    mesh = load_mesh_glb(mesh_path, json_files[0])
    rot = torch.from_numpy(np.array([[-1., 0., 0.], [0., 1., 0.], [0., 0, 1.]])).T
    rot = Rotate(rot)
    mesh = transform_mesh(mesh, rot)

    imagenames = []
    corresps = []
    w2c_pys = []
    foc_selects = []

    raster_settings = RasterizationSettings(
            image_size=resolution,
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size=0
        )

    for files in json_files:
        with open(files, 'r') as f:
            transforms = json.load(f)
        imagestem = Path(files).stem
        imageparent = Path(files).parent
        imagenames.append(f'{imageparent}/{imagestem}.png')

        w2c_py = getw2cpy(transforms)
        fov_select = transforms['x_fov']
        w2c_pys.append(w2c_py)
        foc_selects.append(1/np.tan(fov_select / 2))

    foc_selects = np.stack(foc_selects)
    w2c_pys = np.stack(w2c_pys)

    # ipdb.set_trace()
    if w2c_pys.shape[0] < 2:
        return None, None

    camera_center = torch.einsum('bhw,bwc->bhc', -1*torch.from_numpy(w2c_pys[:, :3, :3]).permute(0, 2, 1), torch.from_numpy(w2c_pys[:, :3, 3]).reshape(-1, 3, 1))
    elev, _, _ = cartesian_to_spherical(camera_center)
    elev_mask = (elev.reshape(-1) > 0)

    imagenames = [x for i, x in enumerate(imagenames) if elev_mask[i]]
    if len(imagenames) < 2:
        return None, None
    w2c_pys = w2c_pys[elev_mask]
    foc_selects = foc_selects[elev_mask]

    cameras = PerspectiveCameras(device=device,
                                 R=torch.from_numpy(w2c_pys[:, :3, :3]).permute(0, 2, 1),
                                 T=torch.from_numpy(w2c_pys[:, :3, 3]).reshape(-1, 3),
                                 focal_length=torch.from_numpy(foc_selects).float()
                                 )

    rasterizer = MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        )

    depth = rasterizer(mesh.to(device).extend(w2c_pys.shape[0]))
    corresps_imagenames = []

    for comb in itertools.combinations(np.arange(len(imagenames)), num):
        corresp_num = []
        cameras_ = [cameras[int(x)] for x in comb]
        for comb_ in [[0, 1], [0, 2], [1, 2]]:
            corresp = getcorresp(depth.zbuf[comb, ..., :1][comb_], [cameras_[int(x)] for x in comb_], resolution, device)
            if corresp.shape[0] > 0.1*resolution*resolution:
                corresp_num.append(corresp)
        if len(corresp_num) == num:
            corresps_imagenames.append([imagenames[x] for x in comb])
            corresps.append(corresp_num)

    return corresps, corresps_imagenames


def get_correspondece(objaverse_ids, outdir, rendered_path):
    torch.cuda.set_device(0)
    os.makedirs(f'{outdir}/correspondence/', exist_ok=True)

    for uid, mesh_path in tqdm(objaverse_ids.items()):
        try:
            with torch.no_grad():
                corresp, imagenames = all_pairs_correspondece(uid, mesh_path, rendered_path, num=3, resolution=128)
                if corresp is not None and len(corresp) > 0:
                    torch.save([corresp, imagenames], f'{outdir}/correspondence/{uid}.pt')
            torch.cuda.empty_cache()
        except Exception as e:
            continue


def parse_args():
    parser = argparse.ArgumentParser(
        description="get pairwise correspondence b/w NUM rendering of objaverse assets"
    )
    parser.add_argument("--outdir", type=str, help="dir to save corresp", required=True)
    parser.add_argument("--objaverse_path", type=str, help="root path to the objaverse downloadded assets", required=True)
    parser.add_argument("--rendered_path", type=str, help="root path to save rendered images", required=True)
    parser.add_argument("--categories", type=str, help="objaverse id list", default='assets/objaverse_ids.pt')
    parser.add_argument('--download', action="store_true")
    args = parser.parse_args()
    return args


def main(args):
    print(args)
    categories = list(torch.load(args.categories))
    categories = list(set(categories).intersection(set([x.split('.zip')[0] for x in os.listdir(args.rendered_path)])))

    if args.download:
        objaverse.BASE_PATH = args.objaverse_path
        objaverse._VERSIONED_PATH = os.path.join(objaverse.BASE_PATH, "hf-objaverse-v1")

        processes = multiprocessing.cpu_count() - 10
        _ = objaverse.load_objects(
            uids=categories, download_processes=processes
        )

    objaverse_ids = {}
    for folders in glob.glob(f'{args.objaverse_path}/hf-objaverse-v1/glbs/*'):
        for files in glob.glob(f'{folders}/*.glb'):
            uid = files.split("/")[-1].split(".glb")[0]
            if f'{uid}.zip' in os.listdir(f'{args.rendered_path}') and uid in categories:
                objaverse_ids[uid] = files

    get_correspondece(objaverse_ids, args.outdir, args.rendered_path)


if __name__ == "__main__":
    args = parse_args()
    main(args)
