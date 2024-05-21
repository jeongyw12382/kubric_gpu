import argparse
import os
import bpy
import math
import numpy as np

import kubric as kb
from kubric.renderer import Blender

import pickle

PICKLE_PATH = "camera-16.pkl"

def read_syncdreamer_camera():
    with open(PICKLE_PATH, "rb") as fp: 
        loaded_poses = pickle.load(fp)

    out_matrix = np.stack([np.eye(4) for i in range(16)])
    out_matrix[:, :3, :4] = loaded_poses[-1]
    w2c_translate = np.linalg.inv(out_matrix)[:, :3, 3] / 1.5 * 1.3
    return w2c_translate


def azimuth_elevation_to_cartesian(azimuth, elevation, radius):
    # Convert azimuth and elevation from degrees to radians
    azimuth_rad = math.radians(azimuth)
    elevation_rad = math.radians(elevation)

    # Calculate the Cartesian coordinates
    x = radius * math.cos(elevation_rad) * math.cos(azimuth_rad)
    y = radius * math.cos(elevation_rad) * math.sin(azimuth_rad)
    z = radius * math.sin(elevation_rad)

    return x, y, z


def get_zero123pp_camera():
    azimuths = [30, 90, 150, 210, 270, 330]
    elevations = [30, -20, 30, -20, 30, -20] #v1.1

    ret = []
    for azimuth, elevation in zip(azimuths, elevations):
        ret.append(np.array(azimuth_elevation_to_cartesian(azimuth, elevation, 1.3)))

    return np.stack(ret)


def set_cameras(scene):

    scene.camera = kb.PerspectiveCamera(focal_length=35., sensor_width=32)
    sync_dreamer_camera = read_syncdreamer_camera()
    zero123pp_camera  = get_zero123pp_camera()
    all_frames = np.concatenate([sync_dreamer_camera, zero123pp_camera])
    for frame_idx in range(1, 23):
        scene.camera.position = all_frames[frame_idx-1]
        scene.camera.look_at((0, 0, 0))
        scene.camera.keyframe_insert("position", frame_idx)
        scene.camera.keyframe_insert("quaternion", frame_idx)

def render(args):
    scene, rng, output_dir, scratch_dir = kb.setup(args)
    renderer = Blender(scene, scratch_dir, samples_per_pixel=64, background_transparency=True)
    gso = kb.AssetSource.from_manifest(args.gso_assets,)
    scene.add(kb.assets.utils.get_clevr_lights(rng=rng))
    scene.ambient_illumination = kb.Color(0.05, 0.05, 0.05)
    
    set_cameras(scene)

    asset_index = int(kb.as_path(args.job_dir).name)
    asset_ids = sorted(gso._assets.keys())
    obj = gso.create(asset_id=asset_ids[asset_index], scale=1.)
    bbox_max = np.abs(obj.aabbox).max()
    scale = 1.3 / bbox_max * 0.30
    obj = gso.create(asset_id=asset_ids[asset_index], scale=scale)
    scene.add(obj)
    data_stack = renderer.render(return_layers=("rgba",))
    kb.file_io.write_image_dict(data_stack, output_dir)

    # --- Metadata
    kb.file_io.write_json(filename=output_dir / "metadata.json", data={
        "metadata": kb.get_scene_metadata(scene),
        "camera": kb.get_camera_info(scene.camera),
        "instances": kb.get_instance_info(scene, assets_subset=[obj]),
    })

    kb.done()


if __name__ == "__main__":
    parser = kb.ArgumentParser()
# Configuration for the objects of the scene
    parser.set_defaults(frame_end=22, resolution=(512, 512))
    parser.add_argument("--gso_assets", type=str, default="gs://kubric-public/assets/GSO/GSO.json")
    args = parser.parse_args()

    render(args)