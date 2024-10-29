import glob
import os
import json
import cv2


def load_from_annos(anno_path):
    with open(anno_path, "r") as f:
        annos = json.load(f)["files"]

    datas = []
    for i, anno in enumerate(annos):
        rgb = anno["rgb"]
        depth = anno["depth"] if "depth" in anno else None
        depth_scale = anno["depth_scale"] if "depth_scale" in anno else 1.0
        intrinsic = anno["cam_in"] if "cam_in" in anno else None
        normal = anno["normal"] if "normal" in anno else None

        data_i = {
            "rgb": rgb,
            "depth": depth,
            "depth_scale": depth_scale,
            "intrinsic": intrinsic,
            "filename": os.path.basename(rgb),
            "folder": rgb.split("/")[-3],
            "normal": normal,
        }
        datas.append(data_i)
    return datas


def load_data(path: str, out_path: str, split: int = 1, part: int = 1):
    rgbs = glob.glob(path + "/*.jpg") + glob.glob(path + "/*.png")
    rgbs.sort()
    num_per_part = len(rgbs) // split
    rgbs = rgbs[num_per_part * (part - 1) : num_per_part * part]
    res_rgbs = []
    for i in rgbs:
        if os.path.exists(
            os.path.join(out_path, os.path.basename(i).replace(".jpg", ".npz"))
        ):
            continue
        res_rgbs.append(i)
    print(f"{len(res_rgbs)} | Total {len(rgbs)} not processed.")
    # intrinsic =  [835.8179931640625, 835.8179931640625, 961.5419921875, 566.8090209960938] #[721.53769, 721.53769, 609.5593, 172.854]
    intrinsic = [
        400,
        400,
        400,
        300,
    ]
    data = [
        {
            "rgb": i,
            "depth": None,
            "intrinsic": intrinsic,
            "filename": os.path.basename(i),
            # "folder": i.split("/")[-3],
            "folder": None,
        }
        for i in res_rgbs
    ]
    return data


if __name__ == "__main__":
    load_data(
        "/home/user/XJH/Geo-Loc/geo-loc-data/data/manville-panos/pov",
        "/home/user/XJH/Geo-Loc/geo-loc-data/data/manville-panos/pov-depth/pred",
        1,
        1,
    )
