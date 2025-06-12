from PIL import Image
import torch
import cv2
from tqdm import tqdm

# from romatch import roma_outdoor
from romatch import tiny_roma_v1_outdoor
import numpy as np
from pathlib import Path
import json


def warp_one(roma_model, im1_path: str, im2_path: str, dst_path: Path):
    dst_path.mkdir(exist_ok=True, parents=True)
    W_A, H_A = Image.open(im1_path).size
    W_B, H_B = Image.open(im2_path).size

    # Match
    # warp, certainty = roma_model.match(im1_path, im2_path, device="cpu")
    warp, certainty = roma_model.match(im1_path, im2_path)
    # Sample matches for estimation
    matches, certainty = roma_model.sample(warp, certainty)
    kpts1, kpts2 = roma_model.to_pixel_coordinates(matches, H_A, W_A, H_B, W_B)
    H, mask = cv2.findHomography(
        kpts1.cpu().numpy(),
        kpts2.cpu().numpy(),
        ransacReprojThreshold=0.2,
        method=cv2.USAC_MAGSAC,
        confidence=0.999999,
        maxIters=10000,
    )
    # print(f"H: {H}")
    cv_img1 = cv2.imread(im1_path)
    cv_img2 = cv2.imread(im2_path)

    dummy_mask = 255 * np.ones((H_A, W_A))
    warped = cv2.warpPerspective(cv_img1, H, (W_A, H_A))
    warped_mask = cv2.warpPerspective(
        dummy_mask, H, (cv_img1.shape[1], cv_img1.shape[0]), flags=cv2.INTER_NEAREST
    )
    cv2.imwrite(str(dst_path / "mask.png"), warped_mask)
    cv2.imwrite(str(dst_path / "warped.png"), warped)
    cv2.imwrite(str(dst_path / "origin.png"), cv_img2)


def warp_dir(roma_model, input_root: Path, output_root: Path):
    with open(input_root / "pair.json") as json_f:
        pairs = json.load(json_f)
    for p in tqdm(pairs, "Warping dir: {}".format(input_root)):
        # output_pair_dir
        db_img_path = Path(p[0])
        qy_img_path = Path(p[1])
        dst = output_root / qy_img_path.stem
        warp_one(
            roma_model=roma_model,
            im1_path=input_root / db_img_path,
            im2_path=input_root / qy_img_path,
            dst_path=dst,
        )


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")

    # device = "cuda"
    device = "cpu"

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "--im_A_path",
        default="/home/gmh/Desktop/down_video_dataset/output/DJI_20250521100316_0002_D/database/01481.png",
        type=str,
    )
    parser.add_argument(
        "--im_B_path",
        default="/home/gmh/Desktop/down_video_dataset/output/DJI_20250521100316_0002_D/query/01421.png",
        type=str,
    )

    args, _ = parser.parse_known_args()
    im1_path = args.im_A_path
    im2_path = args.im_B_path

    # Create model
    # roma_model = roma_outdoor(device=device)
    print(1)
    roma_model = tiny_roma_v1_outdoor(device=device)
    print(2)
    # warp_one(roma_model, im1_path=im1_path, im2_path=im2_path, dst_path=Path("output"))
    all_root = Path("/home/gmh/Desktop/down_video_dataset/output/")
    all_output = Path("output_dir")
    for sub_root in all_root.iterdir():
        sub_name = sub_root.relative_to(all_root)
        warp_dir(
            roma_model=roma_model,
            input_root=sub_root,
            output_root=all_output / sub_name,
        )
