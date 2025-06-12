from typing import List
from pathlib import Path
import scipy.spatial as spt
from tqdm import tqdm
import re
import json
import cv2
import numpy as np
from itertools import combinations

from multiprocessing import Pool

longi_pattern = re.compile(r"\[longitude: .*?\]")
lati_pattern = re.compile(r"\[latitude: .*?\]")

MAX_DISTANCE = 0.0001

def find_nn_pair(db_srt_dict, qy_srt_dict, max_distance=None):
    db_list = []
    db_idx_list = []
    qy_list = []
    qy_idx_list = []
    for db_frame_id in db_srt_dict.keys():
        db_list.append(
            [db_srt_dict[db_frame_id]["longi"], db_srt_dict[db_frame_id]["lati"]]
        )
        db_idx_list.append(db_frame_id)
    for qy_frame_id in qy_srt_dict.keys():
        qy_list.append(
            [qy_srt_dict[qy_frame_id]["longi"], qy_srt_dict[qy_frame_id]["lati"]]
        )
        qy_idx_list.append(qy_frame_id)
    db_array = np.array(db_list)
    qy_array = np.array(qy_list)
    tree = spt.cKDTree(data=db_array)
    match_pair = []  # [db, qy]
    for qy_idx, q in enumerate(qy_array):
        distance, index = tree.query(q, k=1)
        if (max_distance is None) or (distance <= max_distance):
            match_pair.append([db_idx_list[index], qy_idx_list[qy_idx]])
    return match_pair


class SRTParser:
    def __init__(self, srt: Path, div: int = 10):
        self.srt = srt
        self.div = div

    def parse(self) -> dict:
        with open(self.srt) as srtf:
            lines = srtf.readlines()

        out_dict = dict()
        for i in range(len(lines) // 6):
            if i % self.div != 0:
                continue
            frame_id_line = lines[i * 6]
            frame_id = int(frame_id_line.strip())
            gps_line = lines[i * 6 + 4]
            longi_strs = longi_pattern.findall(gps_line)
            lati_strs = lati_pattern.findall(gps_line)
            assert len(longi_strs) == 1
            assert len(lati_strs) == 1
            longi = float(longi_strs[0].lstrip("[longitude: ").rstrip("]"))
            lati = float(lati_strs[0].lstrip("[latitude: ").rstrip("]"))
            out_dict[frame_id] = {"longi": longi, "lati": lati}
        return out_dict


class VideoExtractor:
    def __init__(
        self, video_path: Path, export_dir: Path, div: int = 10, scale: float = 1.0, dst_hw=None
    ):
        self.video = cv2.VideoCapture(str(video_path))
        self.export_dir = export_dir
        self.export_dir.mkdir(exist_ok=True, parents=True)
        self.div = div
        self.scale = scale
        self.dst_hw = dst_hw

    def extract(self) -> int:
        image_cnt = 0
        frame_id = 0
        while True:
            ret, frame = self.video.read()
            if not ret:
                break
            if frame_id % self.div == 0:
                if self.dst_hw is not None:
                    frame = cv2.resize(
                        frame,
                        dsize=(self.dst_hw[1], self.dst_hw[0]),
                        interpolation=cv2.INTER_CUBIC,
                    )
                else:
                    if self.scale != 1.0:
                        frame = cv2.resize(
                            frame,
                            dsize=None,
                            fx=self.scale,
                            fy=self.scale,
                            interpolation=cv2.INTER_CUBIC,
                        )
                cv2.imwrite(
                    str(self.export_dir / "{:05d}.png".format(frame_id + 1)),
                    frame,
                )
                image_cnt += 1
            frame_id += 1
        return image_cnt


class OnePairGenerator:
    def __init__(
        self,
        output_dir: Path,
        input_dir: Path,
        pair: List[str],
        db_div: int = 5,
        qy_div: int = 10,
        scale: float = 1.0,
        dst_hw: tuple = None
    ):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.input_dir = input_dir
        assert len(pair) == 2
        db_srt_path = input_dir / (pair[0] + ".SRT")
        db_video_path = input_dir / (pair[0] + ".MP4")
        qy_srt_path = input_dir / (pair[1] + ".SRT")
        qy_video_path = input_dir / (pair[1] + ".MP4")
        self.db_div = db_div
        self.qy_div = qy_div
        self.scale = scale
        self.dst_hw = dst_hw
        self.db_srt_parser = SRTParser(db_srt_path, div=self.db_div)
        self.qy_srt_parser = SRTParser(qy_srt_path, div=self.qy_div)
        self.db_image_dir = self.output_dir / "database"
        self.qy_image_dir = self.output_dir / "query"
        self.db_video = VideoExtractor(
            db_video_path,
            export_dir=self.db_image_dir,
            div=self.db_div,
            scale=self.scale,
            dst_hw=self.dst_hw,
        )
        self.qy_video = VideoExtractor(
            qy_video_path,
            export_dir=self.qy_image_dir,
            div=self.qy_div,
            scale=self.scale,
            dst_hw=self.dst_hw,
        )

    def run(self):
        db_srt_dict = self.db_srt_parser.parse()
        qy_srt_dict = self.qy_srt_parser.parse()
        with open(self.output_dir / "database.json", "w") as db_json_wf:
            json.dump(db_srt_dict, db_json_wf)
        with open(self.output_dir / "query.json", "w") as qy_json_wf:
            json.dump(qy_srt_dict, qy_json_wf)

        # read db video
        db_image_cnt = self.db_video.extract()
        assert db_image_cnt == len(db_srt_dict)

        # read qy video
        qy_image_cnt = self.qy_video.extract()
        assert qy_image_cnt == len(qy_srt_dict)

        match_pair = find_nn_pair(db_srt_dict, qy_srt_dict, MAX_DISTANCE)
        match_file_list = []
        for m in match_pair:
            match_file_list.append(
                [
                    str(
                        self.db_image_dir.relative_to(self.output_dir)
                        / "{:05d}.png".format(m[0])
                    ),
                    str(
                        self.qy_image_dir.relative_to(self.output_dir)
                        / "{:05d}.png".format(m[1])
                    ),
                ]
            )
        with open(self.output_dir / "pair.json", "w") as pair_wf:
            json.dump(match_file_list, pair_wf)

def process_one(output_dir, video_dir, pair):
    generator = OnePairGenerator(
        output_dir / (pair[1] + "_" + pair[0]),
        video_dir,
        pair,
        db_div=5,
        qy_div=20,
        dst_hw=(1080, 1920),
    )
    generator.run()

if __name__ == "__main__":
    dir_video_dict = {
        # "mavic4-1": [
        #     "DJI_20250520205810_0001_D",
        #     "DJI_20250521100316_0002_D",
        #     "DJI_20250522135940_0006_D",
        #     "DJI_20250522183735_0016_D",
        #     "DJI_20250522191408_0017_D",
        #     "DJI_20250522192504_0019_D",
        # ],
        "air3s-1": [
            "DJI_20250608194755_0081_D",
            "DJI_20250609095809_0083_D",
            "DJI_20250609192454_0087_D",
            "DJI_20250610210126_0091_D",
            "DJI_20250611094711_0091_D",
            "DJI_20250611131943_0100_D",
            "DJI_20250611192045_0104_D",
            "DJI_20250611204601_0109_D",
        ],
        "air3s-2": [
            "DJI_20250609193415_0089_D",
            "DJI_20250610211044_0092_D",
            "DJI_20250611095809_0095_D",
            "DJI_20250611131008_0096_D",
            "DJI_20250611193148_0106_D",
            "DJI_20250611205530_0111_D",
        ],
        "air3s-3": [
            "DJI_20250610211745_0093_D",
            "DJI_20250611095249_0092_D",
            "DJI_20250611132419_0101_D",
            "DJI_20250611192659_0105_D",
            "DJI_20250611205034_0110_D",
        ],
        "air3s-4": [
            "DJI_20250611131522_0099_D",
            "DJI_20250611193546_0107_D",
            "DJI_20250611205925_0112_D",
        ]
    }
    output_dir = Path("output1")
    args = []
    pool = Pool(processes=4)
    for wayline_key in dir_video_dict.keys():
        video_dir = Path(wayline_key)
        pairs = combinations(dir_video_dict[wayline_key], 2)
        for pair in pairs:
            pool.apply_async(process_one, (output_dir, video_dir, pair))
    pool.close()
    pool.join()


