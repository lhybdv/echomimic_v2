import argparse
import os
import random
import time
from multiprocessing.pool import ThreadPool
from pathlib import Path

import cv2
import decord
import numpy as np
import torch
from diffusers import AutoencoderKL, DDIMScheduler
from moviepy.editor import AudioFileClip, VideoFileClip
from omegaconf import OmegaConf
from PIL import Image

from src.models.dwpose.dwpose_detector import dwpose_detector as dwprocessor
from src.models.pose_encoder import PoseEncoder
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d_emo import EMOUNet3DConditionModel
from src.models.whisper.audio2feature import load_audio_model
from src.pipelines.pipeline_echomimicv2 import EchoMimicV2Pipeline
from src.utils.dwpose_util import draw_pose_select_v2
from src.utils.util import save_videos_grid

refimg_path = "./assets/halfbody_demo/refimag/test.png"
audio_path = "./assets/halfbody_demo/audio/chinese/echomimicv2_woman.wav"
using_video_driving = False
if not using_video_driving:
    pose_path = "./assets/halfbody_demo/pose/01"


##################################
process_num = 100  # 1266

start = 0
end = process_num + start
#################################
MAX_SIZE = 768


def convert_fps(src_path, tgt_path, tgt_fps=24, tgt_sr=16000):
    clip = VideoFileClip(src_path)
    new_clip = clip.set_fps(tgt_fps)
    if tgt_fps is not None:
        audio = new_clip.audio
        audio = audio.set_fps(tgt_sr)
        new_clip = new_clip.set_audio(audio)
    if ".mov" in tgt_path:
        tgt_path = tgt_path.replace(".mov", ".mp4")
    new_clip.write_videofile(tgt_path, codec="libx264", audio_codec="aac")


def get_video_pose(video_path: str, sample_stride: int = 1, max_frame=None):
    # read input video
    vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
    sample_stride *= max(1, int(vr.get_avg_fps() / 24))

    frames = vr.get_batch(list(range(0, len(vr), sample_stride))).asnumpy()
    # print(frames[0])
    if max_frame is not None:
        frames = frames[0:max_frame, :, :]
    height, width, _ = frames[0].shape
    detected_poses = [dwprocessor(frm) for frm in frames]
    dwprocessor.release_memory()

    return detected_poses, height, width, frames


def resize_and_pad(img, max_size):
    img_new = np.zeros((max_size, max_size, 3)).astype("uint8")
    imh, imw = img.shape[0], img.shape[1]
    half = max_size // 2
    if imh > imw:
        imh_new = max_size
        imw_new = int(round(imw / imh * imh_new))
        half_w = imw_new // 2
        rb, re = 0, max_size
        cb = half - half_w
        ce = cb + imw_new
    else:
        imw_new = max_size
        imh_new = int(round(imh / imw * imw_new))
        half_h = imh_new // 2
        cb, ce = 0, max_size
        rb = half - half_h
        re = rb + imh_new

    img_resize = cv2.resize(img, (imw_new, imh_new))
    img_new[rb:re, cb:ce, :] = img_resize
    return img_new


def resize_and_pad_param(imh, imw, max_size):
    half = max_size // 2
    if imh > imw:
        imh_new = max_size
        imw_new = int(round(imw / imh * imh_new))
        half_w = imw_new // 2
        rb, re = 0, max_size
        cb = half - half_w
        ce = cb + imw_new
    else:
        imw_new = max_size
        imh_new = int(round(imh / imw * imw_new))
        imh_new = max_size

        half_h = imh_new // 2
        cb, ce = 0, max_size
        rb = half - half_h
        re = rb + imh_new

    return imh_new, imw_new, rb, re, cb, ce


def get_pose_params(detected_poses, max_size):
    print("get_pose_params...")
    # pose rescale
    w_min_all, w_max_all, h_min_all, h_max_all = [], [], [], []
    mid_all = []
    for num, detected_pose in enumerate(detected_poses):
        detected_poses[num]["num"] = num
        candidate_body = detected_pose["bodies"]["candidate"]
        score_body = detected_pose["bodies"]["score"]
        candidate_face = detected_pose["faces"]
        score_face = detected_pose["faces_score"]
        candidate_hand = detected_pose["hands"]
        score_hand = detected_pose["hands_score"]

        # face
        if candidate_face.shape[0] > 1:
            index = 0
            candidate_face = candidate_face[index]
            score_face = score_face[index]
            detected_poses[num]["faces"] = candidate_face.reshape(
                1, candidate_face.shape[0], candidate_face.shape[1]
            )
            detected_poses[num]["faces_score"] = score_face.reshape(
                1, score_face.shape[0]
            )
        else:
            candidate_face = candidate_face[0]
            score_face = score_face[0]

        # body
        if score_body.shape[0] > 1:
            tmp_score = []
            for k in range(0, score_body.shape[0]):
                tmp_score.append(score_body[k].mean())
            index = np.argmax(tmp_score)
            candidate_body = candidate_body[index * 18 : (index + 1) * 18, :]
            score_body = score_body[index]
            score_hand = score_hand[(index * 2) : (index * 2 + 2), :]
            candidate_hand = candidate_hand[(index * 2) : (index * 2 + 2), :, :]
        else:
            score_body = score_body[0]
        all_pose = np.concatenate((candidate_body, candidate_face))
        all_score = np.concatenate((score_body, score_face))
        all_pose = all_pose[all_score > 0.8]

        body_pose = np.concatenate((candidate_body,))
        mid_ = body_pose[1, 0]

        face_pose = candidate_face
        hand_pose = candidate_hand

        h_min, h_max = np.min(face_pose[:, 1]), np.max(body_pose[:7, 1])

        h_ = h_max - h_min

        mid_w = mid_
        w_min = mid_w - h_ // 2
        w_max = mid_w + h_ // 2

        w_min_all.append(w_min)
        w_max_all.append(w_max)
        h_min_all.append(h_min)
        h_max_all.append(h_max)
        mid_all.append(mid_w)

    w_min = np.min(w_min_all)
    w_max = np.max(w_max_all)
    h_min = np.min(h_min_all)
    h_max = np.max(h_max_all)
    mid = np.mean(mid_all)

    margin_ratio = 0.25
    h_margin = (h_max - h_min) * margin_ratio

    h_min = max(h_min - h_margin * 0.8, 0)
    h_max = min(h_max + h_margin * 0.1, 1)

    h_new = h_max - h_min

    h_min_real = int(h_min * height)
    h_max_real = int(h_max * height)
    mid_real = int(mid * width)

    height_new = h_max_real - h_min_real + 1
    width_new = height_new
    w_min_real = mid_real - width_new // 2
    if w_min_real < 0:
        w_min_real = 0
        width_new = mid_real * 2

    w_max_real = w_min_real + width_new
    w_min = w_min_real / width
    w_max = w_max_real / width

    imh_new, imw_new, rb, re, cb, ce = resize_and_pad_param(
        height_new, width_new, max_size
    )
    res = {
        "draw_pose_params": [imh_new, imw_new, rb, re, cb, ce],
        "pose_params": [w_min, w_max, h_min, h_max],
        "video_params": [h_min_real, h_max_real, w_min_real, w_max_real],
    }
    return res


def save_pose_params_item(input_items):
    detected_pose, pose_params, draw_pose_params, save_dir = input_items
    w_min, w_max, h_min, h_max = pose_params
    num = detected_pose["num"]
    candidate_body = detected_pose["bodies"]["candidate"]
    candidate_face = detected_pose["faces"][0]
    candidate_hand = detected_pose["hands"]
    candidate_body[:, 0] = (candidate_body[:, 0] - w_min) / (w_max - w_min)
    candidate_body[:, 1] = (candidate_body[:, 1] - h_min) / (h_max - h_min)
    candidate_face[:, 0] = (candidate_face[:, 0] - w_min) / (w_max - w_min)
    candidate_face[:, 1] = (candidate_face[:, 1] - h_min) / (h_max - h_min)
    candidate_hand[:, :, 0] = (candidate_hand[:, :, 0] - w_min) / (w_max - w_min)
    candidate_hand[:, :, 1] = (candidate_hand[:, :, 1] - h_min) / (h_max - h_min)
    detected_pose["bodies"]["candidate"] = candidate_body
    detected_pose["faces"] = candidate_face.reshape(
        1, candidate_face.shape[0], candidate_face.shape[1]
    )
    detected_pose["hands"] = candidate_hand
    detected_pose["draw_pose_params"] = draw_pose_params
    np.save(save_dir + "/" + str(num) + ".npy", detected_pose)


def save_pose_params(detected_poses, pose_params, draw_pose_params, ori_video_path):
    save_dir = ori_video_path.replace("video", "pose/")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    input_list = []

    for i, detected_pose in enumerate(detected_poses):
        input_list.append([detected_pose, pose_params, draw_pose_params, save_dir])

    pool = ThreadPool(8)
    pool.map(save_pose_params_item, input_list)
    pool.close()
    pool.join()
    return save_dir


def get_img_pose(img_path: str, sample_stride: int = 1, max_frame=None):
    # read input img
    frame = cv2.imread(img_path)
    height, width, _ = frame.shape
    short_size = min(height, width)
    resize_ratio = max(MAX_SIZE / short_size, 1.0)
    frame = cv2.resize(frame, (int(resize_ratio * width), int(resize_ratio * height)))
    height, width, _ = frame.shape
    detected_poses = [dwprocessor(frame)]
    dwprocessor.release_memory()

    return detected_poses, height, width, frame


def save_aligned_img(ori_frame, video_params, max_size):
    h_min_real, h_max_real, w_min_real, w_max_real = video_params
    img = ori_frame[h_min_real:h_max_real, w_min_real:w_max_real, :]
    img_aligened = resize_and_pad(img, max_size=max_size)
    print("aligned img shape:", img_aligened.shape)
    save_dir = "./assets/refimg_aligned"

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "aligned.png")
    cv2.imwrite(save_path, img_aligened)
    return save_path


start = time.time()
detected_poses, height, width, ori_frame = get_img_pose(refimg_path, max_frame=None)
end = time.time()
print(f"get_img_pose: {end-start}")
start = time.time()
res_params = get_pose_params(detected_poses, MAX_SIZE)
end = time.time()
print(f"get_pose_params: {end-start}")
start = time.time()
refimg_aligned_path = save_aligned_img(ori_frame, res_params["video_params"], MAX_SIZE)
end = time.time()
print(f"save_aligned_img: {end-start}")

ffmpeg_path = os.getenv("FFMPEG_PATH")

if ffmpeg_path is None:
    print(
        "please download ffmpeg-static and export to FFMPEG_PATH. \nFor example: export FFMPEG_PATH=./ffmpeg-4.4-amd64-static"
    )
elif ffmpeg_path not in os.getenv("PATH"):
    print("add ffmpeg to path")
    os.environ["PATH"] = f"{ffmpeg_path}:{os.environ['PATH']}"


start = time.time()
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="./configs/prompts/infer.yaml")
parser.add_argument("-W", type=int, default=768)
parser.add_argument("-H", type=int, default=768)
parser.add_argument("-L", type=int, default=240)
parser.add_argument("--seed", type=int, default=3407)

parser.add_argument("--context_frames", type=int, default=12)
parser.add_argument("--context_overlap", type=int, default=3)

parser.add_argument("--cfg", type=float, default=2.5)
parser.add_argument("--steps", type=int, default=30)
parser.add_argument("--sample_rate", type=int, default=16000)
parser.add_argument("--fps", type=int, default=24)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument(
    "--ref_images_dir", type=str, default="./assets/halfbody_demo/refimag"
)
parser.add_argument("--pose_dir", type=str, default=None)
parser.add_argument("--refimg_name", type=str, default="natural_bk_openhand/0035.png")
parser.add_argument("--pose_name", type=str, default="01")
parser.add_argument("--video_dir", type=str, default="./assets/halfbody_demo/video")

args, _ = parser.parse_known_args()
end = time.time()
print(f"parse_known_args: {end-start}")


start = time.time()
config = OmegaConf.load(args.config)
if config.weight_dtype == "fp16":
    weight_dtype = torch.float16
else:
    weight_dtype = torch.float32

device = args.device
if device.__contains__("cuda") and not torch.cuda.is_available():
    device = "cpu"

inference_config_path = config.inference_config
infer_config = OmegaConf.load(inference_config_path)

model_flag = "{}-iter{}".format(
    config.motion_module_path.split("/")[-2],
    config.motion_module_path.split("/")[-1].split("-")[-1][:-4],
)
save_dir = Path(f"outputs/{model_flag}-seed{args.seed}/")
save_dir.mkdir(exist_ok=True, parents=True)
print(save_dir)

end = time.time()
print(f"load_config_save_seed: {end-start}")

start = time.time()
############# model_init started #############
## vae init
vae = AutoencoderKL.from_pretrained(
    config.pretrained_vae_path,
).to(device, dtype=weight_dtype)

## reference net init
reference_unet = UNet2DConditionModel.from_pretrained(
    config.pretrained_base_model_path,
    subfolder="unet",
).to(dtype=weight_dtype, device=device)
reference_unet.load_state_dict(
    torch.load(config.reference_unet_path, map_location="cpu"),
)

## denoising net init
if os.path.exists(config.motion_module_path):
    print("using motion module")
else:
    exit("motion module not found")
    ### stage1 + stage2
denoising_unet = EMOUNet3DConditionModel.from_pretrained_2d(
    config.pretrained_base_model_path,
    config.motion_module_path,
    subfolder="unet",
    unet_additional_kwargs=infer_config.unet_additional_kwargs,
).to(dtype=weight_dtype, device=device)

denoising_unet.load_state_dict(
    torch.load(config.denoising_unet_path, map_location="cpu"), strict=False
)

# pose net init
pose_net = PoseEncoder(
    320, conditioning_channels=3, block_out_channels=(16, 32, 96, 256)
).to(dtype=weight_dtype, device=device)
pose_net.load_state_dict(torch.load(config.pose_encoder_path))

### load audio processor params
audio_processor = load_audio_model(model_path=config.audio_model_path, device=device)

end = time.time()
print(f"load_module: {end-start}")
############# model_init finished #############
width, height = 768, 768  # fixed size
sched_kwargs = OmegaConf.to_container(infer_config.noise_scheduler_kwargs)
scheduler = DDIMScheduler(**sched_kwargs)

pipe = EchoMimicV2Pipeline(
    vae=vae,
    reference_unet=reference_unet,
    denoising_unet=denoising_unet,
    audio_guider=audio_processor,
    pose_encoder=pose_net,
    scheduler=scheduler,
)

start = time.time()
pipe = pipe.to(device, dtype=weight_dtype)

end = time.time()
print(f"pipe: {end-start}")

if args.seed is not None and args.seed > -1:
    generator = torch.manual_seed(args.seed)
else:
    generator = torch.manual_seed(random.randint(100, 1000000))

final_fps = args.fps

inputs_dict = {
    "refimg": refimg_aligned_path,
    "audio": audio_path,
    "pose": pose_path,
}

start_idx = 0

print("Pose:", inputs_dict["pose"])
print("Reference:", inputs_dict["refimg"])
print("Audio:", inputs_dict["audio"])
audio_name = inputs_dict["audio"].split("/")[-1].split(".")[0]

ref_flag = ".".join(
    [inputs_dict["refimg"].split("/")[-2], inputs_dict["refimg"].split("/")[-1]]
)
save_path = Path("outputs")

save_path.mkdir(exist_ok=True, parents=True)
ref_s = inputs_dict["refimg"].split("/")[-1].split(".")[0]
save_name = f"{save_path}/{ref_s}-a-{audio_name}-i{start_idx}"

ref_image_pil = Image.open(inputs_dict["refimg"]).resize((args.W, args.H))
audio_clip = AudioFileClip(inputs_dict["audio"])

args.L = min(int(audio_clip.duration * final_fps), len(os.listdir(inputs_dict["pose"])))

pose_list = []
for index in range(start_idx, start_idx + args.L):
    tgt_musk = np.zeros((args.W, args.H, 3)).astype("uint8")
    tgt_musk_path = os.path.join(inputs_dict["pose"], "{}.npy".format(index))
    detected_pose = np.load(tgt_musk_path, allow_pickle=True).tolist()
    imh_new, imw_new, rb, re, cb, ce = detected_pose["draw_pose_params"]
    im = draw_pose_select_v2(detected_pose, imh_new, imw_new, ref_w=800)
    im = np.transpose(np.array(im), (1, 2, 0))
    tgt_musk[rb:re, cb:ce, :] = im

    tgt_musk_pil = Image.fromarray(np.array(tgt_musk)).convert("RGB")
    pose_list.append(
        torch.Tensor(np.array(tgt_musk_pil))
        .to(dtype=weight_dtype, device=device)
        .permute(2, 0, 1)
        / 255.0
    )

poses_tensor = torch.stack(pose_list, dim=1).unsqueeze(0)
audio_clip = AudioFileClip(inputs_dict["audio"])
width, height = 768, 768
audio_clip = audio_clip.set_duration(args.L / final_fps)
start = time.time()
video = pipe(
    ref_image_pil,
    inputs_dict["audio"],
    poses_tensor[:, :, : args.L, ...],
    width,
    height,
    args.L,
    args.steps,
    args.cfg,
    generator=generator,
    audio_sample_rate=args.sample_rate,
    context_frames=args.context_frames,
    fps=final_fps,
    context_overlap=args.context_overlap,
    start_idx=start_idx,
).videos
end = time.time()
print(f"-----------video-----------: {end-start}")

final_length = min(video.shape[2], poses_tensor.shape[2], args.L)
video_sig = video[:, :, :final_length, :, :]

save_videos_grid(
    video_sig,
    save_name + "_woa_sig.mp4",
    n_rows=1,
    fps=final_fps,
)

video_clip_sig = VideoFileClip(
    save_name + "_woa_sig.mp4",
)
video_clip_sig = video_clip_sig.set_audio(audio_clip)
video_clip_sig.write_videofile(
    save_name + "_sig.mp4", codec="libx264", audio_codec="aac", threads=2
)
os.system("rm {}".format(save_name + "_woa_sig.mp4"))
print(save_name)
