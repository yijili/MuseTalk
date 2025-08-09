import os
import sys
import time
import argparse
import logging
import uuid
from typing import Optional
from datetime import datetime
import asyncio
import threading
import signal
import atexit
import gc
from contextlib import asynccontextmanager

import torch
import numpy as np
import cv2
import imageio
from moviepy.editor import VideoFileClip, AudioFileClip
import glob
import pickle
import shutil
import re
from argparse import Namespace
from tqdm import tqdm
import copy

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置北京时间
import time
import logging.handlers

class BeijingTimeFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        ct = time.localtime(record.created)
        # 转换为北京时间 (UTC+8)
        beijing_time = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(record.created + 8 * 3600))
        return beijing_time

# 重新设置日志格式，使用北京时间
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
    
# 创建新的处理器并设置北京时间格式
handler = logging.StreamHandler()
formatter = BeijingTimeFormatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# 全局变量
device = None
weight_dtype = torch.float32
timesteps = None

# 延迟加载模型，避免在模块导入时加载
vae, unet, pe, audio_processor, whisper = None, None, None, None, None

# 从 app.py 中导入必要的函数和模块
from musetalk.utils.blending import get_image
from musetalk.utils.face_parsing import FaceParsing
from musetalk.utils.audio_processor import AudioProcessor
from musetalk.utils.utils import get_file_type, get_video_fps, datagen, load_all_model
from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs, coord_placeholder, get_bbox_range
from transformers import WhisperModel

# 服务相关全局变量
server_thread = None
should_exit = False

# 任务管理相关
active_tasks = {}  # 存储正在进行的任务
task_lock = threading.Lock()  # 任务锁

class TaskStatus:
    def __init__(self, task_id: str):
        self.task_id = task_id
        self.status = "running"  # running, completed, failed, cancelled
        self.progress = 0
        self.message = ""
        self.result = None
        self.thread = None  # 添加线程引用

def cleanup_models():
    """清理模型以释放内存"""
    global vae, unet, pe, audio_processor, whisper
    logger.info("清理模型内存...")
    
    # 删除模型引用
    if vae:
        del vae
    if unet:
        del unet
    if pe:
        del pe
    if audio_processor:
        del audio_processor
    if whisper:
        del whisper
        
    vae = None
    unet = None
    pe = None
    audio_processor = None
    whisper = None
    
    # 清理CUDA缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    logger.info("模型内存清理完成")

def load_models():
    """加载所有模型"""
    global vae, unet, pe, audio_processor, whisper, device, timesteps
    
    # 使用 GPU 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    timesteps = torch.tensor([0], device=device)
    
    logger.info(f"使用设备: {device}")
    logger.info("开始加载模型...")
    
    # 加载 MuseTalk 模型
    vae, unet, pe = load_all_model(
        unet_model_path="./models/musetalkV15/unet.pth", 
        vae_type="sd-vae",
        unet_config="./models/musetalkV15/musetalk.json",
        device=device
    )
    
    # 移动模型到指定设备
    pe = pe.to(device)
    vae.vae = vae.vae.to(device)
    unet.model = unet.model.to(device)
    
    # 初始化音频处理器和 Whisper 模型
    audio_processor = AudioProcessor(feature_extractor_path="./models/whisper")
    whisper = WhisperModel.from_pretrained("./models/whisper")
    whisper = whisper.to(device=device, dtype=weight_dtype).eval()
    whisper.requires_grad_(False)
    
    logger.info("模型加载完成")
    if torch.cuda.is_available():
        logger.info(f"GPU内存状态: 已分配 {torch.cuda.memory_allocated()/1024**2:.1f} MB, 已缓存 {torch.cuda.memory_reserved()/1024**2:.1f} MB")

# 注册退出处理函数
atexit.register(cleanup_models)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时的初始化代码
    try:
        load_models()
    except Exception as e:
        logger.error(f"模型加载失败: {str(e)}")
        raise
    yield
    # 关闭时的清理代码
    logger.info("正在关闭应用...")
    cleanup_models()
    logger.info("应用已关闭")

app = FastAPI(title="MuseTalk API", description="MuseTalk FastAPI 服务，提供 debug_inpainting 和 inference 接口", lifespan=lifespan)

class DebugInpaintingRequest(BaseModel):
    video_path: str
    bbox_shift: int = 0
    extra_margin: int = 10
    parsing_mode: str = "jaw"
    left_cheek_width: int = 90
    right_cheek_width: int = 90
    output_path: Optional[str] = None

class InferenceRequest(BaseModel):
    audio_path: str
    video_path: str
    bbox_shift: int = 0
    extra_margin: int = 10
    parsing_mode: str = "jaw"
    left_cheek_width: int = 90
    right_cheek_width: int = 90
    output_path: Optional[str] = None
    # 添加一个可选的任务ID，用于支持取消操作
    task_id: Optional[str] = None

class InferenceResponse(BaseModel):
    task_id: str
    message: str
    video_path: Optional[str] = None
    bbox_shift_text: Optional[str] = None

class TaskStatusRequest(BaseModel):
    task_id: str

class TaskResultRequest(BaseModel):
    task_id: str

class CancelTaskRequest(BaseModel):
    task_id: str

def generate_unique_filename(extension: str) -> str:
    """生成带时间戳的唯一文件名"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    return f"{timestamp}_{unique_id}.{extension}"

def check_task_cancelled(task_id: str) -> bool:
    """检查任务是否被取消"""
    with task_lock:
        if task_id in active_tasks:
            return active_tasks[task_id].status == "cancelled"
    return False

@torch.no_grad()
def debug_inpainting_api(
    video_path: str,
    bbox_shift: int,
    extra_margin: int = 10,
    parsing_mode: str = "jaw",
    left_cheek_width: int = 90,
    right_cheek_width: int = 90,
    output_path: Optional[str] = None
) -> dict:
    """API 版本的 debug_inpainting 函数"""
    try:
        logger.info(f"开始执行 debug_inpainting，视频路径: {video_path}")
        
        # 确保模型已加载
        if vae is None or unet is None or pe is None:
            raise HTTPException(status_code=500, detail="模型未加载")
        
        # 设置参数
        result_dir = output_path if output_path else './results/debug'
        args_dict = {
            "result_dir": result_dir, 
            "fps": 25, 
            "batch_size": 1, 
            "output_vid_name": '', 
            "use_saved_coord": False,
            "audio_padding_length_left": 2,
            "audio_padding_length_right": 2,
            "version": "v15",
            "extra_margin": extra_margin,
            "parsing_mode": parsing_mode,
            "left_cheek_width": left_cheek_width,
            "right_cheek_width": right_cheek_width
        }
        args = Namespace(**args_dict)

        # 创建 debug 目录
        os.makedirs(args.result_dir, exist_ok=True)
        
        # 读取第一帧
        if get_file_type(video_path) == "video":
            logger.info("读取视频第一帧")
            reader = imageio.get_reader(video_path)
            first_frame = reader.get_data(0)
            reader.close()
        else:
            logger.info("读取图片")
            first_frame = cv2.imread(video_path)
            first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
        
        # 保存第一帧
        debug_frame_path = os.path.join(args.result_dir, generate_unique_filename("png"))
        cv2.imwrite(debug_frame_path, cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR))
        logger.info(f"第一帧已保存到: {debug_frame_path}")
        
        # 获取面部坐标
        logger.info("获取面部坐标...")
        coord_list, frame_list = get_landmark_and_bbox([debug_frame_path], bbox_shift)
        bbox = coord_list[0]
        frame = frame_list[0]
        
        if bbox == coord_placeholder:
            error_msg = "未检测到面部，请调整 bbox_shift 参数"
            logger.error(error_msg)
            raise HTTPException(status_code=400, detail=error_msg)
        
        # 初始化面部解析器
        fp = FaceParsing(
            left_cheek_width=args.left_cheek_width,
            right_cheek_width=args.right_cheek_width
        )
        
        # 处理第一帧
        x1, y1, x2, y2 = bbox
        y2 = y2 + args.extra_margin
        y2 = min(y2, frame.shape[0])
        crop_frame = frame[y1:y2, x1:x2]
        crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
        
        # 生成随机音频特征
        random_audio = torch.randn(1, 50, 384, device=device, dtype=weight_dtype)
        audio_feature = pe(random_audio)
        
        # 获取潜在变量
        latents = vae.get_latents_for_unet(crop_frame)
        latents = latents.to(dtype=weight_dtype)
        
        # 生成预测结果
        pred_latents = unet.model(latents, timesteps, encoder_hidden_states=audio_feature).sample
        recon = vae.decode_latents(pred_latents)
        
        # 修复回原始图像
        res_frame = recon[0]
        res_frame = cv2.resize(res_frame.astype(np.uint8), (x2-x1, y2-y1))
        combine_frame = get_image(frame, res_frame, [x1, y1, x2, y2], mode=args.parsing_mode, fp=fp)
        
        # 保存结果
        result_filename = generate_unique_filename("png")
        debug_result_path = os.path.join(args.result_dir, result_filename)
        cv2.imwrite(debug_result_path, combine_frame)
        logger.info(f"结果已保存到: {debug_result_path}")
        
        # 创建信息文本
        info_text = f"参数信息:\n" + \
                    f"bbox_shift: {bbox_shift}\n" + \
                    f"extra_margin: {extra_margin}\n" + \
                    f"parsing_mode: {parsing_mode}\n" + \
                    f"left_cheek_width: {left_cheek_width}\n" + \
                    f"right_cheek_width: {right_cheek_width}\n" + \
                    f"检测到的面部坐标: [{x1}, {y1}, {x2}, {y2}]"
        
        logger.info("debug_inpainting 执行完成")
        return {
            "image_path": os.path.abspath(debug_result_path),
            "info_text": info_text
        }
        
    except Exception as e:
        logger.error(f"debug_inpainting 执行出错: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # 强制进行垃圾回收
        gc.collect()

@torch.no_grad()
def inference_api(
    task_id: str,
    audio_path: str,
    video_path: str,
    bbox_shift: int,
    extra_margin: int = 10,
    parsing_mode: str = "jaw",
    left_cheek_width: int = 90,
    right_cheek_width: int = 90,
    output_path: Optional[str] = None
) -> dict:
    """API 版本的 inference 函数，支持任务中断"""
    try:
        logger.info(f"开始执行 inference，任务ID: {task_id}，音频路径: {audio_path}，视频路径: {video_path}")
        
        # 确保模型已加载
        if vae is None or unet is None or pe is None or audio_processor is None or whisper is None:
            raise HTTPException(status_code=500, detail="模型未加载")
        
        # 设置参数
        result_dir = output_path if output_path else './results/output'
        args_dict = {
            "result_dir": result_dir, 
            "fps": 25, 
            "batch_size": 8, 
            "output_vid_name": '', 
            "use_saved_coord": False,
            "audio_padding_length_left": 2,
            "audio_padding_length_right": 2,
            "version": "v15",
            "extra_margin": extra_margin,
            "parsing_mode": parsing_mode,
            "left_cheek_width": left_cheek_width,
            "right_cheek_width": right_cheek_width
        }
        args = Namespace(**args_dict)

        input_basename = os.path.basename(video_path).split('.')[0]
        audio_basename = os.path.basename(audio_path).split('.')[0]
        output_basename = f"{input_basename}_{audio_basename}_{int(time.time())}"
        
        # 检查任务是否已取消
        if check_task_cancelled(task_id):
            logger.info(f"任务 {task_id} 已被取消")
            return {"message": "任务已被取消"}
        
        # 创建临时目录
        temp_dir = os.path.join(args.result_dir, f"{args.version}")
        os.makedirs(temp_dir, exist_ok=True)
        
        # 设置结果保存路径
        result_img_save_path = os.path.join(temp_dir, output_basename)
        crop_coord_save_path = os.path.join(args.result_dir, "../", input_basename+".pkl")
        os.makedirs(result_img_save_path, exist_ok=True)

        if args.output_vid_name == "":
            output_vid_name = os.path.join(temp_dir, generate_unique_filename("mp4"))
        else:
            output_vid_name = os.path.join(temp_dir, args.output_vid_name)
            
        # 更新任务状态
        with task_lock:
            if task_id in active_tasks:
                active_tasks[task_id].progress = 5
                active_tasks[task_id].message = "提取视频帧..."
            
        ############################################## 从源视频提取帧 ##############################################
        logger.info("提取视频帧...")
        if get_file_type(video_path) == "video":
            save_dir_full = os.path.join(temp_dir, input_basename+"_"+str(int(time.time())))
            os.makedirs(save_dir_full, exist_ok=True)
            # 读取视频
            reader = imageio.get_reader(video_path)

            # 保存图像
            for i, im in enumerate(reader):
                # 检查任务是否已取消
                if check_task_cancelled(task_id):
                    logger.info(f"任务 {task_id} 已被取消")
                    return {"message": "任务已被取消"}
                imageio.imwrite(f"{save_dir_full}/{i:08d}.png", im)
            input_img_list = sorted(glob.glob(os.path.join(save_dir_full, '*.[jpJP][pnPN]*[gG]')))
            fps = get_video_fps(video_path)
        else: # 输入图片文件夹
            input_img_list = glob.glob(os.path.join(video_path, '*.[jpJP][pnPN]*[gG]'))
            input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
            fps = args.fps
            
        # 检查任务是否已取消
        if check_task_cancelled(task_id):
            logger.info(f"任务 {task_id} 已被取消")
            return {"message": "任务已被取消"}
            
        # 更新任务状态
        with task_lock:
            if task_id in active_tasks:
                active_tasks[task_id].progress = 15
                active_tasks[task_id].message = "提取音频特征..."
            
        ############################################## 提取音频特征 ##############################################
        logger.info("提取音频特征...")
        # 提取音频特征
        whisper_input_features, librosa_length = audio_processor.get_audio_feature(audio_path)
        whisper_chunks = audio_processor.get_whisper_chunk(
            whisper_input_features, 
            device, 
            weight_dtype, 
            whisper, 
            librosa_length,
            fps=fps,
            audio_padding_length_left=args.audio_padding_length_left,
            audio_padding_length_right=args.audio_padding_length_right,
        )
            
        # 检查任务是否已取消
        if check_task_cancelled(task_id):
            logger.info(f"任务 {task_id} 已被取消")
            return {"message": "任务已被取消"}
            
        # 更新任务状态
        with task_lock:
            if task_id in active_tasks:
                active_tasks[task_id].progress = 25
                active_tasks[task_id].message = "预处理输入图像..."
            
        ############################################## 预处理输入图像 ##############################################
        logger.info("预处理输入图像...")
        if os.path.exists(crop_coord_save_path) and args.use_saved_coord:
            logger.info("使用已提取的坐标")
            with open(crop_coord_save_path,'rb') as f:
                coord_list = pickle.load(f)
            frame_list = read_imgs(input_img_list)
        else:
            logger.info("提取关键点...这可能需要一些时间")
            # 在提取关键点的过程中也检查任务是否被取消
            coord_list = []
            frame_list = []
            
            # 分批处理图像以减少内存占用并允许取消检查
            batch_size = 10  # 每批处理10张图片
            total_images = len(input_img_list)
            
            for i in range(0, total_images, batch_size):
                # 检查任务是否已取消
                if check_task_cancelled(task_id):
                    logger.info(f"任务 {task_id} 已被取消")
                    return {"message": "任务已被取消"}
                
                # 处理一批图像
                batch_end = min(i + batch_size, total_images)
                batch_img_paths = input_img_list[i:batch_end]
                
                # 处理这批图像
                batch_coords, batch_frames = get_landmark_and_bbox(batch_img_paths, bbox_shift)
                coord_list.extend(batch_coords)
                frame_list.extend(batch_frames)
                
                # 更新进度（每批更新一次，避免过多日志）
                progress = 25 + int(10 * batch_end / total_images)
                with task_lock:
                    if task_id in active_tasks:
                        active_tasks[task_id].progress = progress
                        active_tasks[task_id].message = f"提取关键点... ({batch_end}/{total_images})"
                
                logger.info(f"提取关键点进度: {batch_end}/{total_images}")
            
            # 保存坐标
            with open(crop_coord_save_path, 'wb') as f:
                pickle.dump(coord_list, f)
        
        bbox_shift_text = get_bbox_range(input_img_list, bbox_shift)
        
        # 初始化面部解析器
        fp = FaceParsing(
            left_cheek_width=args.left_cheek_width,
            right_cheek_width=args.right_cheek_width
        )
        
        input_latent_list = []
        total_coords = len(coord_list)
        
        for i, (bbox, frame) in enumerate(zip(coord_list, frame_list)):
            # 检查任务是否已取消
            if check_task_cancelled(task_id):
                logger.info(f"任务 {task_id} 已被取消")
                return {"message": "任务已被取消"}
            if bbox == coord_placeholder:
                continue
            x1, y1, x2, y2 = bbox
            y2 = y2 + args.extra_margin
            y2 = min(y2, frame.shape[0])
            crop_frame = frame[y1:y2, x1:x2]
            crop_frame = cv2.resize(crop_frame,(256,256),interpolation = cv2.INTER_LANCZOS4)
            latents = vae.get_latents_for_unet(crop_frame)
            input_latent_list.append(latents)
            
            # 更新进度（每处理10个坐标更新一次，避免过多日志）
            if (i + 1) % 10 == 0 or (i + 1) == total_coords:
                progress = 35 + int(5 * (i + 1) / total_coords)
                with task_lock:
                    if task_id in active_tasks:
                        active_tasks[task_id].progress = progress
                        active_tasks[task_id].message = f"预处理图像... ({i + 1}/{total_coords})"

        # 平滑第一帧和最后一帧
        frame_list_cycle = frame_list + frame_list[::-1]
        coord_list_cycle = coord_list + coord_list[::-1]
        input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
        
        # 检查任务是否已取消
        if check_task_cancelled(task_id):
            logger.info(f"任务 {task_id} 已被取消")
            return {"message": "任务已被取消"}
            
        # 更新任务状态
        with task_lock:
            if task_id in active_tasks:
                active_tasks[task_id].progress = 40
                active_tasks[task_id].message = "开始推理..."
        
        ############################################## 分批推理 ##############################################
        logger.info("开始推理...")
        video_num = len(whisper_chunks)
        batch_size = args.batch_size
        gen = datagen(
            whisper_chunks=whisper_chunks,
            vae_encode_latents=input_latent_list_cycle,
            batch_size=batch_size,
            delay_frame=0,
            device=device,
        )
        res_frame_list = []
        batch_count = 0
        total_batches = int(np.ceil(float(video_num)/batch_size))
        
        # 使用不显示进度条的迭代器以减少日志输出
        for i, (whisper_batch,latent_batch) in enumerate(gen):
            # 检查任务是否已取消
            if check_task_cancelled(task_id):
                logger.info(f"任务 {task_id} 已被取消")
                return {"message": "任务已被取消"}
                
            audio_feature_batch = pe(whisper_batch)
            # 确保 latent_batch 与模型权重类型一致
            latent_batch = latent_batch.to(dtype=weight_dtype)
            
            pred_latents = unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feature_batch).sample
            recon = vae.decode_latents(pred_latents)
            for res_frame in recon:
                res_frame_list.append(res_frame)
                
            # 更新进度（每10个批次更新一次）
            batch_count += 1
            if batch_count % 10 == 0 or batch_count == total_batches:
                # 检查任务是否已取消
                if check_task_cancelled(task_id):
                    logger.info(f"任务 {task_id} 已被取消")
                    return {"message": "任务已被取消"}
                progress = 40 + int(30 * batch_count / total_batches)
                with task_lock:
                    if task_id in active_tasks:
                        active_tasks[task_id].progress = progress
                        active_tasks[task_id].message = f"推理中... ({batch_count}/{total_batches})"
                logger.info(f"推理进度: {batch_count}/{total_batches}")
                
        # 检查任务是否已取消
        if check_task_cancelled(task_id):
            logger.info(f"任务 {task_id} 已被取消")
            return {"message": "任务已被取消"}
            
        # 更新任务状态
        with task_lock:
            if task_id in active_tasks:
                active_tasks[task_id].progress = 70
                active_tasks[task_id].message = "填充到完整图像..."
            
        ############################################## 填充到完整图像 ##############################################
        logger.info("将处理后的图像填充回原始视频")
        frame_count = 0
        total_frames = len(res_frame_list)
        
        # 使用不显示进度条的迭代器以减少日志输出
        for i, res_frame in enumerate(res_frame_list):
            # 检查任务是否已取消
            if check_task_cancelled(task_id):
                logger.info(f"任务 {task_id} 已被取消")
                return {"message": "任务已被取消"}
                
            bbox = coord_list_cycle[i%(len(coord_list_cycle))]
            ori_frame = copy.deepcopy(frame_list_cycle[i%(len(frame_list_cycle))])
            x1, y1, x2, y2 = bbox
            y2 = y2 + args.extra_margin
            y2 = min(y2, ori_frame.shape[0])
            try:
                res_frame = cv2.resize(res_frame.astype(np.uint8),(x2-x1,y2-y1))
            except:
                continue
            
            # 使用 v15 版本混合
            combine_frame = get_image(ori_frame, res_frame, [x1, y1, x2, y2], mode=args.parsing_mode, fp=fp)
                
            cv2.imwrite(f"{result_img_save_path}/{str(i).zfill(8)}.png",combine_frame)
            
            # 更新进度（每50帧更新一次）
            frame_count += 1
            if frame_count % 50 == 0 or frame_count == total_frames:
                # 检查任务是否已取消
                if check_task_cancelled(task_id):
                    logger.info(f"任务 {task_id} 已被取消")
                    return {"message": "任务已被取消"}
                progress = 70 + int(20 * frame_count / total_frames)
                with task_lock:
                    if task_id in active_tasks:
                        active_tasks[task_id].progress = progress
                        active_tasks[task_id].message = f"生成帧... ({frame_count}/{total_frames})"
                logger.info(f"生成帧进度: {frame_count}/{total_frames}")
            
        # 检查任务是否已取消
        if check_task_cancelled(task_id):
            logger.info(f"任务 {task_id} 已被取消")
            return {"message": "任务已被取消"}
            
        # 帧率
        fps = 25
        # 输出视频路径
        output_video = os.path.join(temp_dir, generate_unique_filename("mp4"))

        # 读取图像
        def is_valid_image(file):
            pattern = re.compile(r'\d{8}\.png')
            return pattern.match(file)

        images = []
        files = [file for file in os.listdir(result_img_save_path) if is_valid_image(file)]
        files.sort(key=lambda x: int(x.split('.')[0]))

        for file in files:
            filename = os.path.join(result_img_save_path, file)
            images.append(imageio.imread(filename))
            
        # 更新任务状态
        with task_lock:
            if task_id in active_tasks:
                active_tasks[task_id].progress = 90
                active_tasks[task_id].message = "生成最终视频..."

        # 保存视频
        logger.info("保存视频...")
        imageio.mimwrite(output_video, images, 'FFMPEG', fps=fps, codec='libx264', pixelformat='yuv420p')

        # 检查输入视频和音频路径是否存在
        if not os.path.exists(output_video):
            raise FileNotFoundError(f"输出视频文件未找到: {output_video}")
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"音频文件未找到: {audio_path}")
        
        # 读取视频
        reader = imageio.get_reader(output_video)
        fps = reader.get_meta_data()['fps']  # 获取原始视频帧率
        reader.close()
        # 将帧存储在列表中
        frames = images
        
        logger.info(f"帧数: {len(frames)}")

        # 加载视频
        logger.info("加载视频...")
        video_clip = VideoFileClip(output_video)

        # 加载音频
        logger.info("加载音频...")
        audio_clip = AudioFileClip(audio_path)

        # 将音频设置到视频
        video_clip = video_clip.set_audio(audio_clip)

        # 写入输出视频
        logger.info("写入最终视频...")
        video_clip.write_videofile(output_vid_name, codec='libx264', audio_codec='aac',fps=25, logger=None)

        # 清理临时资源
        del images
        del reader
        del video_clip
        del audio_clip
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        os.remove(output_video)
        # shutil.rmtree(result_img_save_path)
        logger.info(f"结果已保存到: {output_vid_name}")
        
        # 更新任务状态
        with task_lock:
            if task_id in active_tasks:
                active_tasks[task_id].progress = 100
                active_tasks[task_id].status = "completed"
                active_tasks[task_id].message = "任务完成"
                active_tasks[task_id].result = {
                    "video_path": os.path.abspath(output_vid_name),
                    "bbox_shift_text": bbox_shift_text
                }
        
        return {
            "video_path": os.path.abspath(output_vid_name),
            "bbox_shift_text": bbox_shift_text
        }
        
    except Exception as e:
        # 更新任务状态为失败
        with task_lock:
            if task_id in active_tasks:
                active_tasks[task_id].status = "failed"
                active_tasks[task_id].message = f"执行出错: {str(e)}"
        logger.error(f"inference 执行出错: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # 清理临时变量和强制垃圾回收
        try:
            if 'gen' in locals():
                del gen
            if 'whisper_chunks' in locals():
                del whisper_chunks
            if 'input_latent_list_cycle' in locals():
                del input_latent_list_cycle
            if 'coord_list_cycle' in locals():
                del coord_list_cycle
            if 'frame_list_cycle' in locals():
                del frame_list_cycle
            if 'input_latent_list' in locals():
                del input_latent_list
            if 'coord_list' in locals():
                del coord_list
            if 'frame_list' in locals():
                del frame_list
            if 'res_frame_list' in locals():
                del res_frame_list
            if 'fp' in locals():
                del fp
        except:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def run_inference_background(task_id: str, request: InferenceRequest):
    """在后台线程中运行推理任务"""
    try:
        result = inference_api(
            task_id=task_id,
            audio_path=request.audio_path,
            video_path=request.video_path,
            bbox_shift=request.bbox_shift,
            extra_margin=request.extra_margin,
            parsing_mode=request.parsing_mode,
            left_cheek_width=request.left_cheek_width,
            right_cheek_width=request.right_cheek_width,
            output_path=request.output_path
        )
        logger.info(f"任务 {task_id} 推理完成")
        
        # 更新任务状态为完成
        with task_lock:
            if task_id in active_tasks:
                if "message" in result and result["message"] == "任务已被取消":
                    active_tasks[task_id].status = "cancelled"
                    active_tasks[task_id].message = "任务已完成取消"
                else:
                    active_tasks[task_id].status = "completed"
                    active_tasks[task_id].result = result
    except Exception as e:
        # 更新任务状态为失败
        with task_lock:
            if task_id in active_tasks:
                active_tasks[task_id].status = "failed"
                active_tasks[task_id].message = f"执行出错: {str(e)}"
        
        logger.error(f"处理任务 {task_id} 时出错: {str(e)}")
    finally:
        # 任务完成后强制垃圾回收
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

@app.post("/debug_inpainting", summary="调试修复参数")
async def debug_inpainting_endpoint(request: DebugInpaintingRequest):
    """
    调试 inpainting 参数，只处理第一帧
    """
    logger.info("收到 debug_inpainting 请求")
    
    # 检查视频文件是否存在
    if not os.path.exists(request.video_path):
        logger.error(f"视频文件不存在: {request.video_path}")
        raise HTTPException(status_code=404, detail=f"视频文件不存在: {request.video_path}")
    
    try:
        result = debug_inpainting_api(
            video_path=request.video_path,
            bbox_shift=request.bbox_shift,
            extra_margin=request.extra_margin,
            parsing_mode=request.parsing_mode,
            left_cheek_width=request.left_cheek_width,
            right_cheek_width=request.right_cheek_width,
            output_path=request.output_path
        )
        logger.info("debug_inpainting 请求处理完成")
        return result
    except Exception as e:
        logger.error(f"处理 debug_inpainting 请求时出错: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/inference", response_model=InferenceResponse, summary="执行推理生成视频")
async def inference_endpoint(request: InferenceRequest):
    """
    执行完整的推理过程，生成口型同步的视频
    """
    logger.info("收到 inference 请求")
    
    # 检查音频和视频文件是否存在
    if not os.path.exists(request.audio_path):
        logger.error(f"音频文件不存在: {request.audio_path}")
        raise HTTPException(status_code=404, detail=f"音频文件不存在: {request.audio_path}")
    
    if not os.path.exists(request.video_path):
        logger.error(f"视频文件不存在: {request.video_path}")
        raise HTTPException(status_code=404, detail=f"视频文件不存在: {request.video_path}")
    
    # 如果请求中包含task_id，则将其添加到active_tasks中
    task_id = request.task_id
    if not task_id:
        # 如果没有提供task_id，则生成一个新的
        task_id = str(uuid.uuid4())
    
    # 创建任务状态对象（如果需要跟踪进度）
    task_status = TaskStatus(task_id)
    
    # 将任务添加到活动任务列表
    with task_lock:
        active_tasks[task_id] = task_status
    
    # 在后台线程中运行推理任务
    thread = threading.Thread(target=run_inference_background, args=(task_id, request))
    thread.start()
    
    # 保存线程引用以便可能的管理
    with task_lock:
        active_tasks[task_id].thread = thread
    
    logger.info(f"任务 {task_id} 已启动后台推理线程")
    
    # 立即返回响应
    return InferenceResponse(
        task_id=task_id,
        message="任务已启动"
    )

@app.get("/task_status/{task_id}", summary="获取任务状态")
async def get_task_status(task_id: str):
    """
    获取指定任务的当前状态和进度
    """
    with task_lock:
        if task_id in active_tasks:
            task = active_tasks[task_id]
            return {
                "task_id": task_id,
                "status": task.status,
                "progress": task.progress,
                "message": task.message,
                "result": task.result
            }
        else:
            raise HTTPException(status_code=404, detail="任务不存在")

@app.post("/task_result", summary="等待并获取任务结果")
async def get_task_result(request: TaskResultRequest):
    """
    等待任务完成并返回最终结果
    """
    task_id = request.task_id
    logger.info(f"收到获取任务 {task_id} 结果的请求")
    
    # 首先检查任务是否存在
    with task_lock:
        if task_id not in active_tasks:
            raise HTTPException(status_code=404, detail="任务不存在")
    
    # 等待任务完成
    while True:
        with task_lock:
            if task_id in active_tasks:
                task = active_tasks[task_id]
                # 如果任务已完成、失败或被取消，则退出循环
                if task.status in ["completed", "failed", "cancelled"]:
                    break
            else:
                raise HTTPException(status_code=404, detail="任务不存在")
        
        # 等待一段时间再检查
        await asyncio.sleep(1)
    
    # 返回任务结果
    with task_lock:
        task = active_tasks[task_id]
        if task.status == "completed":
            logger.info(f"任务 {task_id} 已完成，返回结果")
            return {
                "task_id": task_id,
                "status": task.status,
                "message": task.message,
                "result": task.result
            }
        elif task.status == "failed":
            logger.info(f"任务 {task_id} 执行失败")
            raise HTTPException(status_code=500, detail=task.message)
        elif task.status == "cancelled":
            logger.info(f"任务 {task_id} 已被取消")
            raise HTTPException(status_code=499, detail="任务已被取消")  # 499表示客户端关闭请求
        else:
            raise HTTPException(status_code=500, detail="未知任务状态")

@app.post("/cancel_task", summary="取消任务")
async def cancel_task(request: CancelTaskRequest):
    """
    取消指定的任务，并等待取消完成
    """
    task_id = request.task_id
    logger.info(f"收到取消任务 {task_id} 的请求")
    
    # 首先检查任务是否存在
    with task_lock:
        if task_id not in active_tasks:
            raise HTTPException(status_code=404, detail="任务不存在")
    
    # 标记任务为取消状态
    with task_lock:
        active_tasks[task_id].status = "cancelled"
        active_tasks[task_id].message = "任务已被取消"
    
    # 等待任务真正完成取消
    while True:
        with task_lock:
            if task_id in active_tasks:
                task = active_tasks[task_id]
                # 如果任务已取消完成，则退出循环
                if task.status == "cancelled" and task.message == "任务已完成取消":
                    break
            else:
                # 任务已被完全清理
                break
        
        # 等待一段时间再检查
        await asyncio.sleep(0.5)
    
    logger.info(f"任务 {task_id} 取消完成")
    return {"message": "任务取消完成"}

@app.get("/", summary="API 根路径")
async def root():
    return {"message": "MuseTalk API 服务正在运行", 
            "endpoints": [
                "/debug_inpainting", 
                "/inference",
                "/task_status/{task_id}", 
                "/task_result",
                "/cancel_task"
            ]}

@app.get("/results/{file_path:path}", summary="获取结果文件")
async def get_result_file(file_path: str):
    """
    提供对结果文件的访问
    """
    file_full_path = os.path.join("./results", file_path)
    if os.path.exists(file_full_path):
        return FileResponse(file_full_path)
    else:
        raise HTTPException(status_code=404, detail="文件未找到")

def signal_handler(signum, frame):
    """信号处理函数"""
    logger.info(f"收到信号 {signum}，正在关闭服务...")
    global should_exit
    should_exit = True
    cleanup_models()
    sys.exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1", help="主机地址")
    parser.add_argument("--port", type=int, default=7862, help="端口号")
    parser.add_argument("--isdebug", action="store_true", help="是否输出调试日志")
    parser.add_argument("--use_cpu", action="store_true", help="是否强制使用CPU")
    
    args = parser.parse_args()
    
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 根据参数决定设备
    if args.use_cpu:
        device = torch.device("cpu")
        logger.info("强制使用CPU运行")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"使用设备: {device}")
    
    # 根据 isdebug 参数设置日志级别
    if args.isdebug:
        logger.setLevel(logging.DEBUG)
        logger.debug("调试模式已启用")
    
    logger.info(f"启动 MuseTalk API 服务，当前版本1.0.29，主机: {args.host}，端口: {args.port}")
    
    # 不使用 reload 参数，避免重复导入
    uvicorn.run("api:app", host=args.host, port=args.port, reload=False)