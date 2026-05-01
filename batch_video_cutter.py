import cv2
import numpy as np
import subprocess
import os
import sys
from pathlib import Path
import shutil
import tempfile


def find_timestamp_in_video(video_path, end_image_path, search_duration=5):
    """
    使用 OpenCV 在视频最后指定秒数范围内查找结束标志图片。
    返回找到的时间点（秒），如果未找到则返回 None。
    """
    print(f"🔍 正在分析视频画面: {os.path.basename(video_path)}")

    if not os.path.exists(video_path):
        print(f"❌ 错误：视频文件不存在: {video_path}")
        return None

    if not os.path.exists(end_image_path):
        print(f"❌ 错误：结束标志图片不存在: {end_image_path}")
        return None

    # 加载结束标志图片
    end_img = cv2.imread(end_image_path)
    if end_img is None:
        print(f"❌ 错误：无法加载结束标志图片，请检查文件格式")
        return None

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0 or total_frames <= 0:
        print("❌ 错误：无法获取有效的视频帧率或总帧数")
        cap.release()
        return None

    duration = total_frames / fps
    print(f"ℹ️ 视频信息: {duration:.2f}秒, {fps:.2f} FPS, {width}x{height}")

    # 计算搜索范围
    search_start_frame = int(max(0, duration - search_duration) * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, search_start_frame)

    # 调整模板图片大小以匹配视频分辨率（提高匹配成功率）
    resized_template = cv2.resize(end_img, (width, height))

    target_time = None
    frame_idx = search_start_frame
    threshold = 0.85  # 匹配阈值

    print(f"⏳ 开始从第 {search_start_frame} 帧 ({search_start_frame / fps:.2f}s) 进行图像匹配...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 使用模板匹配
        result = cv2.matchTemplate(frame, resized_template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)

        if max_val >= threshold:
            target_time = frame_idx / fps
            print(f"✅ 匹配成功! 在第 {frame_idx} 帧 ({target_time:.2f}s), 相似度: {max_val:.2f}")
            break

        frame_idx += 1
        # 简单的进度显示
        if frame_idx % (fps * 2) == 0:
            print(".", end="", flush=True)

    cap.release()
    print()

    if target_time is None:
        print("⚠️ 未在指定范围内找到结束标志。")

    return target_time


def check_gpu_availability():
    """
    检查 NVIDIA GPU 是否可用
    """
    try:
        # 尝试运行一个简单的 ffmpeg 命令来检查 nvenc 支持
        result = subprocess.run(
            ['ffmpeg', '-hwaccels'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if 'cuda' in result.stdout or 'nvenc' in result.stdout:
            return True
        return False
    except:
        return False


def convert_and_cut_ffmpeg(input_path, output_path, cut_time):
    """
    使用 FFmpeg 将视频转码为 H.264 并在指定时间裁剪，同时彻底清洗元信息。
    优先使用 GPU 加速。
    """

    # 1. 检查 GPU 是否可用
    use_gpu = check_gpu_availability()

    if use_gpu:
        print(f"🚀 检测到 NVIDIA GPU，正在启用硬件加速 (h264_nvenc)...")
        # GPU 编码参数
        # -cq 21: 相当于 CPU 模式的 CRF 23，数值越小画质越好，文件越大
        # -preset p1: NVENC 的最快速度预设 (p1 到 p7)
        cmd = [
            'ffmpeg',
            '-i', input_path,
            '-to', str(cut_time),
            '-c:v', 'h264_nvenc',
            '-preset', 'p1',
            '-cq', '21',
            '-c:a', 'aac',
            '-b:a', '128k',
            '-avoid_negative_ts', '1',
            '-map_metadata', '-1',
            '-movflags', '+faststart',
            '-y',
            output_path
        ]
    else:
        print(f"⚠️ 未检测到 GPU 支持，回退到 CPU 编码 (libx264)...")
        # CPU 编码参数 (备用方案)
        cmd = [
            'ffmpeg',
            '-i', input_path,
            '-to', str(cut_time),
            '-c:v', 'libx264',
            '-preset', 'veryfast',
            '-crf', '23',
            '-c:a', 'aac',
            '-b:a', '128k',
            '-avoid_negative_ts', '1',
            '-map_metadata', '-1',
            '-movflags', '+faststart',
            '-y',
            output_path
        ]

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='ignore',
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        )

        if result.returncode == 0:
            print(f"🎉 处理完成: {output_path}")
            return True
        else:
            # 如果是 GPU 模式失败，可能是显存不足或其他问题，可以尝试提示用户
            if use_gpu and "nvenc" in result.stderr.lower():
                print(f"❌ GPU 编码失败，建议检查显卡驱动或显存。错误信息: {result.stderr[:200]}...")
            else:
                print(f"❌ FFmpeg 错误: {result.stderr}")
            return False

    except Exception as e:
        print(f"❌ 执行异常: {e}")
        return False


def batch_process(input_dir, output_dir, end_image_path, search_duration=10):
    """
    主批处理函数
    """
    # 检查目录
    if not os.path.exists(input_dir):
        print(f"❌ 输入目录不存在: {input_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)

    # 支持的视频扩展名
    extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.mpeg', '.mpg']

    # 获取文件列表
    video_files = []
    for file in Path(input_dir).iterdir():
        if file.suffix.lower() in extensions and file.is_file():
            video_files.append(file)

    if not video_files:
        print(f"⚠️ 在 {input_dir} 中未找到视频文件")
        return

    print(f"📂 找到 {len(video_files)} 个视频文件，开始处理...\n")

    success_count = 0
    fail_count = 0

    for i, video_file in enumerate(video_files, 1):
        print(f"=" * 40)
        print(f"[{i}/{len(video_files)}] 正在处理: {video_file.name}")

        # 1. 查找时间点
        cut_time = find_timestamp_in_video(str(video_file), end_image_path, search_duration)

        if cut_time is None:
            print("⚠️ 未找到标志，跳过此文件")
            fail_count += 1
            continue

        # 2. 构建输出路径
        # 直接使用原文件名
        output_filename = video_file.name
        output_filepath = os.path.join(output_dir, output_filename)

        # 3. 执行转码裁剪
        success = convert_and_cut_ffmpeg(str(video_file), output_filepath, cut_time)

        if success:
            success_count += 1
        else:
            fail_count += 1

    print("\n" + "=" * 40)
    print("🏁 所有任务处理完毕")
    print(f"✅ 成功: {success_count}")
    print(f"❌ 失败/跳过: {fail_count}")


if __name__ == "__main__":
    # ================= 配置区域 =================
    # 获取脚本所在的目录作为根目录
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    # 输入文件夹路径：脚本所在目录下的 input_videos
    INPUT_DIR = os.path.join(SCRIPT_DIR, "input_videos")

    # 输出文件夹路径：脚本所在目录下的 output_videos
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output_videos")

    # 结束标志图片路径：脚本所在目录下的 end.png
    END_IMAGE = os.path.join(SCRIPT_DIR, "end.png")

    # 在视频末尾多少秒内搜索
    SEARCH_TIME = 20
    # ===========================================

    print("🚀 视频批量转码裁剪工具 (GPU 加速版) 启动")
    print(f"当前目录: {SCRIPT_DIR}")
    print(f"输入: {INPUT_DIR}")
    print(f"输出: {OUTPUT_DIR}")
    print(f"标志: {END_IMAGE}")

    # 检查 FFmpeg 是否可用
    try:
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        print("❌ 错误: 未找到 FFmpeg，请确保已安装并添加到系统 PATH 环境变量中。")
        input("按回车键退出...")
        sys.exit(1)

    batch_process(INPUT_DIR, OUTPUT_DIR, END_IMAGE, SEARCH_TIME)

    print("\n按回车键退出...")
    input()