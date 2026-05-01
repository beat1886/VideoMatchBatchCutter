import cv2
import subprocess
import os
import sys
from pathlib import Path

# ... (find_timestamp_in_video 函数保持不变，省略以节省篇幅) ...
def find_timestamp_in_video(video_path, end_image_path, search_duration=5):
    # ... 保持原有代码不变 ...
    print(f"🔍 正在分析视频画面: {os.path.basename(video_path)}")
    if not os.path.exists(video_path) or not os.path.exists(end_image_path): return None
    end_img = cv2.imread(end_image_path)
    if end_img is None: return None
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or total_frames <= 0: cap.release(); return None
    duration = total_frames / fps
    search_start_frame = int(max(0, duration - search_duration) * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, search_start_frame)
    resized_template = cv2.resize(end_img, (width, height))
    target_time = None
    frame_idx = search_start_frame
    threshold = 0.85
    while True:
        ret, frame = cap.read()
        if not ret: break
        result = cv2.matchTemplate(frame, resized_template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        if max_val >= threshold:
            target_time = frame_idx / fps
            print(f"✅ 匹配成功! 时间点: {target_time:.2f}s")
            break
        frame_idx += 1
        if frame_idx % (fps * 2) == 0: print(".", end="", flush=True)
    cap.release()
    return target_time

def detect_best_encoder():
    """
    【核心通用逻辑】
    自动扫描 FFmpeg 支持的所有硬件编码器，并按性能优先级返回最佳方案。
    """
    try:
        # 获取 FFmpeg 支持的所有编码器
        result = subprocess.run(
            ['ffmpeg', '-encoders'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        encoders = result.stdout
        
        # 优先级列表：按顺序查找，找到即返回
        # 1. NVIDIA (NVENC) - 支持 GTX 10/16/20/30/40/50 系列
        if 'h264_nvenc' in encoders:
            return "nvenc", "NVIDIA GPU (NVENC)"
        
        # 2. Intel (QSV) - 支持 HD 4600 及更新型号
        elif 'h264_qsv' in encoders:
            return "qsv", "Intel GPU (QSV)"
            
        # 3. AMD (AMF - Windows) - 支持 RX 系列
        elif 'h264_amf' in encoders:
            return "amf", "AMD GPU (AMF)"
            
        # 4. AMD (VAAPI - Linux)
        elif 'h264_vaapi' in encoders:
            return "vaapi", "AMD/Intel GPU (VAAPI)"

        # 5. 兜底：CPU
        else:
            return "cpu", "CPU (Software)"
            
    except Exception as e:
        print(f"⚠️ 检测出错: {e}")
        return "cpu", "CPU (Error)"

def convert_and_cut_ffmpeg(input_path, output_path, cut_time, encoder_type):
    """
    根据探测到的编码器类型，动态构建 FFmpeg 命令
    """
    cmd = ['ffmpeg', '-i', input_path, '-to', str(cut_time)]

    # 动态参数配置
    if encoder_type == "nvenc":
        # NVIDIA 专用参数 (通用性强)
        print("🚀 使用 NVIDIA 硬件加速 (高画质模式)...")
        cmd.extend([
            '-c:v', 'h264_nvenc',
            '-preset', 'p5',  # 质量与速度的平衡点
            '-cq', '21',      # 恒定画质 (类似 CRF)
            '-rc', 'vbr',     # 动态码率
            '-c:a', 'aac', '-b:a', '128k',
            '-movflags', '+faststart', '-y', output_path
        ])
        
    elif encoder_type == "qsv":
        # Intel 专用参数
        print("⚡ 使用 Intel 硬件加速...")
        cmd.extend([
            '-c:v', 'h264_qsv',
            '-preset', 'veryfast',
            '-global_quality', '23',
            '-c:a', 'aac', '-b:a', '128k',
            '-movflags', '+faststart', '-y', output_path
        ])

    elif encoder_type == "amf":
        # AMD Windows 专用参数
        print("🔴 使用 AMD 硬件加速 (AMF)...")
        cmd.extend([
            '-c:v', 'h264_amf',
            '-quality', 'quality',
            '-c:a', 'aac', '-b:a', '128k',
            '-movflags', '+faststart', '-y', output_path
        ])
        
    elif encoder_type == "vaapi":
        # AMD/Intel Linux 专用参数
        print("🐧 使用 Linux 硬件加速 (VAAPI)...")
        cmd.extend([
            '-c:v', 'h264_vaapi',
            '-qp', '23',
            '-c:a', 'aac', '-b:a', '128k',
            '-movflags', '+faststart', '-y', output_path
        ])

    else:
        # CPU 兜底
        print("💻 使用 CPU 编码 (通用兼容)...")
        cmd.extend([
            '-c:v', 'libx264',
            '-preset', 'veryfast',
            '-crf', '23',
            '-c:a', 'aac', '-b:a', '128k',
            '-movflags', '+faststart', '-y', output_path
        ])

    # 执行命令
    try:
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding='utf-8', errors='ignore',
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        )
        if result.returncode == 0:
            return True
        else:
            print(f"❌ 编码失败: {result.stderr[:150]}...")
            return False
    except Exception as e:
        print(f"❌ 执行异常: {e}")
        return False

def batch_process(input_dir, output_dir, end_image_path, search_duration=10):
    if not os.path.exists(input_dir): return
    os.makedirs(output_dir, exist_ok=True)

    extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.mpeg', '.mpg']
    video_files = [f for f in Path(input_dir).iterdir() if f.suffix.lower() in extensions and f.is_file()]

    if not video_files:
        print(f"⚠️ 在 {input_dir} 中未找到视频文件")
        return

    print(f"📂 找到 {len(video_files)} 个视频文件\n")
    
    # --- 核心：启动时自动探测最强硬件 ---
    best_encoder, encoder_name = detect_best_encoder()
    print(f"🔎 系统探测结果: 最佳可用编码器为 [{encoder_name}]")
    print("-" * 40)

    success_count = 0
    fail_count = 0

    for i, video_file in enumerate(video_files, 1):
        print(f"[{i}/{len(video_files)}] 处理: {video_file.name}")
        
        cut_time = find_timestamp_in_video(str(video_file), end_image_path, search_duration)
        if cut_time is None:
            fail_count += 1
            continue

        output_filepath = os.path.join(output_dir, video_file.name)
        if convert_and_cut_ffmpeg(str(video_file), output_filepath, cut_time, best_encoder):
            success_count += 1
        else:
            fail_count += 1

    print(f"\n🏁 完成 | 成功: {success_count}, 失败: {fail_count}")

if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    INPUT_DIR = os.path.join(SCRIPT_DIR, "input_videos")
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output_videos")
    END_IMAGE = os.path.join(SCRIPT_DIR, "end.png")
    
    # 检查 FFmpeg
    try: subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        print("❌ 未找到 FFmpeg")
        sys.exit(1)

    batch_process(INPUT_DIR, OUTPUT_DIR, END_IMAGE, 20)
    input("\n按回车退出...")
