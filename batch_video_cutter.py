import cv2
import subprocess
import os
import sys
from pathlib import Path

# ================= 配置区域 =================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(SCRIPT_DIR, "input_videos")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output_videos")
END_IMAGE = os.path.join(SCRIPT_DIR, "end.png")
SEARCH_TIME = 20


# ===========================================

def find_timestamp_in_video(video_path, end_image_path, search_duration=5):
    """
    使用 OpenCV 在视频最后指定秒数范围内查找结束标志图片。
    (逻辑保持不变)
    """
    print(f"🔍 正在分析: {os.path.basename(video_path)}")
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
            print(f"✅ 匹配成功: {target_time:.2f}s")
            break
        frame_idx += 1
        if frame_idx % (fps * 2) == 0: print(".", end="", flush=True)
    cap.release()
    return target_time


def convert_and_cut_ffmpeg(input_path, output_path, cut_time, use_gpu):
    """
    根据 use_gpu 参数选择编码策略
    """
    if use_gpu:
        # --- GPU 模式 (NVIDIA NVENC) ---
        print(f"🚀 正在使用 NVIDIA GPU 加速 (h264_nvenc)...")
        cmd = [
            'ffmpeg', '-i', input_path, '-to', str(cut_time),
            '-c:v', 'h264_nvenc',  # NVIDIA 编码器
            '-preset', 'p5',  # 质量预设 (p1-p7，p5 平衡)
            '-cq', '21',  # 恒定画质
            '-rc', 'vbr',  # 动态码率
            '-c:a', 'aac', '-b:a', '128k',
            '-movflags', '+faststart', '-y', output_path
        ]
    else:
        # --- CPU 模式 (libx264) ---
        print(f"💻 正在使用 CPU 编码 (libx264)...")
        cmd = [
            'ffmpeg', '-i', input_path, '-to', str(cut_time),
            '-c:v', 'libx264',  # CPU 编码器
            '-preset', 'veryfast',  # 极速预设
            '-crf', '23',  # 恒定画质
            '-c:a', 'aac', '-b:a', '128k',
            '-movflags', '+faststart', '-y', output_path
        ]

    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8',
                                errors='ignore', creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        if result.returncode == 0:
            return True
        else:
            if use_gpu:
                print(f"❌ GPU 编码失败 (建议检查驱动或切换 CPU 模式): {result.stderr[:150]}")
            else:
                print(f"❌ CPU 编码失败: {result.stderr[:150]}")
            return False
    except Exception as e:
        print(f"❌ 异常: {e}")
        return False


def batch_process(input_dir, output_dir, end_image_path, search_duration, use_gpu):
    if not os.path.exists(input_dir): return
    os.makedirs(output_dir, exist_ok=True)
    extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.mpeg', '.mpg']
    video_files = [f for f in Path(input_dir).iterdir() if f.suffix.lower() in extensions and f.is_file()]
    if not video_files: print("⚠️ 无视频文件"); return

    mode_str = "🚀 GPU 加速模式" if use_gpu else "💻 CPU 稳定模式"
    print(f"📂 找到 {len(video_files)} 个视频 ({mode_str})\n")

    success_count = 0
    for i, video_file in enumerate(video_files, 1):
        print(f"[{i}/{len(video_files)}] 处理: {video_file.name}")
        cut_time = find_timestamp_in_video(str(video_file), end_image_path, search_duration)
        if cut_time is None:
            print("⚠️ 未找到标志，跳过")
            continue
        output_filepath = os.path.join(output_dir, video_file.name)
        if convert_and_cut_ffmpeg(str(video_file), output_filepath, cut_time, use_gpu):
            success_count += 1
    print(f"\n🏁 完成 | 成功: {success_count}")


if __name__ == "__main__":
    # 检查 FFmpeg
    try:
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        print("❌ 未找到 FFmpeg，请确保已添加到环境变量"); input("按回车退出..."); sys.exit(1)

    # --- 交互式菜单 ---
    print("=" * 40)
    print("   视频批量转码裁剪工具")
    print("=" * 40)
    print("1. 使用 CPU 编码 (稳定，兼容性好)")
    print("2. 使用 GPU 编码 (极速，需 NVIDIA 显卡)")
    print("=" * 40)

    choice = input("请输入选项 (1 或 2) [默认 1]: ").strip()

    # 默认为 CPU 模式，输入 2 则开启 GPU
    enable_gpu = (choice == '2')

    batch_process(INPUT_DIR, OUTPUT_DIR, END_IMAGE, SEARCH_TIME, enable_gpu)

    print("\n按回车键退出...")
    input()
