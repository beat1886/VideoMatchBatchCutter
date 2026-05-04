import cv2
import subprocess
import os
import sys
from pathlib import Path
import multiprocessing
from tqdm import tqdm

# ================= 配置区域 =================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(SCRIPT_DIR, "input_videos")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output_videos")
END_IMAGE = "end.png" 
SEARCH_TIME = 20

# 创建一个全局的打印锁，防止多进程同时打印导致换行混乱
print_lock = multiprocessing.Lock()
# ===========================================

def safe_print(*args, **kwargs):
    """带锁的安全打印函数"""
    with print_lock:
        print(*args, **kwargs)

def find_timestamp_in_video(video_path, end_image_path, search_duration=5):
    """使用 OpenCV 在视频最后指定秒数范围内查找结束标志图片。"""
    if not os.path.exists(video_path) or not os.path.exists(end_image_path): 
        return None
    end_img = cv2.imread(end_image_path)
    if end_img is None: 
        return None
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or total_frames <= 0: 
        cap.release()
        return None
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
            # 使用带锁的安全打印
            safe_print(f"\n[匹配成功] {os.path.basename(video_path)} -> 裁剪点: {target_time:.2f}s")
            break
        frame_idx += 1
    cap.release()
    return target_time


def convert_and_cut_ffmpeg(input_path, output_path, cut_time, use_gpu):
    """根据 use_gpu 参数选择编码策略"""
    if use_gpu:
        cmd = [
            'ffmpeg', '-i', input_path, '-to', str(cut_time),
            '-c:v', 'h264_nvenc', '-preset', 'p5', '-cq', '21', '-rc', 'vbr',
            '-c:a', 'aac', '-b:a', '128k', '-movflags', '+faststart', '-y', output_path
        ]
    else:
        cmd = [
            'ffmpeg', '-i', input_path, '-to', str(cut_time),
            '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23',
            '-c:a', 'aac', '-b:a', '128k', '-movflags', '+faststart', '-y', output_path
        ]

    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, 
                                encoding='utf-8', errors='ignore', timeout=3600,
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        return result.returncode == 0
    except Exception:
        return False


def process_single_video(args):
    """多进程/单进程调用的独立函数"""
    index, video_file, output_dir, end_image_path, search_duration, use_gpu = args
    
    # 使用带锁的安全打印，确保每个任务输出独占一行
    safe_print(f"\n🎬 [任务 {index+1}] 开始处理: {video_file.name}")
    
    cut_time = find_timestamp_in_video(str(video_file), end_image_path, search_duration)
    if cut_time is None:
        return (index, video_file.name, False, "未找到结束标志(阈值0.85)")
        
    output_filepath = os.path.join(output_dir, video_file.name)
    success = convert_and_cut_ffmpeg(str(video_file), output_filepath, cut_time, use_gpu)
    
    if success:
        return (index, video_file.name, True, f"裁剪至 {cut_time:.2f}s")
    else:
        return (index, video_file.name, False, "FFmpeg 转码失败")


def batch_process(input_dir, output_dir, end_image_path, search_duration, mode):
    """
    mode: 1=CPU多核并行, 2=GPU单核, 3=CPU单核非并行
    """
    if not os.path.exists(input_dir): 
        safe_print("❌ 输入文件夹不存在！")
        return
    os.makedirs(output_dir, exist_ok=True)
    
    extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.mpeg', '.mpg']
    video_files = [f for f in Path(input_dir).iterdir() if f.suffix.lower() in extensions and f.is_file()]
    video_files.sort()
    
    if not video_files: 
        safe_print("⚠️ 无视频文件")
        return

    # 根据模式设定进程数和描述
    if mode == 1:
        processes = min(multiprocessing.cpu_count(), len(video_files))
        use_gpu = False
        mode_str = f"💻 CPU 稳定模式 (开启 {processes} 进程并行)"
    elif mode == 2:
        processes = 1
        use_gpu = True
        mode_str = "🚀 GPU 加速模式 (单核运行)"
    else: # mode == 3
        processes = 1
        use_gpu = False
        mode_str = "💻 CPU 稳定模式 (单核非并行)"

    safe_print(f"📂 找到 {len(video_files)} 个视频，当前模式：{mode_str}\n")

    tasks = [(i, f, output_dir, end_image_path, search_duration, use_gpu) for i, f in enumerate(video_files)]
    
    final_results = []
    # 进度条 position=1 固定在底部，desc 根据模式动态变化
    desc_text = "总体处理进度" if mode != 3 else "处理中"
    
    # 如果是单核模式（选项2或3），直接用普通循环，避免多进程开销，日志更干净
    if mode in [2, 3]:
        for task in tqdm(tasks, desc=desc_text, position=1, leave=True):
            result = process_single_video(task)
            final_results.append(result)
    else:
        # 多核并行模式（选项1），使用进程池
        with multiprocessing.Pool(processes=processes) as pool:
            for result in tqdm(pool.imap(process_single_video, tasks, chunksize=1), total=len(tasks), desc=desc_text, position=1, leave=True):
                final_results.append(result)
        
    final_results.sort(key=lambda x: x[0])
    
    # 打印最终报告
    safe_print("\n" + "="*60)
    safe_print("📊 处理结果报告 (按文件顺序排列)")
    safe_print("="*60)
    
    success_count = 0
    for index, filename, success, msg in final_results:
        status_icon = "✅" if success else "❌"
        safe_print(f"{status_icon} [{index+1:02d}] {filename:<40} | {msg}")
        if success: success_count += 1
        
    safe_print("="*60)
    safe_print(f"🏁 全部任务结束 | 成功: {success_count} / 总数: {len(video_files)}")


if __name__ == "__main__":
    # 检查 FFmpeg
    try:
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        safe_print("❌ 未找到 FFmpeg，请确保已添加到环境变量")
        input("按回车退出...")
        sys.exit(1)

    # --- 交互式菜单 ---
    print("=" * 45)
    print("   视频批量转码裁剪工具 (多进程版)")
    print("=" * 45)
    print("1. CPU 多核并行 (极速，适合大批量处理)")
    print("2. GPU 硬件加速 (最快，需 NVIDIA 显卡)")
    print("3. CPU 单核非并行 (最稳，日志输出最清晰)")
    print("=" * 45)

    choice = input("请输入选项 (1 / 2 / 3) [默认 1]: ").strip()
    
    # 默认为 1，输入 2 或 3 则切换对应模式
    run_mode = 1
    if choice == '2':
        run_mode = 2
    elif choice == '3':
        run_mode = 3

    batch_process(INPUT_DIR, OUTPUT_DIR, END_IMAGE, SEARCH_TIME, run_mode)

    print("\n按回车键退出...")
    input()
