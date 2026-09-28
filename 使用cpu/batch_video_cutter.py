import cv2
import numpy as np
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
END_IMAGE = os.path.join(SCRIPT_DIR, "end.png")
SEARCH_TIME = 20

# 创建一个全局的打印锁，防止多进程同时打印导致换行混乱
print_lock = multiprocessing.Lock()
# ===========================================

def safe_print(*args, **kwargs):
    """带锁的安全打印函数"""
    with print_lock:
        print(*args, **kwargs)

def find_timestamp_in_video(video_path, end_image_path, search_duration=SEARCH_TIME):
    """
    在视频末尾的搜索窗口内定位片尾结束标志。
    策略（片尾锚定）：结束标志必须出现在视频的【最后一帧】，再向前回溯它
    连续出现片段的起点，作为裁剪时间点。
    这样可避免录屏类视频【开头】出现相同的播放器待机画面时被误判为片尾
    （误判会得到裁剪点 0，进而导致 ffmpeg 转码失败）。
    返回裁剪时间戳(秒)；结尾未出现标志或正片过短时返回 None。
    """
    if not os.path.exists(video_path) or not os.path.exists(end_image_path):
        return None
    # cv2.imread 在 Windows 上无法读取含中文的路径，改用 imdecode + fromfile
    try:
        end_img = cv2.imdecode(np.fromfile(end_image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except OSError:
        end_img = None
    if end_img is None:
        safe_print(f"[警告] 无法读取结束标志图片: {end_image_path}")
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
    threshold = 0.85
    min_cut_duration = 1.0  # 裁剪后正片不足 1 秒视为无效，防止 ffmpeg 输出空视频

    frame_idx = search_start_frame
    final_score = -1.0       # 视频真实最后一帧与标志图的相似度
    seg_start = None         # 贴近结尾的片尾连续片段起点（秒，基于真实PTS）
    seg_start_pts = None     # 片尾首帧的真实PTS(秒)
    gap_frames = 0
    gap_tolerance = max(1, int(0.5 * fps))  # 片段内容许约 0.5 秒的单帧抖动
    pts_ok = False           # PTS 时间戳是否可用

    # 顺序读到真实结尾（不依赖总帧数 seek，避免越过文件末尾读不到帧）
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        pts_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if pts_sec > 0:
            pts_ok = True
        result = cv2.matchTemplate(frame, resized_template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        final_score = max_val
        if max_val >= threshold:
            if seg_start is None:
                seg_start = frame_idx / fps
                seg_start_pts = pts_sec
            gap_frames = 0
        elif seg_start is not None:
            gap_frames += 1
            if gap_frames > gap_tolerance:
                seg_start = None
                seg_start_pts = None
                gap_frames = 0
        frame_idx += 1
    cap.release()

    # 片尾标志必须出现在视频结尾，否则不做危险裁剪
    if final_score < threshold:
        return None
    if seg_start is None or seg_start < min_cut_duration:
        return None

    # 录屏多为可变帧率(VFR)，帧序号/平均帧率与真实时间戳存在秒级累计偏差，
    # 会导致裁剪点偏晚、输出残留片尾。优先使用每帧真实PTS。
    cut_time = seg_start_pts if (pts_ok and seg_start_pts) else seg_start
    if cut_time < min_cut_duration:
        return None

    safe_print(f"\n[匹配成功] {os.path.basename(video_path)} -> 裁剪点: {cut_time:.2f}s")
    return cut_time


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
        if result.returncode == 0:
            return True
        err_tail = ' | '.join((result.stderr or '').strip().splitlines()[-3:])
        safe_print(f"[FFmpeg错误] {os.path.basename(input_path)}: {err_tail}")
        return False
    except Exception as e:
        safe_print(f"[FFmpeg异常] {os.path.basename(input_path)}: {e}")
        return False


def process_single_video(args):
    """多进程/单进程调用的独立函数"""
    index, video_file, output_dir, end_image_path, search_duration, use_gpu = args
    
    # 使用带锁的安全打印，确保每个任务输出独占一行
    safe_print(f"\n🎬 [任务 {index+1}] 开始处理: {video_file.name}")
    
    cut_time = find_timestamp_in_video(str(video_file), end_image_path, search_duration)
    if cut_time is None:
        return (index, video_file.name, False, "结尾未检测到结束标志(阈值0.85)，已跳过")
        
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
    if not os.path.exists(end_image_path):
        safe_print(f"❌ 结束标志图片不存在: {end_image_path}")
        safe_print("   请确认 end.png 与脚本位于同一目录，再重新运行。")
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
