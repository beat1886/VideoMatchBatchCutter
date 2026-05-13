import cv2
import subprocess
import os
import sys
from pathlib import Path
import multiprocessing
from multiprocessing import Queue
import time

# ================= 配置区域 =================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(SCRIPT_DIR, "input_videos")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output_videos")
END_IMAGE = "end.png"
SEARCH_TIME = 20
# ===========================================

def draw_interface(video_files, results, cpu_processing_set, gpu_processing_set, mode_str, start_time):
    FIXED_VISUAL_WIDTH = 44
    spinners = ['\\', '|', '/', '-']
    spinner_char = spinners[int(time.time() * 2) % 4]

    def get_visual_len(text):
        length = 0
        for c in text:
            if '\u4e00' <= c <= '\u9fff' or c in '【】（）《》':
                length += 2
            else:
                length += 1
        return length

    def format_filename(name):
        v_len = get_visual_len(name)
        if v_len > FIXED_VISUAL_WIDTH:
            target_len = FIXED_VISUAL_WIDTH - 3
            current_len = 0
            truncated = ""
            for c in name:
                cl = 2 if ('\u4e00' <= c <= '\u9fff' or c in '【】（）《》') else 1
                if current_len + cl > target_len:
                    break
                truncated += c
                current_len += cl
            return truncated + "... "
        else:
            pad = FIXED_VISUAL_WIDTH - v_len
            return name + " " * pad

    total = len(video_files)
    done = len(results)
    percent = int((done / total) * 100) if total > 0 else 0
    # 仅修改：秒转时分秒
    cost = int(time.time() - start_time)
    h = cost // 3600
    m = (cost % 3600) // 60
    s = cost % 60
    time_str = f"{h:02d}:{m:02d}:{s:02d}"

    print("=" * 85)
    print(f"  视频批量转码裁剪工具 | 模式: {mode_str}")
    print("=" * 85)

    for i, path in enumerate(video_files):
        res = next((r for r in results if r[0] == i), None)
        if res:
            status = f"✅ 成功 ({res[3]})" if res[2] else f"❌ 失败 ({res[3]})"
        elif i in gpu_processing_set:
            status = f"{spinner_char} GPU硬件编码裁剪"
        elif i in cpu_processing_set:
            status = f"{spinner_char} CPU逐帧匹配片尾"
        else:
            status = "⏳ 排队等待处理"

        name = os.path.basename(path)
        fmt_name = format_filename(name)
        print(f"[{i+1:02d}/{total}] {fmt_name} {status}")

    print("-" * 85)
    bar_len = 45
    filled = int(bar_len * done // total) if total else 0
    bar = '█' * filled + '-' * (bar_len - filled)
    # 改用时分秒显示
    print(f"总进度: |{bar}| {percent}%  已完成 {done}/{total}  耗时 {time_str}")
    print("=" * 85)
    print()

def find_timestamp_in_video(video_path, end_image_path, search_duration=5):
    if not os.path.exists(video_path) or not os.path.exists(end_image_path):
        return None
    end_img = cv2.imread(end_image_path)
    if end_img is None:
        return None
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or total_frames <= 0:
        cap.release()
        return None
    duration = total_frames / fps
    start_frame = int(max(0, duration - search_duration) * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    tmpl = cv2.resize(end_img, (w, h))
    target = None
    idx = start_frame
    while True:
        ret, frame = cap.read()
        if not ret: break
        res = cv2.matchTemplate(frame, tmpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        if max_val >= 0.85:
            target = idx / fps
            break
        idx += 1
    cap.release()
    return target

def convert_and_cut_ffmpeg(in_path, out_path, cut_time, use_gpu):
    if use_gpu:
        cmd = [
            'ffmpeg', '-hwaccel', 'cuda', '-i', in_path, '-to', str(cut_time),
            '-c:v', 'h264_nvenc', '-preset', 'p5', '-cq', '21', '-rc', 'vbr',
            '-c:a', 'aac', '-b:a', '128k', '-movflags', '+faststart', '-y', out_path
        ]
    else:
        cmd = [
            'ffmpeg', '-i', in_path, '-to,', str(cut_time),
            '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23',
            '-c:a', 'aac', '-b:a', '128k', '-movflags', '+faststart', '-y', out_path
        ]
    try:
        flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=flags, timeout=3600)
        return True
    except Exception:
        return False

def cpu_worker(task_queue, result_queue, update_queue, end_image_path, search_duration):
    while True:
        task = task_queue.get()
        if task is None: break
        idx, f = task
        update_queue.put(("cpu_start", idx))
        cut_time = find_timestamp_in_video(f, end_image_path, search_duration)
        update_queue.put(("cpu_end", idx))
        result_queue.put((idx, f, cut_time))

def gpu_worker(result_queue, output_dir, final_results, lock, update_queue):
    while True:
        task = result_queue.get()
        if task is None: break
        idx, f, cut_time = task
        update_queue.put(("gpu_start", idx))
        out = os.path.join(output_dir, os.path.basename(f))
        if cut_time is None:
            with lock:
                final_results.append((idx, f, False, "未匹配到片尾"))
        else:
            ok = convert_and_cut_ffmpeg(f, out, cut_time, use_gpu=True)
            with lock:
                final_results.append((idx, f, ok, f"裁剪至 {cut_time:.2f}s"))
        update_queue.put(("gpu_end", idx))

def batch_process(input_dir, output_dir, end_image_path, search_duration, mode):
    exts = ['.mp4', '.avi', '.mov', '.mkv', '.flv']
    files = sorted([str(f) for f in Path(input_dir).iterdir() if f.suffix.lower() in exts])
    if not files:
        print("未找到视频")
        return

    start_time = time.time()
    mgr = multiprocessing.Manager()
    results = mgr.list()
    cpu_proc = mgr.list()
    gpu_proc = mgr.list()
    update_queue = Queue()

    if mode == 3:
        cpu_n = 6
        gpu_n = 2
        mode_str = f"CPU+GPU流水线 (CPU:{cpu_n}核 + GPU:{gpu_n}进程)"

        task_queue = Queue()
        result_queue = Queue()

        cpu_list = []
        for _ in range(cpu_n):
            p = multiprocessing.Process(target=cpu_worker, args=(task_queue, result_queue, update_queue, end_image_path, search_duration))
            cpu_list.append(p)
            p.start()

        gpu_list = []
        lock = mgr.Lock()
        for _ in range(gpu_n):
            p = multiprocessing.Process(target=gpu_worker, args=(result_queue, output_dir, results, lock, update_queue))
            gpu_list.append(p)
            p.start()

        for i in range(len(files)):
            task_queue.put((i, files[i]))

        draw_interface(files, results, cpu_proc, gpu_proc, mode_str, start_time)
        while len(results) < len(files):
            updated = False
            while not update_queue.empty():
                t, idx = update_queue.get()
                if t == "cpu_start" and idx not in cpu_proc:
                    cpu_proc.append(idx)
                elif t == "cpu_end" and idx in cpu_proc:
                    cpu_proc.remove(idx)
                elif t == "gpu_start" and idx not in gpu_proc:
                    gpu_proc.append(idx)
                elif t == "gpu_end" and idx in gpu_proc:
                    gpu_proc.remove(idx)
                updated = True
            if updated:
                draw_interface(files, results, cpu_proc, gpu_proc, mode_str, start_time)
            time.sleep(0.1)

        for _ in cpu_list: task_queue.put(None)
        for _ in gpu_list: result_queue.put(None)
        for p in cpu_list: p.join()
        for p in gpu_list: p.join()

    draw_interface(files, results, cpu_proc, gpu_proc, mode_str, start_time)

def main():
    print("============================================================")
    print("            视频批量转码裁剪工具")
    print("============================================================")
    print(" 1 - CPU 多核并行")
    print(" 2 - GPU 硬件加速")
    print(" 3 - CPU+GPU 流水线（分别并行处理）")
    print(" 4 - CPU 单核调试")
    print("============================================================")
    c = input("请选择模式 [1/2/3/4]，默认3：").strip()
    mode = 3
    if c == "1": mode = 1
    elif c == "2": mode = 2
    elif c == "4": mode = 4
    batch_process(INPUT_DIR, OUTPUT_DIR, END_IMAGE, SEARCH_TIME, mode)

if __name__ == "__main__":
    main()
