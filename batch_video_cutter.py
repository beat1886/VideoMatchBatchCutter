import cv2
import numpy as np
import subprocess
import os
import sys
from pathlib import Path
import shutil

def find_and_cut_video(video_path, end_image_path, output_path, search_duration=5):
    """
    在视频最后指定秒数范围内查找匹配end.png的帧，并裁剪掉该帧及其后续内容（无损裁剪）
    支持 H.264, HEVC, MPEG-4 等多种编码
    """
    # 清理路径引号
    video_path = video_path.strip('"')
    end_image_path = end_image_path.strip('"')
    output_path = output_path.strip('"')

    print(f"处理视频: {os.path.basename(video_path)}")

    # 检查文件是否存在
    if not os.path.exists(video_path):
        print(f"  错误：视频文件不存在: {video_path}")
        return False

    if not os.path.exists(end_image_path):
        print(f"  错误：结束标志图片不存在: {end_image_path}")
        return False

    # 加载结束标志图片
    end_img = cv2.imread(end_image_path)
    if end_img is None:
        print(f"  错误：无法加载结束标志图片: {end_image_path}")
        return False

    # 获取视频信息
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # H.264 兼容性修复：有些 H.264 视频 OpenCV 读取 FPS 可能为 0 或极低
    if fps <= 0 or fps > 1000:
        print(f"  警告：检测到异常FPS值 ({fps})，尝试强制设定为 30.00")
        fps = 30.00

    if total_frames <= 0:
        print("  错误：无法获取视频总帧数")
        cap.release()
        return False

    duration = total_frames / fps
    print(f"  视频信息: FPS={fps:.2f}, 总帧数={total_frames}, 分辨率={width}x{height}")
    print(f"  视频时长: {duration:.2f}秒")

    # 计算最后指定秒数的帧数
    last_search_frames = int(search_duration * fps)
    start_search_frame = max(0, total_frames - last_search_frames)

    print(f"  从第 {start_search_frame} 帧开始搜索（最后{search_duration}秒）")

    # 跳转到开始搜索的位置
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_search_frame)

    target_frame_idx = -1
    frame_count = start_search_frame

    # 调整结束标志图片大小以匹配视频分辨率
    # 注意：如果图片比例与视频不同，这里可能会变形，但为了匹配速度通常这样做
    try:
        resized_end_img = cv2.resize(end_img, (width, height))
    except cv2.error as e:
        print(f"  错误：调整图片大小失败，可能是分辨率异常: {e}")
        cap.release()
        return False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 使用模板匹配查找结束标志
        # 使用 TM_CCOEFF_NORMED 算法，对亮度变化有一定鲁棒性
        result = cv2.matchTemplate(frame, resized_end_img, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)

        # 设置匹配阈值 (0.8 是一个比较平衡的值)
        threshold = 0.8
        if max_val >= threshold:
            print(f"  [成功] 在第 {frame_count} 帧找到结束标志 (匹配度: {max_val:.4f})")
            target_frame_idx = frame_count
            break

        frame_count += 1
        if frame_count >= total_frames:
            break

    cap.release()

    if target_frame_idx == -1:
        print("  [提示] 未找到结束标志，直接复制原文件")
        shutil.copy2(video_path, output_path)
        return True

    # 计算裁剪时长
    cut_time = target_frame_idx / fps
    print(f"  执行裁剪: 保留前 {cut_time:.2f} 秒")

    # 确保输出路径包含文件扩展名
    if not output_path.lower().endswith(('.mp4', '.mov', '.mkv', '.avi')):
        output_path += '.mp4'

    # 创建输出目录
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 使用 FFmpeg 进行无损裁剪
    # 关键参数说明：
    # -c copy: 直接复制流，不重新编码（速度快，无损）
    # -to: 指定结束时间点
    # -avoid_negative_ts 1: 避免负时间戳，防止播放器无法播放
    # -fflags +genpts: 强制生成新的时间戳，修复 H.264 裁剪后的花屏或同步问题
    cmd = [
        'ffmpeg',
        '-i', video_path,
        '-to', str(cut_time),
        '-c', 'copy',
        '-avoid_negative_ts', '1',
        '-fflags', '+genpts',
        '-y',
        output_path
    ]

    try:
        # 隐藏 FFmpeg 的控制台窗口（仅在 Windows 下有效）
        startupinfo = None
        if os.name == 'nt':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='ignore',
            startupinfo=startupinfo
        )
        print(f"  [完成] 视频已保存: {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  [错误] FFmpeg 执行失败: {e}")
        return False
    except Exception as e:
        print(f"  [错误] 未知异常: {e}")
        return False

def batch_process_videos(input_dir, output_dir, end_image_path, search_duration=5):
    """
    批量处理视频文件
    """
    os.makedirs(output_dir, exist_ok=True)

    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.mpg', '.mpeg', '.m4v']
    video_files = []

    # 仅搜索当前目录，不递归子目录
    for ext in video_extensions:
        for file in Path(input_dir).iterdir():
            if file.is_file() and file.suffix.lower() == ext:
                video_files.append(file)

    if not video_files:
        print(f"在目录 {input_dir} 中未找到支持的视频文件")
        return

    print(f"找到 {len(video_files)} 个视频文件，开始处理...")

    success_count = 0
    fail_count = 0

    for i, video_file in enumerate(video_files, 1):
        print(f"\n--- [{i}/{len(video_files)}] 正在处理: {video_file.name} ---")

        output_file = os.path.join(output_dir, f"{video_file.stem}_裁剪后{video_file.suffix}")

        success = find_and_cut_video(str(video_file), end_image_path, output_file, search_duration)

        if success:
            success_count += 1
        else:
            fail_count += 1

    print(f"\n" + "="*30)
    print(f"批量处理完成！")
    print(f"成功: {success_count} | 失败: {fail_count}")
    print("="*30)

def main():
    # ================== 配置区域 ==================
    INPUT_DIR = r"D:\cut\新建文件夹\视频"      # 输入视频文件夹
    OUTPUT_DIR = r"D:\cut\新建文件夹\裁切完成"  # 输出视频文件夹
    END_IMAGE_PATH = "end.png"                  # 结束标志图片路径
    SEARCH_DURATION = 5                        # 在视频最后多少秒内搜索
    # =============================================

    print("=== 批量视频自动裁剪工具 (H.264 Optimized) ===")
    print(f"输入目录: {INPUT_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")

    if not os.path.exists(INPUT_DIR):
        print(f"错误：输入目录不存在 -> {INPUT_DIR}")
        input("按回车键退出...")
        return

    if not os.path.exists(END_IMAGE_PATH):
        print(f"错误：结束标志图片不存在 -> {END_IMAGE_PATH}")
        input("按回车键退出...")
        return

    batch_process_videos(INPUT_DIR, OUTPUT_DIR, END_IMAGE_PATH, SEARCH_DURATION)

    print("\n所有任务结束。")
    input("按回车键退出...")

if __name__ == "__main__":
    main()