@echo off
chcp 65001 >nul

:: 清理 input_videos 目录
if exist "input_videos\" (
    del /f /q "input_videos\*" >nul 2>&1
    for /d %%i in ("input_videos\*") do rd /s /q "%%i"
)

:: 清理 output_videos 目录
if exist "output_videos\" (
    del /f /q "output_videos\*" >nul 2>&1
    for /d %%i in ("output_videos\*") do rd /s /q "%%i"
)

:: 退出脚本（不弹框，直接关闭窗口）
exit
