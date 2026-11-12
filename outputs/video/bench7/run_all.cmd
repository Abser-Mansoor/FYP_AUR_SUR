@echo off
cd /d D:\FYP_AUR_SUR
envs\video\Scripts\python.exe scripts\video_phase7_bench.py --part D0 > outputs\video\bench7\run_D0.log 2>&1
envs\video\Scripts\python.exe scripts\video_phase7_bench.py --part B > outputs\video\bench7\run_B.log 2>&1
envs\video\Scripts\python.exe scripts\video_phase7_bench.py --part E > outputs\video\bench7\run_E.log 2>&1
