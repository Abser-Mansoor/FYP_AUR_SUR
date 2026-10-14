@echo off
cd /d D:\FYP_AUR_SUR
if exist outputs\sr\.score_lock rmdir outputs\sr\.score_lock
envs\diag\Scripts\python.exe -W ignore scripts\sr_pipeline.py arm --arms invsr osdface >> outputs\sr\worker_A.log 2>&1
