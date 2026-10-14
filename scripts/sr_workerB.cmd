@echo off
cd /d D:\FYP_AUR_SUR
envs\diag\Scripts\python.exe -W ignore scripts\sr_pipeline.py arm --arms tadsr bicubic esrgan >> outputs\sr\worker_B.log 2>&1
