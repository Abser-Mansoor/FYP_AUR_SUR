@echo off
rem Single Phase 3 worker: runs every arm one after another. Resumable, so a restart continues where it stopped.
cd /d D:\FYP_AUR_SUR
if exist outputs\sr\.score_lock rmdir outputs\sr\.score_lock
:retry
envs\diag\Scripts\python.exe -W ignore scripts\sr_pipeline.py arm --arms tadsr >> outputs\sr\worker.log 2>&1
if errorlevel 1 (
  echo [%date% %time%] worker exited with an error, retrying in 60 s >> outputs\sr\worker.log
  timeout /t 60 /nobreak >nul
  goto retry
)
echo [%date% %time%] WORKER FINISHED >> outputs\sr\worker.log
