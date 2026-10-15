@echo off
rem LFW high-resolution-gallery check: restore every arm (resumable), then score. A restart continues where it stopped.
cd /d D:\FYP_AUR_SUR
if exist outputs\sr\.score_lock rmdir outputs\sr\.score_lock
:retry
envs\diag\Scripts\python.exe -W ignore scripts\sr_lfw_check.py all >> outputs\sr\lfw_worker.log 2>&1
if errorlevel 1 (
  echo [%date% %time%] lfw worker exited with an error, retrying in 60 s >> outputs\sr\lfw_worker.log
  timeout /t 60 /nobreak >nul
  goto retry
)
echo [%date% %time%] LFW CHECK FINISHED >> outputs\sr\lfw_worker.log
