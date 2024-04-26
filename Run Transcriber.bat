@echo off
cd /d "%~dp0\transcribe_res"
call "%~dp0transcribe_res\anaconda3\Scripts\activate.bat"
echo Loading Transcriber...
python gui.py
pause