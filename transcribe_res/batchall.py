#pip installs
"""
numpy psutil pandas openpyxl transformers rich PyQt6 torch torchvision torchaudio
numpy psutil pandas openpyxl rich PyQt6 transformers
pip install transformers==4.40.1 torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
"""
#start /wait "" Miniconda3-py311_24.3.0-0-Windows-x86_64.exe /InstallationType=JustMe /AddToPath=0 /RegisterPython=0 /NoRegistry=1 /S /D=.\anaconda3

import os
import sys
import numpy as np
import shutil
import psutil
import subprocess
import threading
import queue
import json
import csv
import pandas as pd
import copy
from math import ceil
from datetime import timedelta
import importlib

cachepath = os.path.dirname(sys.executable)
if not os.path.dirname(sys.executable).endswith("transcribe_res\\anaconda3"):
    raise Exception("python.exe should be located in '.\\transcribe_res\\anaconda3'")
cachepath = cachepath[:-len("anaconda3")]+"cache"
os.environ['HF_HOME'] = cachepath
os.environ['HF_DATASETS_CACHE'] = cachepath
os.environ['TORCH_HOME'] = cachepath

from rich.progress import Progress, TimeElapsedColumn, BarColumn, TextColumn



def logger(header,msg):
    print(f"[{header}] {msg}")

FLASH = False


class Transcriber:
    def __init__(self,batch_size,device_id : str):
        import torch
        from transformers import pipelines
        importlib.reload(pipelines)
        self.batch_size = batch_size
        self.currentstep = 0
        
        model_name = "openai/whisper-large-v3"
        
        dtype = torch.float32
        if device_id != "cpu":
            dtype = torch.float16 if not torch.cuda.is_bf16_supported() else torch.bfloat16
        else:
            torch.set_num_threads(os.cpu_count())
        
        self.pipe = pipelines.pipeline(
            "automatic-speech-recognition",
            model=model_name,
            torch_dtype=dtype,
            device=device_id,
            model_kwargs={"attn_implementation": "flash_attention_2"} if (FLASH and device_id.startswith("cuda:"))  else {"attn_implementation": "sdpa"},
        )
        
        originalchunk_iter= copy.copy(pipelines.automatic_speech_recognition.chunk_iter)
        def wrapper(*args):
            try:
                inputs = args[0]
                chunk_len = args[2]
                stride_left = args[3]
                stride_right = args[4]
                inputs_len = inputs.shape[0]
                step = chunk_len - stride_left - stride_right

                logger("TOTALSTEPS",ceil(ceil(inputs_len/step)/self.batch_size))
            except:
                pass
            return originalchunk_iter(*args)
        pipelines.automatic_speech_recognition.chunk_iter = wrapper

        
        originalforward = copy.copy(self.pipe.forward)
        def wrapper_method(pipeself,*args,**kwargs):
            out = originalforward(pipeself,*args,**kwargs)
            self.currentstep += 1
            logger("STEP",self.currentstep)
            return out

        self.pipe.forward = wrapper_method

        if device_id == "mps":
            torch.mps.empty_cache()

    def transcribe(self,file_name : str,language=None,transcript_path=None):
        self.currentstep = 0
        #if language == none, autodetect language

        generate_kwargs = {"task": "transcribe", "language": language}

        #if model_name.split(".")[-1] == "en":
        #    generate_kwargs.pop("task")
        with Progress(
            TextColumn("🤗 [progress.description]{task.description}"),
            BarColumn(style="yellow1", pulse_style="white"),
            TimeElapsedColumn(),
            TextColumn(f"| {os.path.basename(file_name).split('/')[-1]}",style="yellow1")
        ) as progress:
            progress.add_task("[yellow]Transcribing...", total=None)

            outputs = self.pipe(
                file_name,
                chunk_length_s=30,
                batch_size=self.batch_size,
                generate_kwargs=generate_kwargs,
                return_timestamps=True, #"word" #True for chunk level or "word" for word level timestamps
            )

        if transcript_path != None:
            for chunk in outputs["chunks"]:
                chunk["start"] = None
                chunk["end"] = None
                try:
                    start = str(timedelta(seconds=chunk["timestamp"][0]))
                    end = str(timedelta(seconds=chunk["timestamp"][1]))
                    split = start.split(".")
                    if len(split) > 1:
                        split[1] = split[1][:2]
                        start = ".".join(split)
                    chunk["start"] = start

                    split = end.split(".")
                    if len(split) > 1:
                        split[1] = split[1][:2]
                        end = ".".join(split)
                    chunk["end"] = end
                except Exception as e:
                    print(e)
                    pass
            with open(transcript_path, "w", encoding="utf8") as fp:
                result = {"chunks": outputs["chunks"],"text": outputs["text"]}
                json.dump(result, fp, ensure_ascii=False)


def transcribeFiles(FILES,BATCHSIZE : int,DEVICE : str,OUTPUTDIR : str):
    temppath = ".\\temp"
    device = DEVICE
    

    def ffmpegCheckFile(filepath):
        """
        fileData = bytes()
        with open(filepath, "rb") as f:
            fileData = f.read()
        
        ar = f"{16000}" #sample rate
        ac = "1"
        format_for_conversion = "f32le"
        ffmpeg_command = [
            "ffmpeg",
            "-i",
            "pipe:0",
            "-ac",
            ac,
            "-ar",
            ar,
            "-f",
            format_for_conversion,
            "-hide_banner",
            "-loglevel",
            "quiet",
            "pipe:1",
        ]
        print("start")
        try:
            ffmpeg_process = subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        except FileNotFoundError:
            raise ValueError("ffmpeg was not found but is required to load audio files from filename")
        output_stream = ffmpeg_process.communicate(fileData)
        out_bytes = output_stream[0]

        audio = np.frombuffer(out_bytes, np.float32)
        if audio.shape[0] == 0:
            return False
            #raise ValueError("Malformed soundfile")
        print("stop")
        """
        return True
            
    class ProcessManager:
        def __init__(self):
            self.subprocesses = dict()
            def whisperthreadfunc(in_q):
                logger("LOADING_MODEL","")
                transcriber = Transcriber(BATCHSIZE,device)
                logger("LOADED_MODEL","")
                while True:
                    work = in_q.get()
                    if work == False:
                        return
                    else:
                        filename,language = work
                        filenameNoExt = os.path.splitext(filename)[0]
                        filepath = f"{OUTPUTDIR}\\{filenameNoExt}"
                        os.makedirs(filepath)
                        logger("STARTING_TRANSCRIPTION",f"{filename} - {language}")
                        transcriber.transcribe(f"{temppath}\\{filename}",language,f"{filepath}\\{filenameNoExt}.json")
                        with open(f"{filepath}\\{filenameNoExt}.json","r",encoding="utf-8") as json_file:
                            data = json.load(json_file)
                            with open(f"{filepath}\\{filenameNoExt}.txt", 'w',encoding="utf-8") as txtfile:
                                txtfile.write(data["text"])

                        for chunk in data["chunks"]:
                            chunk.pop("timestamp")
                        with open(f"{filepath}\\{filenameNoExt}.csv", 'w',encoding="utf-8") as csvfile:
                            fields = ["start","end","text"]
                            writer = csv.DictWriter(csvfile, fieldnames = fields)
                            # writing headers (field names)
                            writer.writeheader()    
                            # writing data rows
                            writer.writerows(data["chunks"])   

                        read_file = pd.read_csv(f"{filepath}\\{filenameNoExt}.csv")
                        read_file.to_excel(f"{filepath}\\{filenameNoExt}.xlsx", index=None, header=True)
                        os.remove(f"{filepath}\\{filenameNoExt}.csv")
                        logger("COMPLETE",f"{filename} - {language}")                                                             
                                               

            self.transcriptFileGenQueue = queue.Queue()
            self.whisperThread = threading.Thread(target=whisperthreadfunc,args=(self.transcriptFileGenQueue,))
            self.whisperThread.start()

        def addProcess(self,name,language,ffmpeg_command):
            self.subprocesses[psutil.Popen(ffmpeg_command)] = (name,language)

        def checkAudio(self,filename):
            filepath = f"{temppath}\\{filename}"
            if not os.path.exists(filepath):
                logger("UNSUPPORTED_AUDIO",filename)
            elif not (ffmpegCheckFile(filepath)):
                os.remove(filepath)
                logger("UNSUPPORTED_AUDIO",filename)
            else:
                return True
            return False
        

        def processComplete(self,process):
            extractedFileName,language = self.subprocesses.pop(process)
            logger("EXTRACTED",extractedFileName)
            if self.checkAudio(extractedFileName):
                logger("QUEUED",f"{extractedFileName} - {language}")
                self.transcriptFileGenQueue.put((extractedFileName,language))

        def wait(self):
            while True:
                gone, alive = psutil.wait_procs(self.subprocesses.keys(),callback=self.processComplete)
                return (len(self.subprocesses) != 0)

    processManager = ProcessManager()



    def hashletter(item):
        chars = "abcdefghijklmnopqrstufwxyz0123456789"
        start = str(abs(hash(item)))
        out = ""
        for i in range(0,len(start),3):
            idx = int(start[i:i+3])
            if idx >= len(chars):
                idx = idx- (idx//len(chars) * len(chars))
            out += chars[idx]
        return out
    
    def ffmpegExtractAudio(filepath,language):
        filename = os.path.basename(filepath).split('/')[-1]
        filename = os.path.splitext(filename)[0]
        pathhash = hashletter(filepath)
        ffmpeg_command = [
            "ffmpeg",
            "-i",
            filepath,
            "-vn",
            "-loglevel",
            "quiet",
            "-c:a",
            "flac",
            f"{temppath}\\{filename} -{pathhash}.flac",
        ]
        try:
            processManager.addProcess(f"{filename} -{pathhash}.flac",language,ffmpeg_command)
            #ffmpeg_process = subprocess.Popen(ffmpeg_command)
            
        except FileNotFoundError:
            raise ValueError("ffmpeg was not found but is required to load audio files from filename")



    if os.path.exists(temppath):
        shutil.rmtree(temppath)
    os.makedirs("temp")

    if os.path.exists(temppath):
        for filepath,language in FILES:
            if language == "Autodetect":
                language = None
            logger("EXTRACTING_AUDIO",filepath)
            ffmpegExtractAudio(filepath,language)

    while processManager.wait():
        pass
    processManager.transcriptFileGenQueue.put(False)
    processManager.whisperThread.join()

    logger("DONE","")

    shutil.rmtree(temppath)
