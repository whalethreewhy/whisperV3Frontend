import torch
import sys
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *
import os
import json
import threading
from datetime import timedelta
import time

class ThreadSignaller(QObject):
    progress_changed = pyqtSignal(tuple)

class TranscribeThread():
    def __init__(self,signaller : ThreadSignaller) -> None:
        self.signaller = signaller
        self.worker = None
    def start(self,FILES,BATCHSIZE : int,DEVICE : str,OUTPUTDIR : str):
        if DEVICE == "cpu":
            BATCHSIZE = 1
        def work():
            import batchall
            def passSignal(header,msg):
                self.signaller.progress_changed.emit((header,msg))
                #print(header,msg)
            batchall.logger = passSignal
            batchall.transcribeFiles(FILES,BATCHSIZE,DEVICE,OUTPUTDIR)

        self.worker = threading.Thread(target=work)
        self.worker.daemon = True
        self.worker.start()

class MainWindow(QMainWindow):
    def __init__(self,onStartTranscribe):
        super().__init__()
        self.selectedFiles = []
        self.devices = {f"{torch.cuda.get_device_properties(i).name} - {i}" : f"cuda:{i}" for i in range(torch.cuda.device_count())}
        self.devices.update({"CPU":"cpu"})
        
        self.languages = [
                'English',
                #'Autodetect',
                'Malay',
                'Chinese',
                'Tamil',
                'Afrikaans',
                'Arabic',
                'Armenian',
                'Azerbaijani',
                'Belarusian',
                'Bosnian',
                'Bulgarian',
                'Catalan',
                'Croatian',
                'Czech',
                'Danish',
                'Dutch',
                'Estonian',
                'Finnish',
                'French',
                'Galician',
                'German',
                'Greek',
                'Hebrew',
                'Hindi',
                'Hungarian',
                'Icelandic',
                'Indonesian',
                'Italian',
                'Japanese',
                'Kannada',
                'Kazakh',
                'Korean',
                'Latvian',
                'Lithuanian',
                'Macedonian',
                'Marathi',
                'Maori',
                'Nepali',
                'Norwegian',
                'Persian',
                'Polish',
                'Portuguese',
                'Romanian',
                'Russian',
                'Serbian',
                'Slovak',
                'Slovenian',
                'Spanish',
                'Swahili',
                'Swedish',
                'Tagalog',
                'Thai',
                'Turkish',
                'Ukrainian',
                'Urdu',
                'Vietnamese',
                'Welsh'
            ]
        
        window = QWidget()
        self.setCentralWidget(window)
        
        layout = QVBoxLayout()
        window.setLayout(layout)

        fileSelection_GroupBox = QGroupBox()
        fileSelection_GroupBox.setSizePolicy(QSizePolicy.Policy.MinimumExpanding,QSizePolicy.Policy.Maximum)
        fileSelection_GroupBox.setTitle("File Selection")
        layout.addWidget(fileSelection_GroupBox)

        fileSelection_Layout = QVBoxLayout()
        fileSelection_Layout.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        fileSelection_GroupBox.setLayout(fileSelection_Layout)
    


        self.selectedFiles_List = QListWidget()
        fileSelection_Layout.addWidget(self.selectedFiles_List)
        
        self.selectedFiles_Label = QLabel("0 Files Selected")
        self.selectedFiles_Label.setSizePolicy(QSizePolicy.Policy.Maximum,QSizePolicy.Policy.Maximum)
        fileSelection_Layout.addWidget(self.selectedFiles_Label)

        selectFiles_Button = QPushButton("Select Files")
        selectFiles_Button.setSizePolicy(QSizePolicy.Policy.Fixed,QSizePolicy.Policy.Fixed)
        selectFiles_Button.clicked.connect(self.openMultiFileDialog)

        clearSelected_Button = QPushButton("Clear Selected")
        clearSelected_Button.setSizePolicy(QSizePolicy.Policy.Fixed,QSizePolicy.Policy.Fixed)
        clearSelected_Button.clicked.connect(self.clearSelectedFiles)


        fileSelection_Layout.addLayout(self.create_horizontal_layout(selectFiles_Button,clearSelected_Button))

        self.sameLanguage_CheckBox = QCheckBox()
        self.sameLanguage_CheckBox.setText("Apply Language to All")

        self.sameLanguage_ComboBox = QComboBox()
        self.sameLanguage_ComboBox.addItems(self.languages)
        self.sameLanguage_ComboBox.setDisabled(True)

        def onSameLanguageToggle(state):
            self.updateSelectedFiles_Widgets()
            self.sameLanguage_ComboBox.setDisabled(not state)


        self.sameLanguage_CheckBox.toggled.connect(onSameLanguageToggle)

        fileSelection_Layout.addLayout(self.create_horizontal_layout(self.sameLanguage_CheckBox,self.sameLanguage_ComboBox))



        
        options_GroupBox = QGroupBox()
        options_GroupBox.setSizePolicy(QSizePolicy.Policy.MinimumExpanding,QSizePolicy.Policy.Maximum)
        options_GroupBox.setTitle("Advanced Options")
        layout.addWidget(options_GroupBox)

        options_Layout = QVBoxLayout()
        options_Layout.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        options_GroupBox.setLayout(options_Layout)

        warning_Label = QLabel("WARNING: These settings vary based on hardware, may crash the transcriber if exceeds hardware capacity.")
        warning_Label.setSizePolicy(QSizePolicy.Policy.Maximum,QSizePolicy.Policy.Maximum)
        options_Layout.addWidget(warning_Label)

        device_Label = QLabel("Device:")
        device_Label.setSizePolicy(QSizePolicy.Policy.Maximum,QSizePolicy.Policy.Maximum)
        self.device_ComboBox = QComboBox()
        self.device_ComboBox.addItems(self.devices.keys())
        
        if len(self.devices) < 2:
            self.device_ComboBox.setDisabled(True)
            
        options_Layout.addLayout(self.create_horizontal_layout(device_Label,self.device_ComboBox))
        
        batchSize_Widget = QWidget()
        if self.device_ComboBox.currentText() == "CPU":
            batchSize_Widget.setHidden(True)
        options_Layout.addWidget(batchSize_Widget)
        batchSize_Layout = QVBoxLayout()
        batchSize_Widget.setLayout(batchSize_Layout)

        batchSize_Label = QLabel("Batch Size:")
        batchSize_Label.setSizePolicy(QSizePolicy.Policy.Maximum,QSizePolicy.Policy.Maximum)
        batchSize_Layout.addWidget(batchSize_Label)

        batchSize_IntValidator = QIntValidator()
        batchSize_IntValidator.setRange(1,64)

        self.batchSize_LineEdit = QLineEdit()
        self.batchSize_LineEdit.setSizePolicy(QSizePolicy.Policy.Maximum,QSizePolicy.Policy.Maximum)
        self.batchSize_LineEdit.setValidator(batchSize_IntValidator)
        self.batchSize_LineEdit.setText("4")
        self.batchSize_LineEdit.textChanged.connect(lambda: self.batchSize_LineEdit.setPalette(QLineEdit().palette()))
        #batchSize_Layout.addWidget(self.batchSize_LineEdit)

        self.testBatchSize_Button = QPushButton("Autodetermine Batch Size")
        self.testBatchSize_Button.setHidden(True)
        batchSize_Layout.addLayout(self.create_horizontal_layout(self.batchSize_LineEdit,self.testBatchSize_Button))

        baseFont = QLabel().font()
        baseFont.setPointSizeF(baseFont.pointSizeF()*4/5)
        batchSizeInfo_Label = QLabel("(Increasing batch size requires more VRAM but may speed up transcription, reduce if encountering Out Of Memory errors)\nRecommended Setting:\n8GB VRAM: BatchSize 8\t6GB VRAM: BatchSize 6\tBelow 6GB VRAM: BatchSize 1")
        batchSizeInfo_Label.setFont(baseFont)
        batchSizeInfo_Label.setSizePolicy(QSizePolicy.Policy.Maximum,QSizePolicy.Policy.Maximum)
        batchSize_Layout.addWidget(batchSizeInfo_Label)

        

        self.device_ComboBox.currentTextChanged.connect(lambda device: batchSize_Widget.setHidden(True) if device=="CPU" else batchSize_Widget.setHidden(False))

        self.outputDir_LineEdit = QLineEdit()
        self.outputDir_LineEdit.textChanged.connect(lambda: self.outputDir_LineEdit.setPalette(QLineEdit().palette()))

        #self.outputDir_LineEdit.setSizePolicy(QSizePolicy.Policy.Maximum,QSizePolicy.Policy.Maximum)
        selectOutputDir_Button = QPushButton("Select Output Folder")
        selectOutputDir_Button.clicked.connect(self.openOutputFileDialog)
        layout.addLayout(self.create_horizontal_layout(self.outputDir_LineEdit,selectOutputDir_Button))



     
        transcribe_Button = QPushButton("Transcribe")
        buttonfont = transcribe_Button.font()
        buttonfont.setBold(True)
        transcribe_Button.setFont(buttonfont)
        palette = transcribe_Button.palette()
        

        palette.setColor(QPalette.ColorRole.ButtonText,QColor('#278048'))

        #transcribe_Button.setStyleSheet("background-color: #278048;")
        transcribe_Button.setPalette(palette)

        def onTranscribeClicked():
            allFieldsValid = True
            palette = QLineEdit().palette()
            
            if not os.path.exists(self.outputDir_LineEdit.text()):
                palette.setColor(QPalette.ColorRole.Base,QColor(255, 0, 0,50))
                allFieldsValid = False
            self.outputDir_LineEdit.setPalette(palette)

            palette = QLineEdit().palette()
            try:
                batchsize = int(self.batchSize_LineEdit.text())
                if batchsize < 1:
                    raise Exception
            except Exception:
                palette.setColor(QPalette.ColorRole.Base,QColor(255, 0, 0,50))
                allFieldsValid = False
            self.batchSize_LineEdit.setPalette(palette)

            palette = QListWidget().palette()
            if len(self.selectedFiles) == 0:
                palette.setColor(QPalette.ColorRole.Base,QColor(255, 0, 0,50))
                allFieldsValid = False
            self.selectedFiles_List.setPalette(palette)

            if allFieldsValid:
                if self.sameLanguage_CheckBox.isChecked():
                    language = self.sameLanguage_ComboBox.currentText()
                    for i in range(len(self.selectedFiles)):
                        self.selectedFiles[i][1] = language
                onStartTranscribe(self.selectedFiles,batchsize,self.devices[self.device_ComboBox.currentText()],self.outputDir_LineEdit.text())

            


        transcribe_Button.clicked.connect(onTranscribeClicked)
        layout.addWidget(transcribe_Button,alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom)
 
        try:
            if os.path.exists("settings.json"):
                with open("settings.json", "r", encoding="utf8") as fp:
                    settings = json.load(fp)
                    self.sameLanguage_CheckBox.setChecked(settings['islanguageoverride'])
                    if settings['languageoverride'] in self.languages:
                        self.sameLanguage_ComboBox.setCurrentText(settings['languageoverride'])
                    if settings['device'] in self.devices:
                        self.device_ComboBox.setCurrentText(settings['device'])
                    self.batchSize_LineEdit.setText(settings['batchsize'])
                    self.outputDir_LineEdit.setText(settings['outputdir'])
        except Exception as e:
            print(e) 
            pass
        

    def create_horizontal_layout(self,*widgets):
        layout = QHBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        for widget in widgets:
            layout.addWidget(widget)
        return layout
    
    def resizeEvent(self, event):
        super().resizeEvent(event)  # Call the parent's resize event
        self.selectedFiles_List.setFixedHeight(int(self.height() / 4))

    def clearSelectedFiles(self):
        self.selectedFiles = []
        self.updateSelectedFiles_Widgets()

    def removeSelectedFile(self,idx):
        self.selectedFiles.pop(idx)
        self.updateSelectedFiles_Widgets()

    def updateLanguageChoice(self,idx,language):
        self.selectedFiles[idx][1] = language

    def updateSelectedFiles_Widgets(self):
        
        self.selectedFiles_Label.setText(f"{len(self.selectedFiles)} File{'s' if len(self.selectedFiles) != 1 else ''} Selected")
        self.selectedFiles_List.clear()
        self.selectedFiles_List.setPalette(QListWidget().palette())
        for i in range(len(self.selectedFiles)):
            item = QListWidgetItem()
            
            item_widget = QWidget()
            item_widget.setObjectName(str(i))

            item_layout = QHBoxLayout()
            item_layout.setContentsMargins(0,0,0,0)
            item_widget.setLayout(item_layout)
            item_layout.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)


            line_cancel_button = QToolButton()
            line_cancel_button.setText("❌")
            line_cancel_button.clicked.connect(lambda state, x=i: self.removeSelectedFile(x))
            item_layout.addWidget(line_cancel_button)
            
            if not self.sameLanguage_CheckBox.isChecked():
                line_language_combobox = QComboBox()
                line_language_combobox.addItems(self.languages)
                line_language_combobox.setCurrentText(self.selectedFiles[i][1])
                line_language_combobox.currentTextChanged.connect(lambda state,x=i: self.updateLanguageChoice(x,state))
                #line_language_combobox.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                item_layout.addWidget(line_language_combobox,alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)


            
            line_text = QLabel(self.selectedFiles[i][0])
            item_layout.addWidget(line_text,alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

            
            item.setSizeHint(item_widget.sizeHint())
            self.selectedFiles_List.addItem(item)
            self.selectedFiles_List.setItemWidget(item, item_widget)

    def openOutputFileDialog(self):
        dlg = QFileDialog()
        #dlg.setAcceptMode(QFileDialog.AcceptMode.AcceptOpen)
        dlg.setFileMode(QFileDialog.FileMode.Directory)

        if dlg.exec():
            self.outputDir_LineEdit.setText(dlg.selectedFiles()[0])

    def openMultiFileDialog(self):
        dlg = QFileDialog()
        dlg.setAcceptMode(QFileDialog.AcceptMode.AcceptOpen)
        dlg.setFileMode(QFileDialog.FileMode.ExistingFiles)

        if dlg.exec():
            filenames = dlg.selectedFiles()
            
            self.selectedFiles.extend([[filename,"English"] for filename in filter(lambda filename: filename not in self.selectedFiles,filenames)])
            self.updateSelectedFiles_Widgets()
            
class TestWindow(QMainWindow):
    def __init__(self,mainPage):
        super().__init__()
        self.mainPage = mainPage

        window = QWidget()
        self.setCentralWidget(window)
        layout = QVBoxLayout()
        window.setLayout(layout)
        
        self.back_Button = QPushButton("Back")
        layout.addWidget(self.back_Button,alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom)

class RunWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.signaller = ThreadSignaller()
        self.remainingFiles = []
        self.initialStartTime = time.time()
        self.currentStartTime = time.time()
        window = QWidget()
        self.setCentralWidget(window)
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        window.setLayout(layout)

        totalTime_Label = QLabel("Total Elapsed Time:")
        totalTime_Label.setSizePolicy(QSizePolicy.Policy.Maximum,QSizePolicy.Policy.Maximum)
        layout.addWidget(totalTime_Label)

        transcriptionQueue_GroupBox = QGroupBox()
        transcriptionQueue_GroupBox.setSizePolicy(QSizePolicy.Policy.MinimumExpanding,QSizePolicy.Policy.Maximum)
        transcriptionQueue_GroupBox.setTitle("Transcription Queue")
        layout.addWidget(transcriptionQueue_GroupBox,alignment=Qt.AlignmentFlag.AlignTop)

        transcriptionQueue_Layout = QVBoxLayout()
        transcriptionQueue_Layout.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        transcriptionQueue_GroupBox.setLayout(transcriptionQueue_Layout)
    


        self.transcriptionQueue_List = QListWidget()
        transcriptionQueue_Layout.addWidget(self.transcriptionQueue_List)
        
        self.numRemaining_Label = QLabel("0 Files Remaining")
        self.numRemaining_Label.setSizePolicy(QSizePolicy.Policy.Maximum,QSizePolicy.Policy.Maximum)
        transcriptionQueue_Layout.addWidget(self.numRemaining_Label)
        


        self.currentFile_GroupBox = QGroupBox()
        self.currentFile_GroupBox.setHidden(True)
        self.currentFile_GroupBox.setSizePolicy(QSizePolicy.Policy.MinimumExpanding,QSizePolicy.Policy.Maximum)
        self.currentFile_GroupBox.setTitle("Currently Transcribing")
        layout.addWidget(self.currentFile_GroupBox,alignment=Qt.AlignmentFlag.AlignTop)

        currentFile_Layout = QVBoxLayout()
        currentFile_Layout.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.currentFile_GroupBox.setLayout(currentFile_Layout)

        self.currentFileName_LineEdit = QLineEdit()
        self.currentFileName_LineEdit.setDisabled(True)
        currentFile_Layout.addWidget(self.currentFileName_LineEdit)

        self.currentFile_ProgressBar = QProgressBar()
        self.currentFile_ProgressBar.setOrientation(Qt.Orientation.Horizontal)
        
        self.currentFile_ProgressBar.setValue(0)
        currentFile_Layout.addWidget(self.currentFile_ProgressBar)

        currentTime_Label = QLabel("Elapsed Time:")
        currentTime_Label.setSizePolicy(QSizePolicy.Policy.Maximum,QSizePolicy.Policy.Maximum)
        currentFile_Layout.addWidget(currentTime_Label)


        log_Label = QLabel("Log:")
        log_Label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        

        log_Label.setSizePolicy(QSizePolicy.Policy.Maximum,QSizePolicy.Policy.Maximum)
        layout.addWidget(log_Label)

        self.log_List = QListWidget()
        self.log_List.setFont(QFont("Consolas"))
        layout.addWidget(self.log_List)

        self.done_Label = QLabel("Transcription Complete")
        font = self.done_Label.font()
        font.setBold(True)
        font.setPointSize(int(font.pointSize()*1.5))
        self.done_Label.setFont(font)
        self.done_Label.setStyleSheet("color: #278048")

        self.done_Label.setSizePolicy(QSizePolicy.Policy.Maximum,QSizePolicy.Policy.Maximum)
        self.done_Label.setHidden(True)
        layout.addWidget(self.done_Label,alignment=Qt.AlignmentFlag.AlignHCenter)

        self.timer = QTimer()
        def clockUpdate():
            currentTime = time.time()
            totalTime_Label.setText(f"Total Elapsed Time: {timedelta(seconds=int(currentTime-self.initialStartTime))}")
            currentTime_Label.setText(f"Elapsed Time: {timedelta(seconds=int(currentTime-self.currentStartTime))}")
        self.timer.timeout.connect(clockUpdate)

        def signalProcessor(header,msg):
            #print(header,msg)
            logmsg = '{:<25s} {:<15s}'.format(f"[{header}]", str(msg))
            print(logmsg)

            showmsg = True

            try:
                
                if header == 'QUEUED':
                    self.remainingFiles.append(msg)
                    self.updateRemainingFiles()
                elif header == 'COMPLETE':
                    self.remainingFiles.remove(msg)
                    self.updateRemainingFiles()
                    self.currentFile_GroupBox.setHidden(True)
                elif header == 'STARTING_TRANSCRIPTION':
                    self.currentStartTime = time.time()
                    self.currentFileName_LineEdit.setText(msg)
                    self.currentFile_GroupBox.setHidden(False)
                elif header == 'DONE':
                    self.timer.stop()
                    self.done_Label.setHidden(False)
                elif header == 'TOTALSTEPS':
                    self.currentFile_ProgressBar.setValue(0)
                    self.currentFile_ProgressBar.setRange(0,msg)
                    showmsg = False
                elif header == 'STEP':
                    self.currentFile_ProgressBar.setValue(msg)
                    showmsg = False
                if showmsg:
                    self.log_List.addItem(logmsg)
            except Exception as e:
                print(e)
            
                

        self.signaller.progress_changed.connect(lambda data: signalProcessor(data[0],data[1]))
    
    def updateRemainingFiles(self):
        self.numRemaining_Label.setText(f"{len(self.remainingFiles)} File{'s' if len(self.remainingFiles) != 1 else ''} Remaining")
        self.transcriptionQueue_List.clear()
        self.transcriptionQueue_List.addItems([f"{i+1}. {self.remainingFiles[i]}" for i in range(len(self.remainingFiles))])

    def resizeEvent(self, event):
        super().resizeEvent(event)  # Call the parent's resize event
        self.transcriptionQueue_List.setFixedHeight(int(self.height() / 4))

if __name__ == "__main__":
    app = QApplication(sys.argv)

    pagewidget = QStackedWidget()
    pagewidget.setWindowTitle("Transcription")
    pagewidget.resize(720,600)

    runPage = RunWindow()
    pagewidget.addWidget(runPage)

    transcribeThread = TranscribeThread(runPage.signaller)

    def startTranscribe(*args):
        pagewidget.setWindowTitle("Transcribing...")
        pagewidget.setCurrentWidget(runPage)
        runPage.initialStartTime = time.time()
        runPage.timer.start(850)
        transcribeThread.start(*args)

    mainPage = MainWindow(startTranscribe)
    pagewidget.addWidget(mainPage)

    testPage = TestWindow(mainPage)
    pagewidget.addWidget(testPage)

    mainPage.testBatchSize_Button.clicked.connect(lambda : pagewidget.setCurrentWidget(testPage))
    testPage.back_Button.clicked.connect(lambda : pagewidget.setCurrentWidget(mainPage))

    pagewidget.setCurrentWidget(mainPage)
    
    pagewidget.show()
    print("Loaded")
    app.exec()
    with open("settings.json", "w", encoding="utf8") as fp:
        settings = {
            "islanguageoverride": mainPage.sameLanguage_CheckBox.isChecked(),
            "languageoverride": mainPage.sameLanguage_ComboBox.currentText(),
            "device": mainPage.device_ComboBox.currentText(),
            "batchsize": mainPage.batchSize_LineEdit.text(),
            "outputdir": mainPage.outputDir_LineEdit.text()
        }
        json.dump(settings, fp, ensure_ascii=False)


