import subprocess
from PyQt6.QtCore import QThread

class VoiceDispatchThread(QThread):
    """Asynchronous background worker executing native Windows SAPI TTS without pip dependencies."""

    def __init__(self, text):
        super().__init__()
        self.text = text

    def run(self):
        try:
            # Escape quotes safely for PowerShell execution
            clean_text = self.text.replace("'", "")
            cmd = f'powershell -Command "Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak(\'{clean_text}\')"'
            subprocess.run(cmd, shell=True)
        except Exception:
            pass