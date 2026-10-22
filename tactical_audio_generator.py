"""
GeoGuard Edge Tactical Audio Dispatch & Radio Squelch Generator
Supports English, French, Spanish, and Hindi with tactical radio chirps.
"""

import os
import tempfile
import threading
import time
import numpy as np
import sounddevice as sd
import pyttsx3
from gtts import gTTS
import soundfile as sf


class TacticalAudioDispatcher:
    """Manages audio squelch sound effects and multilingual vocal dispatch."""

    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.is_playing = False

    def generate_chirp_squelch(self, duration_sec: float = 0.12) -> np.ndarray:
        """Generates a high-frequency walkie-talkie chirp/squelch burst."""
        t = np.linspace(0, duration_sec, int(self.sample_rate * duration_sec), endpoint=False)
        freq_sweep = np.linspace(1600.0, 2600.0, len(t))
        chirp = 0.35 * np.sin(2 * np.pi * freq_sweep * t)
        noise = 0.12 * (np.random.rand(len(t)) * 2.0 - 1.0)
        burst = (chirp + noise).astype(np.float32)
        envelope = np.exp(-t * 22)
        return burst * envelope

    def play_sound_burst(self):
        """Plays the radio activation squelch tone."""
        squelch = self.generate_chirp_squelch(0.14)
        sd.play(squelch, self.sample_rate)
        sd.wait()

    def format_nato_dispatch(
        self, 
        scenario: str, 
        severity: float, 
        sector: str = "CHARLIE", 
        lang: str = "English",
        dispatched_assets: list = None
    ) -> str:
        """
        Generates formal military/incident command radio broadcast syntax.
        Incorporates telemetry, structural triage, and automated dispatch orders.
        """
        pct = int(severity * 100)
        lang_lower = lang.lower()
        incident_code = f"TANGO-{int(severity * 89 + 10)}"

        # Default asset string if none supplied
        asset_summary_en = "Zodiac rapid units and aerial recon UAVs"
        if dispatched_assets and len(dispatched_assets) > 0:
            asset_summary_en = ", ".join([a.get('asset_name', 'Rescue Asset') for a in dispatched_assets[:2]])

        if "french" in lang_lower:
            tier = "CODE ROUGE // URGENCE ABSOLUE" if severity > 0.70 else "CODE ORANGE // VIGILANCE ACCRUE"
            return (
                f"Ici le Poste de Commandement Tactique GeoGuard. {tier}. "
                f"Incident matricule {incident_code}. "
                f"Nature du sinistre : {scenario}. "
                f"Secteur opérationnel {sector}, intégrité structurelle compromise à {pct} pour cent. "
                f"Avis à la population et aux secours : couloir d'évacuation principal ouvert. "
                f"Unités d'intervention d'urgence déployées sur zone. "
                f"Maintenez le silence radio sur ce canal tactique. Terminé."
            )

        elif "spanish" in lang_lower:
            tier = "CÓDIGO ROJO // EMERGENCIA OPERACIONAL MÁXIMA" if severity > 0.70 else "CÓDIGO ÁMBAR // ALERTA TÁCTICA"
            return (
                f"Atención a todas las unidades. Aquí el Puesto de Mando GeoGuard. {tier}. "
                f"Incidente táctico {incident_code}. "
                f"Peligro registrado : {scenario}. "
                f"Sector {sector} bajo compromiso estructural severo al {pct} por ciento. "
                f"Líneas de evacuación autorizadas y transmitidas a las unidades de campo. "
                f"Recursos de rescate anfibio y aéreo en fase de aproximación. "
                f"Mantengan la frecuencia despejada. Cambio."
            )

        elif "hindi" in lang_lower:
            tier = "कोड रेड // आपातकालीन सामरिक चेतावनी" if severity > 0.70 else "कोड एम्बर // परिचालन अलर्ट"
            return (
                f"सभी फील्ड यूनिट्स और राहत दल ध्यान दें। यह जियो गार्ड कमांड मुख्यालय है। {tier}। "
                f"आपदा नियंत्रण कोड {incident_code}। "
                f"आपदा स्वरूप : {scenario}। "
                f"सेक्टर {sector} में ढांचागत क्षति और जलभराव का स्तर {pct} प्रतिशत दर्ज किया गया है। "
                f"सुरक्षित निकासी गलियारा निर्धारित कर दिया गया है। "
                f"स्वायत्त त्वरित राहत एवं बचाव इकाइयों को सेक्टर में तुरंत रवाना किया जा रहा है। "
                f"रेडियो फ्रीक्वेंसी खाली रखें। ओवर।"
            )

        else:
            # Military-grade Standard English Incident Command Broadcast
            tier = "DEFCON 2 // FLASH OPERATIONAL EMERGENCY" if severity > 0.70 else "PRIORITY TWO // TACTICAL ALERT"
            return (
                f"Attention all forward operating units. This is GeoGuard Tactical Incident Command. {tier}. "
                f"Incident authorization: {incident_code}. "
                f"Threat profile: {scenario.upper()}. "
                f"Sector {sector} indicates critical inundation with structural compromise index at {pct} percent. "
                f"Primary evacuation corridor active. "
                f"Autonomous task force mobilization: {asset_summary_en} deployed to grid coordinates. "
                f"Enforce immediate perimeter containment. Maintain operational radio silence. Out."
            )

    def _broadcast_thread(self, text_script: str, lang: str = "English"):
        """Plays squelch + vocal broadcast across native SAPI5 or gTTS fallback."""
        try:
            # 1. Play opening walkie-talkie chirp
            self.play_sound_burst()
            time.sleep(0.05)

            lang_lower = lang.lower()

            # For Hindi, or if non-English language voice is missing, gTTS delivers natural speech
            if "hindi" in lang_lower or "french" in lang_lower or "spanish" in lang_lower:
                lang_tag = "hi" if "hindi" in lang_lower else ("fr" if "french" in lang_lower else "es")
                try:
                    tts = gTTS(text=text_script, lang=lang_tag, slow=False)
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
                        temp_path = fp.name
                    # Save as mp3/wav and play through soundfile
                    temp_mp3 = temp_path.replace(".wav", ".mp3")
                    tts.save(temp_mp3)

                    data, fs = sf.read(temp_mp3)
                    sd.play(data, fs)
                    sd.wait()

                    if os.path.exists(temp_mp3):
                        os.remove(temp_mp3)
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                except Exception:
                    # Fallback to pyttsx3 if offline
                    self._speak_pyttsx3(text_script, lang_lower)
            else:
                self._speak_pyttsx3(text_script, lang_lower)

            # 2. Play closing walkie-talkie chirp
            time.sleep(0.05)
            self.play_sound_burst()

        except Exception as e:
            print(f"[TACTICAL AUDIO ERROR] {e}")
        finally:
            self.is_playing = False

    def _speak_pyttsx3(self, text_script: str, lang_lower: str):
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.setProperty('volume', 1.0)
        voices = engine.getProperty('voices')
        if voices:
            for v in voices:
                v_name = (v.id + " " + v.name).lower()
                if "hindi" in lang_lower and any(k in v_name for k in ["hi", "hin", "kalpana", "hemant"]):
                    engine.setProperty('voice', v.id)
                    break
        engine.say(text_script)
        engine.runAndWait()
        engine.stop()

    def dispatch_vocal_alert(self, text_script: str, lang: str = "English"):
        """Fires asynchronous audio alert without freezing PyQt GUI."""
        if self.is_playing:
            return
        self.is_playing = True
        thread = threading.Thread(target=self._broadcast_thread, args=(text_script, lang), daemon=True)
        thread.start()