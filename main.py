import numpy as np
import cv2
from PIL import Image, ImageEnhance
import librosa
import matplotlib.pyplot as plt
from scipy import signal
import colorsys
import os
from pathlib import Path
import argparse
import sys
from scipy.ndimage import gaussian_filter
import math

class AudioVisualDataBender:
    def __init__(self, image_path, audio_path=None):
        """
        Inițializează databender-ul cu o imagine și opțional un fișier audio
        """
        self.image_path = image_path
        self.audio_path = audio_path
        self.original_image = None
        self.audio_data = None
        self.sr = None
        self.load_image()
        if audio_path:
            self.load_audio()
    
    def load_image(self):
        """Încarcă imaginea și o convertește în format utilizabil"""
        try:
            self.original_image = cv2.imread(self.image_path)
            if self.original_image is None:
                raise ValueError(f"Nu pot încărca imaginea: {self.image_path}")
            self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            print(f"Imaginea încărcată: {self.original_image.shape}")
        except Exception as e:
            print(f"Eroare la încărcarea imaginii: {e}")
    
    def load_audio(self):
        """Încarcă fișierul audio și extrage caracteristicile"""
        try:
            self.audio_data, self.sr = librosa.load(self.audio_path, sr=None)
            print(f"Audio încărcat: {len(self.audio_data)} samples la {self.sr} Hz")
        except Exception as e:
            print(f"Eroare la încărcarea audio: {e}")
    
    def generate_synthetic_audio(self, duration=5.0, style='harmonics'):
        """
        Generează audio sintetic pentru experimente când nu avem fișier audio
        """
        t = np.linspace(0, duration, int(self.sr * duration) if self.sr else int(44100 * duration))
        
        if style == 'harmonics':
            # Generează armonici complexe
            frequencies = [220, 440, 660, 880]  # A3, A4, E5, A5
            audio = np.zeros_like(t)
            for i, freq in enumerate(frequencies):
                audio += np.sin(2 * np.pi * freq * t) * (0.5 ** i)
        
        elif style == 'noise':
            # Zgomot colorat
            audio = np.random.normal(0, 0.3, len(t))
            # Filtru pentru a face zgomotul mai plăcut
            b, a = signal.butter(4, 0.1, 'low')
            audio = signal.filtfilt(b, a, audio)
        
        elif style == 'sweep':
            # Sweep de frecvențe
            audio = signal.chirp(t, 100, duration, 2000)
        
        self.audio_data = audio
        self.sr = 44100 if not self.sr else self.sr
        print(f"Audio sintetic generat: {style}")
    
    def extract_audio_features(self, segment_length=1024):
        """
        Extrage caracteristici audio relevante pentru maparea vizuală
        """
        if self.audio_data is None:
            self.generate_synthetic_audio()
        
        # Spectrogramă
        stft = librosa.stft(self.audio_data, n_fft=segment_length)
        spectrogram = np.abs(stft)
        
        # Caracteristici temporale
        rms_energy = librosa.feature.rms(y=self.audio_data)[0]
        spectral_centroid = librosa.feature.spectral_centroid(y=self.audio_data, sr=self.sr)[0]
        zero_crossing_rate = librosa.feature.zero_crossing_rate(self.audio_data)[0]
        
        # Caracteristici harmonice
        pitches, magnitudes = librosa.piptrack(y=self.audio_data, sr=self.sr)
        
        # MFCC pentru textură
        mfcc = librosa.feature.mfcc(y=self.audio_data, sr=self.sr, n_mfcc=13)
        
        features = {
            'spectrogram': spectrogram,
            'rms_energy': rms_energy,
            'spectral_centroid': spectral_centroid,
            'zero_crossing_rate': zero_crossing_rate,
            'pitches': pitches,
            'magnitudes': magnitudes,
            'mfcc': mfcc,
            'raw_audio': self.audio_data
        }
        
        return features
    
    def map_frequency_to_color(self, frequencies, method='hsv'):
        """
        Mapează frecvențele audio la culori - baza sinesteziei
        """
        # Verifică dacă toate frecvențele sunt egale pentru a evita diviziunea la zero
        min_freq = np.min(frequencies)
        max_freq = np.max(frequencies)
        
        if max_freq == min_freq:
            # Dacă toate valorile sunt egale, rezultatul normalizării este 0
            freq_norm = np.zeros_like(frequencies, dtype=float)
        else:
            # Altfel, normalizează normal
            freq_norm = (frequencies - min_freq) / (max_freq - min_freq)

        
        if method == 'hsv':
            # Mapează frecvența la hue (culoare)
            hue = freq_norm * 0.8  # Evită roșu-violet overlap
            colors = np.array([colorsys.hsv_to_rgb(h, 1.0, 1.0) for h in hue])
        
        elif method == 'temperature':
            # Mapează ca temperatură de culoare
            colors = np.zeros((len(frequencies), 3))
            for i, f in enumerate(freq_norm):
                if f < 0.3:  # Frecvențe joase -> albastru-violet
                    colors[i] = [f*2, 0, 1]
                elif f < 0.7:  # Frecvențe medii -> verde-galben
                    colors[i] = [1, f, 0]
                else:  # Frecvențe înalte -> roșu-portocaliu
                    colors[i] = [1, 0.5, (1-f)*2]
        
        elif method == 'spectral':
            # Mapează ca spectru vizibil (violet la roșu)
            colors = plt.cm.plasma(freq_norm)[:, :3]
        
        return colors
    
    def apply_audio_to_image_transformation(self, method='frequency_bands'):
        """
        Aplică transformarea audio-vizuală principală
        """
        if self.original_image is None:
            raise ValueError("Nu există imagine încărcată")
        
        features = self.extract_audio_features()
        result_image = self.original_image.copy().astype(np.float32) / 255.0
        
        if method == 'frequency_bands':
            # Împarte imaginea în benzi orizontale, fiecare influențată de o bandă de frecvențe
            spectrogram = features['spectrogram']
            h, w, c = result_image.shape
            
            # Redimensionează spectrograma la înălțimea imaginii
            freq_bands = cv2.resize(spectrogram.T, (w, h))
            
            # Aplică culori bazate pe frecvențe
            for y in range(h):
                freq_slice = freq_bands[y, :]
                colors = self.map_frequency_to_color(freq_slice, 'hsv')
                color_influence = np.mean(colors, axis=0)
                
                # Amestecă culoarea originală cu influența audio
                blend_factor = 0.3
                result_image[y, :, :] = (1 - blend_factor) * result_image[y, :, :] + \
                                       blend_factor * color_influence
        
        elif method == 'energy_waves':
            # Folosește energia RMS pentru a crea unde în imagine
            rms = features['rms_energy']
            h, w, c = result_image.shape
            
            # Creează o hartă de undă bazată pe energie
            x = np.linspace(0, len(rms)-1, w)
            energy_wave = np.interp(x, np.arange(len(rms)), rms)
            
            for i in range(w):
                # Calculează deplasarea verticală bazată pe energie
                displacement = int(energy_wave[i] * h * 0.1)
                
                # Aplică deplasarea
                if displacement > 0:
                    result_image[:, i, :] = np.roll(result_image[:, i, :], displacement, axis=0)
        
        elif method == 'spectral_painting':
            # Folosește centrul spectral pentru a "picta" peste imagine
            centroid = features['spectral_centroid']
            h, w, c = result_image.shape

            # Verifică și aici pentru a preveni eroarea
            min_centroid = np.min(centroid)
            max_centroid = np.max(centroid)

            if max_centroid == min_centroid:
                # Dacă toate valorile sunt egale, nu face nimic sau atribuie o valoare constantă
                y_coords = np.full_like(centroid, fill_value=h//2, dtype=int)
            else:
                y_coords = ((centroid - min_centroid) / (max_centroid - min_centroid) * (h-1)).astype(int)

            # Mapează centrul spectral la coordonate în imagine
            x_coords = np.linspace(0, w-1, len(centroid)).astype(int)
            y_coords = ((centroid - np.min(centroid)) / (np.max(centroid) - np.min(centroid)) * (h-1)).astype(int)
            
            # Creează "pensule" colorate
            for i in range(len(x_coords)):
                x, y = x_coords[i], y_coords[i]
                color = self.map_frequency_to_color([centroid[i]], 'temperature')[0]
                
                # Aplică o pată de culoare
                brush_size = 5
                y_start = max(0, y - brush_size)
                y_end = min(h, y + brush_size)
                x_start = max(0, x - brush_size)
                x_end = min(w, x + brush_size)
                
                result_image[y_start:y_end, x_start:x_end, :] = \
                    0.7 * result_image[y_start:y_end, x_start:x_end, :] + 0.3 * color
        
        return np.clip(result_image * 255, 0, 255).astype(np.uint8)
    
    def create_synesthetic_variations(self, base_image):
        """
        Creează variații sinestezice din imaginea de bază
        """
        variations = {}
        
        if self.audio_data is None:
            print("Generez audio sintetic pentru demonstrație...")
            self.generate_synthetic_audio(style='harmonics')
        
        features = self.extract_audio_features()
        h, w, c = base_image.shape
        
        # Variația "Gustul Culorilor" - mapează frecvențele la "arome" vizuale
        taste_image = base_image.copy()
        spectrogram = features['spectrogram']
        low_freq = np.mean(spectrogram[:spectrogram.shape[0]//3, :], axis=0)
        mid_freq = np.mean(spectrogram[spectrogram.shape[0]//3:2*spectrogram.shape[0]//3, :], axis=0)
        high_freq = np.mean(spectrogram[2*spectrogram.shape[0]//3:, :], axis=0)
        
        # Aplică "aromele"
        for i in range(w):
            segment_idx = int(i * len(low_freq) / w)
            if segment_idx < len(low_freq):
                bitter = low_freq[segment_idx]
                sweet = mid_freq[segment_idx]
                sour = high_freq[segment_idx]
                
                # Normalizează
                total = bitter + sweet + sour + 1e-10
                bitter, sweet, sour = bitter/total, sweet/total, sour/total
                
                # Aplică culorile "gustului"
                taste_color = np.array([sour, sweet, bitter])  # RGB
                taste_image[:, i, :] = 0.6 * taste_image[:, i, :] + 0.4 * taste_color * 255
        
        variations['taste'] = taste_image
        
        # Variația "Temperatura Sunetului"
        temp_image = base_image.copy().astype(np.float32)
        rms_energy = features['rms_energy']
        
        # Energia înaltă = cald (roșu-galben)
        # Energia scăzută = rece (albastru-cyan)
        for i in range(w):
            energy_idx = int(i * len(rms_energy) / w)
            if energy_idx < len(rms_energy):
                energy = rms_energy[energy_idx]
                # Mapează la temperatură
                if energy > np.mean(rms_energy):
                    # Cald
                    temp_factor = (energy - np.mean(rms_energy)) / (np.max(rms_energy) - np.mean(rms_energy))
                    temp_image[:, i, 0] = np.minimum(temp_image[:, i, 0] * (1 + temp_factor), 255)
                    temp_image[:, i, 1] = np.minimum(temp_image[:, i, 1] * (1 + temp_factor*0.5), 255)
                else:
                    # Rece
                    temp_factor = (np.mean(rms_energy) - energy) / (np.mean(rms_energy) - np.min(rms_energy))
                    temp_image[:, i, 2] = np.minimum(temp_image[:, i, 2] * (1 + temp_factor), 255)
                    temp_image[:, i, 1] = np.minimum(temp_image[:, i, 1] * (1 + temp_factor*0.3), 255)
        
        variations['temperature'] = temp_image.astype(np.uint8)
        
        # Variația "Textura Timpului"
        texture_image = base_image.copy()
        mfcc = features['mfcc']
        
        # Folosește MFCC pentru a crea texturi
        texture_pattern = np.mean(mfcc, axis=0)
        texture_pattern = cv2.resize(texture_pattern.reshape(1, -1), (w, h))
        
        # Aplică textura ca displacement
        for y in range(h):
            for x in range(w):
                offset = int(texture_pattern[y, x] * 10)
                new_x = np.clip(x + offset, 0, w-1)
                texture_image[y, x, :] = base_image[y, new_x, :]
        
        variations['texture'] = texture_image
        
        # NOUL: Variația "Mișcarea Sunetului" - mapează dinamica audio la fluxuri vizuale
        motion_image = self.create_motion_synesthesia(base_image, features)
        variations['motion'] = motion_image
        
        # NOUL: Variația "Formele Armoniei" - mapează armonicele la geometrii
        geometry_image = self.create_geometric_synesthesia(base_image, features)
        variations['geometry'] = geometry_image
        
        # NOUL: Variația "Fractalii Ritmului" - mapează pattern-urile ritmice la fractali
        fractal_image = self.create_fractal_synesthesia(base_image, features)
        variations['fractal'] = fractal_image
        
        return variations
    
    def create_motion_synesthesia(self, base_image, features):
        """
        Creează sinestezia mișcării - transformă dinamica audio în fluxuri vizuale
        """
        h, w, c = base_image.shape
        motion_image = base_image.copy().astype(np.float32)
        
        # Extrage rata de schimbare a energiei (accelerația sonoră)
        rms_energy = features['rms_energy']
        energy_diff = np.diff(rms_energy)
        
        # Creează câmp vectorial bazat pe schimbările de energie
        for i in range(1, len(energy_diff)):
            if i < len(energy_diff) - 1:
                # Calculează "viteza" și "direcția" mișcării
                velocity = energy_diff[i] * 20  # amplificare
                direction = energy_diff[i-1] - energy_diff[i+1]  # derivata secundă
                
                # Mapează la coordonate în imagine
                x_pos = int(i * w / len(energy_diff))
                
                # Creează efectul de "curent"
                for y in range(h):
                    if 0 <= x_pos < w:
                        # Calculează offset-ul bazat pe poziția verticală
                        wave_offset = int(velocity * np.sin(y * 0.1 + direction))
                        new_x = np.clip(x_pos + wave_offset, 0, w-1)
                        
                        # Amestecă pixelii pentru efectul de flux
                        if abs(wave_offset) > 1:
                            motion_image[y, x_pos, :] = 0.7 * motion_image[y, x_pos, :] + \
                                                       0.3 * base_image[y, new_x, :]
        
        return motion_image.astype(np.uint8)
    
    def create_geometric_synesthesia(self, base_image, features):
        """
        Creează sinestezia geometrică - transformă armonicele în forme geometrice
        """
        h, w, c = base_image.shape
        geometry_image = base_image.copy()
        
        # Extrage pitch-urile și magnitudinile
        pitches = features['pitches']
        magnitudes = features['magnitudes']
        
        # Creează o mască pentru forme geometrice
        geometric_mask = np.zeros((h, w), dtype=np.float32)
        
        # Pentru fiecare frame temporal
        for t in range(min(pitches.shape[1], w//10)):
            # Găsește pitch-urile dominante
            frame_pitches = pitches[:, t]
            frame_magnitudes = magnitudes[:, t]
            
            # Selectează pitch-urile cu magnitudine mare
            strong_pitches = frame_pitches[frame_magnitudes > np.mean(frame_magnitudes)]
            
            if len(strong_pitches) > 0:
                # Calculează proprietățile geometrice
                base_freq = np.mean(strong_pitches[strong_pitches > 0])
                if base_freq > 0:
                    # Mapează frecvența la numărul de laturi (3-8 laturi)
                    sides = int(3 + (base_freq % 5))
                    
                    # Calculează centrul și raza
                    center_x = int(t * w / (pitches.shape[1] / 10))
                    center_y = int(h * (base_freq % 100) / 100)
                    radius = int(20 + (np.mean(frame_magnitudes) * 50))
                    
                    # Desenează poligonul
                    self.draw_polygon(geometric_mask, center_x, center_y, radius, sides)
        
        # Aplică masca geometrică
        geometric_mask = gaussian_filter(geometric_mask, sigma=2)
        geometric_mask = geometric_mask / np.max(geometric_mask) if np.max(geometric_mask) > 0 else geometric_mask
        
        # Creează efectul geometric
        for y in range(h):
            for x in range(w):
                if geometric_mask[y, x] > 0.1:
                    # Intensitatea geometriei influențează culoarea
                    intensity = geometric_mask[y, x]
                    # Rotește culorile bazat pe intensitate
                    geometry_image[y, x, :] = self.rotate_color(geometry_image[y, x, :], intensity * 180)
        
        return geometry_image
    
    def draw_polygon(self, mask, center_x, center_y, radius, sides):
        """
        Desenează un poligon regulat în mască
        """
        h, w = mask.shape
        
        for i in range(sides):
            angle1 = 2 * np.pi * i / sides
            angle2 = 2 * np.pi * (i + 1) / sides
            
            x1 = int(center_x + radius * np.cos(angle1))
            y1 = int(center_y + radius * np.sin(angle1))
            x2 = int(center_x + radius * np.cos(angle2))
            y2 = int(center_y + radius * np.sin(angle2))
            
            # Desenează linia între puncte
            if 0 <= x1 < w and 0 <= y1 < h and 0 <= x2 < w and 0 <= y2 < h:
                cv2.line(mask, (x1, y1), (x2, y2), 1.0, 2)
    
    def rotate_color(self, color, angle_deg):
        """
        Rotește culoarea în spațiul HSV
        """
        # Convertește la HSV
        rgb_norm = color.astype(np.float32) / 255.0
        hsv = colorsys.rgb_to_hsv(rgb_norm[0], rgb_norm[1], rgb_norm[2])
        
        # Rotește hue
        new_hue = (hsv[0] + angle_deg / 360.0) % 1.0
        
        # Convertește înapoi la RGB
        new_rgb = colorsys.hsv_to_rgb(new_hue, hsv[1], hsv[2])
        return np.array(new_rgb) * 255
    
    def create_fractal_synesthesia(self, base_image, features):
        """
        Creează sinestezia fractală - transformă pattern-urile ritmice în structuri fractale
        """
        h, w, c = base_image.shape
        fractal_image = base_image.copy()
        
        # Extrage pattern-urile ritmice din spectrogramă
        spectrogram = features['spectrogram']
        
        # Calculează auto-corelația pentru a găsi pattern-uri repetitive
        autocorr = np.zeros(spectrogram.shape[1])
        for i in range(spectrogram.shape[0]):
            freq_band = spectrogram[i, :]
            autocorr += np.correlate(freq_band, freq_band, mode='same')
        
        # Normalizează
        autocorr = autocorr / np.max(autocorr) if np.max(autocorr) > 0 else autocorr
        
        # Creează structuri fractale bazate pe auto-corelație
        fractal_mask = np.zeros((h, w), dtype=np.float32)
        
        for i in range(1, len(autocorr)):
            if autocorr[i] > 0.3:  # Prag pentru pattern-uri semnificative
                # Calculează parametrii fractalului
                x_center = int(i * w / len(autocorr))
                y_center = int(h * 0.5)
                complexity = autocorr[i]
                
                # Generează fractal simplu (Sierpinski-like)
                self.draw_fractal_pattern(fractal_mask, x_center, y_center, 
                                        int(complexity * 100), int(complexity * 5))
        
        # Aplică fractalii la imagine
        fractal_mask = gaussian_filter(fractal_mask, sigma=1)
        fractal_mask = fractal_mask / np.max(fractal_mask) if np.max(fractal_mask) > 0 else fractal_mask
        
        # Creează efectul fractal prin modularea intensității
        for y in range(h):
            for x in range(w):
                if fractal_mask[y, x] > 0.05:
                    intensity = fractal_mask[y, x]
                    # Amplifică contrastul bazat pe intensitatea fractală
                    fractal_image[y, x, :] = np.clip(
                        base_image[y, x, :] * (1 + intensity * 2), 0, 255
                    )
        
        return fractal_image
    
    def draw_fractal_pattern(self, mask, center_x, center_y, size, depth):
        """
        Desenează un pattern fractal simplu
        """
        h, w = mask.shape
        
        if depth <= 0 or size < 2:
            return
        
        # Desenează triunghiul central
        points = np.array([
            [center_x, center_y - size//2],
            [center_x - size//2, center_y + size//2],
            [center_x + size//2, center_y + size//2]
        ], dtype=np.int32)
        
        # Verifică boundurile
        if (0 <= center_x < w and 0 <= center_y < h and 
            size > 0 and all(0 <= p[0] < w and 0 <= p[1] < h for p in points)):
            cv2.fillPoly(mask, [points], 1.0)
        
        # Recursivitate pentru sub-pattern-uri
        if depth > 1:
            new_size = size // 2
            self.draw_fractal_pattern(mask, center_x, center_y - size//4, new_size, depth-1)
            self.draw_fractal_pattern(mask, center_x - size//4, center_y + size//4, new_size, depth-1)
            self.draw_fractal_pattern(mask, center_x + size//4, center_y + size//4, new_size, depth-1)
    
    def save_results(self, images_dict, output_dir='databending_results'):
        """
        Salvează rezultatele în directorul specificat
        """
        Path(output_dir).mkdir(exist_ok=True)
        
        for name, image in images_dict.items():
            output_path = os.path.join(output_dir, f'{name}_databend.png')
            cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            print(f"Salvat: {output_path}")
    
    def process_complete_workflow(self, methods=['frequency_bands', 'energy_waves', 'spectral_painting']):
        """
        Procesează workflow-ul complet de databending audio-vizual
        """
        print("=== Databending Audio-Vizual ===")
        print("1. Procesez transformările de bază...")
        
        results = {}
        
        for method in methods:
            print(f"   - Aplicând metoda: {method}")
            transformed = self.apply_audio_to_image_transformation(method)
            results[method] = transformed
        
        print("2. Generez variații sinestezice...")
        # Folosește prima transformare ca bază pentru variații
        base_image = results[methods[0]]
        synesthetic_variations = self.create_synesthetic_variations(base_image)
        results.update(synesthetic_variations)
        
        print("3. Salvez rezultatele...")
        self.save_results(results)
        
        print("✓ Procesare completă!")
        return results

def create_argument_parser():
    """
    Creează parser-ul pentru argumentele din linia de comandă
    """
    parser = argparse.ArgumentParser(
        description='Databending Audio-Vizual - Transformă imagini folosind caracteristici audio',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Exemple de utilizare:
  python databending.py -i imagine.jpg
  python databending.py -i imagine.jpg -a sunet.wav
  python databending.py -i imagine.jpg -a sunet.wav -m frequency_bands energy_waves
  python databending.py -i imagine.jpg -s harmonics -o rezultate/
  python databending.py -i imagine.jpg --synesthetic-only
  python databending.py -i imagine.jpg --list-methods
        '''
    )
    
    # Argumente obligatorii
    parser.add_argument('-i', '--image', 
                       required=True,
                       help='Calea către imaginea de intrare (JPG, PNG, etc.)')
    
    # Argumente opționale
    parser.add_argument('-a', '--audio',
                       help='Calea către fișierul audio (WAV, MP3, etc.)')
    
    parser.add_argument('-o', '--output',
                       default='databending_results',
                       help='Directorul pentru rezultate (default: databending_results)')
    
    parser.add_argument('-m', '--methods',
                       nargs='+',
                       choices=['frequency_bands', 'energy_waves', 'spectral_painting'],
                       default=['frequency_bands', 'energy_waves', 'spectral_painting'],
                       help='Metodele de transformare audio-vizuală')
    
    parser.add_argument('-s', '--synthetic-audio',
                       choices=['harmonics', 'noise', 'sweep'],
                       help='Generează audio sintetic dacă nu este furnizat fișier audio')
    
    parser.add_argument('--synesthetic-only',
                       action='store_true',
                       help='Generează doar variațiile sinestezice (nu și transformările de bază)')
    
    parser.add_argument('--list-methods',
                       action='store_true',
                       help='Afișează lista metodelor disponibile și iese')
    
    parser.add_argument('--duration',
                       type=float,
                       default=5.0,
                       help='Durata audio-ului sintetic în secunde (default: 5.0)')
    
    parser.add_argument('-v', '--verbose',
                       action='store_true',
                       help='Afișează informații detaliate despre procesare')
    
    return parser

def list_available_methods():
    """
    Afișează metodele disponibile
    """
    print("=== Metode de Transformare Audio-Vizuală ===")
    print("\nMetode de bază:")
    print("  frequency_bands    - Mapează benzile de frecvență la culori")
    print("  energy_waves       - Creează unde bazate pe energia audio")
    print("  spectral_painting  - 'Pictează' folosind centrul spectral")
    
    print("\nVariații Sinestezice:")
    print("  taste       - Mapează frecvențele la 'arome' vizuale")
    print("  temperature - Energia devine temperatură de culoare")
    print("  texture     - MFCC creează texturi temporale")
    print("  motion      - Dinamica audio devine fluxuri vizuale")
    print("  geometry    - Armonicele se transformă în forme geometrice")
    print("  fractal     - Pattern-urile ritmice creează structuri fractale")
    
    print("\nTipuri de Audio Sintetic:")
    print("  harmonics   - Armonici complexe (A3, A4, E5, A5)")
    print("  noise       - Zgomot colorat filtrat")
    print("  sweep       - Sweep de frecvențe (100Hz - 2kHz)")

def main():
    """
    Funcția principală pentru interfața în linia de comandă
    """
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Afișează metodele disponibile și iese
    if args.list_methods:
        list_available_methods()
        return
    
    # Verifică existența fișierului imagine
    if not os.path.exists(args.image):
        print(f"❌ Eroare: Imaginea '{args.image}' nu există!")
        return
    
    # Verifică existența fișierului audio (dacă este specificat)
    if args.audio and not os.path.exists(args.audio):
        print(f"❌ Eroare: Fișierul audio '{args.audio}' nu există!")
        return
    
    try:
        print("🎨 Inițializez Databending Audio-Vizual...")
        print(f"📸 Imagine: {args.image}")
        if args.audio:
            print(f"🎵 Audio: {args.audio}")
        else:
            print(f"🎵 Audio: Sintetic ({args.synthetic_audio or 'harmonics'})")
        
        # Inițializează databender-ul
        databender = AudioVisualDataBender(args.image, args.audio)
        
        # Generează audio sintetic dacă este necesar
        if not args.audio:
            style = args.synthetic_audio or 'harmonics'
            databender.generate_synthetic_audio(duration=args.duration, style=style)
        
        results = {}
        
        # Procesează transformările de bază (dacă nu este synesthetic-only)
        if not args.synesthetic_only:
            print("\n🔄 Procesez transformările de bază...")
            for method in args.methods:
                if args.verbose:
                    print(f"   ⚙️  Aplicând metoda: {method}")
                transformed = databender.apply_audio_to_image_transformation(method)
                results[method] = transformed
        
        # Generează variațiile sinestezice
        print("\n🧠 Generez variații sinestezice...")
        base_image = databender.original_image
        if results:
            # Folosește prima transformare ca bază
            base_image = list(results.values())[0]
        
        synesthetic_variations = databender.create_synesthetic_variations(base_image)
        results.update(synesthetic_variations)
        
        # Salvează rezultatele
        print(f"\n💾 Salvez rezultatele în '{args.output}'...")
        databender.save_results(results, args.output)
        
        # Afișează rezumatul
        print(f"\n✅ Procesare completă!")
        print(f"📁 Rezultate salvate: {len(results)} imagini în '{args.output}'")
        
        if args.verbose:
            print("\n📊 Imagini generate:")
            for name in results.keys():
                print(f"   🖼️  {name}_databend.png")
        
    except Exception as e:
        print(f"❌ Eroare: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

# Exemplu de utilizare
if __name__ == "__main__":
    if len(sys.argv) == 1:
        # Dacă nu sunt argumente, afișează help
        print("Databending Audio-Vizual - Transformă imagini folosind caracteristici audio")
        print("Utilizare: python databending.py -i imagine.jpg [opțiuni]")
        print("Pentru ajutor detaliat: python databending.py -h")
        print("Pentru lista metodelor: python databending.py --list-methods")
        sys.exit(0)
    
    main()
