# 🎨 Audio-Visual Databender

Un instrument creativ în Python care transformă imagini statice folosind caracteristici extrase din fișiere audio, generând lucrări de artă vizuală unice, inspirate de conceptul de sinestezie.


---

## ✨ Ce este Databending-ul Audio-Vizual?

**Databending-ul** este procesul de manipulare a datelor unui fișier pentru a crea efecte vizuale sau auditive neintenționate. Acest script duce conceptul mai departe, realizând un **"cross-bend"**: nu corupe datele imaginii direct, ci folosește datele dintr-un fișier **audio** pentru a dicta cum să fie transformată imaginea.

Rezultatul este o fuziune artistică în care ritmul, frecvența și dinamica unui sunet "pictează" peste o imagine, creând o reprezentare vizuală a experienței auditive—un fel de **sinestezie artificială**.

---

## 🚀 Caracteristici

- **Transformări multiple**: Aplică diverse tehnici de bază pentru a altera imaginea:
    
    - `frequency_bands`: Mapează benzile de frecvență audio la culori în imagine.
        
    - `energy_waves`: Creează distorsiuni ondulatorii bazate pe energia (RMS) a sunetului.
        
    - `spectral_painting`: Utilizează centrul spectral pentru a "picta" tușe de culoare pe imagine.
        
- **Variații sinestezice avansate**: Generează interpretări artistice complexe care imită diferite forme de sinestezie:
    
    - **Gustul Culorilor (`taste`)**: Asociază frecvențele joase, medii și înalte cu arome vizuale (acru, dulce, amar).
        
    - **Temperatura Sunetului (`temperature`)**: Modifică "temperatura" culorilor imaginii în funcție de energia sunetului (sunete puternice = culori calde, sunete line = culori reci).
        
    - **Textura Timpului (`texture`)**: Folosește coeficienții cepstrali (MFCC) pentru a crea distorsiuni texturale subtile.
        
    - **Mișcarea Sunetului (`motion`)**: Transformă dinamica audio în fluxuri vizuale, creând efecte de mișcare.
        
    - **Formele Armoniei (`geometry`)**: Desenează forme geometrice (poligoane) bazate pe armonicile detectate în sunet.
        
    - **Fractalii Ritmului (`fractal`)**: Generează structuri fractale inspirate de pattern-urile ritmice repetitive din audio.
        
- **Generator de audio sintetic**: Nu ai un fișier audio? Nicio problemă! Scriptul poate genera sunete (`harmonics`, `noise`, `sweep`) pentru a experimenta.
    
- **Interfață în linia de comandă (CLI)**: Flexibilitate maximă datorită argumentelor personalizabile.
    

---

## 🛠️ Instalare

1. **Clonează repository-ul:**
    
    Bash
    
    ```
    git clone https://github.com/your-username/your-repository-name.git
    cd your-repository-name
    ```
    
2. **Creează un mediu virtual (recomandat):**
    
    Bash
    
    ```
    python -m venv venv
    source venv/bin/activate  # Pe Windows: venv\Scripts\activate
    ```
    
3. **Instalează dependențele:** Scriptul se bazează pe câteva librării puternice. Instalează-le folosind `pip`:
    
    Bash
    
    ```
    pip install numpy opencv-python Pillow librosa matplotlib scipy
    ```
    
    _Notă: `librosa` necesită `ffmpeg` pentru a încărca anumite formate audio (ex: MP3). Asigură-te că ai [ffmpeg](https://ffmpeg.org/download.html) instalat și adăugat în PATH-ul sistemului._
    

---

## ⚙️ Utilizare

Folosește scriptul direct din linia de comandă. Iată câteva exemple:

#### **Utilizare de bază**

Transformă o imagine folosind un fișier audio și salvează toate variațiile posibile.

Bash

```
python main.py -i imagini/peisaj.jpg -a sunete/melodie.wav
```

#### **Folosind audio sintetic**

Dacă nu ai un fișier audio, generează sunete de armonici.

Bash

```
python main.py -i imagini/portret.png -s harmonics
```

#### **Selectarea unor metode specifice**

Aplică doar metodele `energy_waves` și `spectral_painting`.

Bash

```
python main.py -i poza.jpg -a sunet.mp3 -m energy_waves spectral_painting
```

#### **Generarea doar a variațiilor sinestezice**

Sari peste transformările de bază și generează direct arta sinestezică.

Bash

```
python main.py -i arta.jpg -a sunet_ritmat.wav --synesthetic-only
```

#### **Specificarea directorului de output**

Salvează rezultatele într-un folder personalizat.

Bash

```
python main.py -i imagine.jpg -s noise -o "Rezultate Creative"
```

#### **Ajutor și listarea metodelor**

Pentru a vedea toate opțiunile disponibile:

Bash

```
python main.py -h
```

Pentru a vedea o descriere a tuturor metodelor:

Bash

```
python main.py --list-methods
```

---

## 🖼️ Exemple de output

Rezultatele sunt salvate implicit în directorul `databending_results`. Fiecare fișier este denumit după metoda folosită.

|Metoda|Descriere|
|---|---|
|`frequency_bands_databend.png`|Imagine colorată de frecvențele sunetului.|
|`energy_waves_databend.png`|Imagine deformată de undele de energie.|
|`taste_databend.png`|O interpretare vizual-gustativă a sunetului.|
|`geometry_databend.png`|Forme geometrice suprapuse peste imagine.|
|... și multe altele||

Exportă în Foi de calcul

---

## 🤝 Contribuții

Contribuțiile sunt binevenite! Dacă ai idei pentru noi metode de transformare, optimizări sau ai găsit un bug, te rog deschide un "Issue" sau un "Pull Request".
