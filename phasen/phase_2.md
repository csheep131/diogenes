# Phase 2 – SFT Training

**Dauer:** Tag 3

## Ziele

- [ ] Supervised Fine-Tuning durchführen
- [ ] Modusverhalten & Routing stabilisieren
- [ ] 80k Samples über 3 Epochen trainieren

## 0. Remote Maschine vorbereiten

### Voraussetzungen
- SSH-Zugang zu Remote-Server (GPU mit mind. 80GB VRAM, z.B. H100/A100)
- Sudo-Rechte auf Remote-Maschine
- Lokales Python 3.10+ für Setup-Skript

### Setup-Ablauf

**Option A: Mit CLI-Argumenten**
```bash
python scripts/prepare_remote_machine.py --host <IP> --user <username>
```

**Option B: Mit Konfigurationsdatei**
```bash
# configs/remote_config.yaml bearbeiten
python scripts/prepare_remote_machine.py --config configs/remote_config.yaml
```

**Das Skript führt folgende Schritte aus:**
1. ✓ SSH-Verbindung testen
2. ✓ Hardware prüfen (GPU, VRAM, CPU, RAM, Disk)
3. ✓ Dependencies installieren (Docker, NVIDIA Container Toolkit, Python)
4. ✓ Projektverzeichnis anlegen (`/opt/diogenes`)
5. ✓ Code zur Remote-Maschine synchronisieren
6. ✓ Optional: Datasets synchronisieren
7. ✓ Training-Launch-Skript erstellen (`train.sh`)

### Training starten

**Nach der Vorbereitung:**
```bash
# Direkt via SSH
ssh <user>@<host> 'cd /opt/diogenes && ./train.sh'

# Oder interaktiv auf der Remote-Maschine
ssh <user>@<host>
cd /opt/diogenes
./train.sh
```

### 1. Training vorbereiten
- SFT Dataset laden (~80.000 Samples)
- Data Preprocessing & Tokenization
- LoRA Adapter initialisieren (rank 32, alpha 64)
- QLoRA 4-bit Quantisierung aktivieren

### 2. Training konfigurieren
- Target Modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- Learning Rate: optimieren (empfohlen: 2e-4 bis 1e-3)
- Batch Size: an GPU-Speicher anpassen
- 3 Epochen einstellen
- Checkpoint-Intervalle setzen

### 3. Training durchführen
- Start SFT Training (`~4 Stunden` auf H100)
- Loss-Kurven monitoren
- Gradient Explosion/Vanishing prüfen
- Checkpoints speichern

### 4. Post-Training Validierung
- Inference-Tests auf Holdout-Set
- Mode Accuracy prüfen
- Erste qualitative Bewertung

## Deliverables

- [ ] SFT-trained Model (Checkpoint)
- [ ] Training Logs & Metrics
- [ ] Erste Validierungsergebnisse

## Erfolgskriterien

- Training abgeschlossen ohne Errors
- Loss konvergiert
- Model kann 7 Modi unterscheiden
- Qualitative Tests zeigen korrektes Routing

## Metriken

- Train Loss: sollte sinken
- Eval Loss: sollte sinken
- Mode Classification Accuracy (erste Schätzung)
