# Phase 2 – SFT Training

**Dauer:** Tag 3

**Status:** ⏳ **BEREIT FÜR START** (wartet auf Remote-H100)

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
- [ ] SFT Dataset laden (~80.000 Samples)
- [ ] Data Preprocessing & Tokenization
- [ ] LoRA Adapter initialisieren (rank 32, alpha 64)
- [ ] QLoRA 4-bit Quantisierung aktivieren

### 2. Training konfigurieren
- [ ] Target Modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- [ ] Learning Rate: optimieren (empfohlen: 2e-4 bis 1e-3)
- [ ] Batch Size: an GPU-Speicher anpassen
- [ ] 3 Epochen einstellen
- [ ] Checkpoint-Intervalle setzen

### 3. Training durchführen
- [ ] Start SFT Training (`~4 Stunden` auf H100)
- [ ] Loss-Kurven monitoren
- [ ] Gradient Explosion/Vanishing prüfen
- [ ] Checkpoints speichern

### 4. Post-Training Validierung
- [ ] Inference-Tests auf Holdout-Set
- [ ] Mode Accuracy prüfen
- [ ] Erste qualitative Bewertung

## Deliverables

- [ ] SFT-trained Model (Checkpoint)
- [ ] Training Logs & Metrics
- [ ] Erste Validierungsergebnisse

## Erfolgskriterien

- [ ] Training abgeschlossen ohne Errors
- [ ] Loss konvergiert
- [ ] Model kann 7 Modi unterscheiden
- [ ] Qualitative Tests zeigen korrektes Routing

## Metriken

| Metrik | Erwartet |
|--------|----------|
| Train Loss | sinkend |
| Eval Loss | sinkend |
| Mode Classification Accuracy | > 70% (erste Schätzung) |

## Pass@1 Protection

**Während des Trainings:**

```python
from diogenes import Pass1RegressionTracker, compute_core_reliability_metrics

tracker = Pass1RegressionTracker(checkpoint_dir="./checkpoints")

# Nach jedem Epoch
core_metrics = compute_core_reliability_metrics(...)
result = tracker.record_checkpoint(
    checkpoint_name=f"epoch_{epoch}",
    core_metrics=core_metrics,
    pass_at_k_math={5: pass_at_5, 10: pass_at_10},
)

if not result.should_promote:
    print(f"⚠️ Regression detected: {result.regression_details}")
```

**Achtung:** Pass@k (Math/Code) nur für Monitoring verwenden!

## Nächste Schritte

➡️ **Phase 3**: DPO Training

- DPO-Dataset generieren (60k Paare)
- DPO-Audit durchführen
- DPO Training mit SFT-Checkpoint als Basis

## Referenzen

- `src/diogenes/train_sft.py` – Training Script
- `src/diogenes/pass1_protection.py` – Regression Detection
- `docs/PASS1_GUARDRAILS.md` – Pass@1 Richtlinien
