# Phase 2 – SFT Training

**Dauer:** Tag 3

## Ziele

- [ ] Supervised Fine-Tuning durchführen
- [ ] Modusverhalten & Routing stabilisieren
- [ ] 80k Samples über 3 Epochen trainieren

## Aufgaben

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
