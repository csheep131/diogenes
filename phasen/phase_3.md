# Phase 3 – DPO Training

**Dauer:** Tag 4

## Ziele

- [ ] Direct Preference Optimization durchführen
- [ ] Halluzinationen bestrafen
- [ ] Ehrliche Antworten belohnen
- [ ] 60k Preference Pairs trainieren

## Aufgaben

### 1. Training vorbereiten
- DPO Dataset laden (~60.000 Paare)
- Ranking-Klassen: Gold > Acceptable > Weak > Hallucination
- Data Preprocessing für Preference Learning

### 2. DPO Training konfigurieren
- SFT-Checkpoint als Basis laden
- Preference Loss Funktion einstellen
- Beta-Parameter optimieren (typisch: 0.1–0.5)
- Batch Size & Learning Rate anpassen

### 3. Training durchführen
- Start DPO Training (`~6 Stunden` auf H100)
- Preference Accuracy monitoren
- Reward-Modell-Verlauf tracken
- Checkpoints speichern

### 4. Post-Training Validierung
- Halluzinationsrate auf Testset prüfen
- Preference Accuracy messen
- Qualitative Bewertung der Antworten

## Deliverables

- [ ] DPO-trained Model (Checkpoint)
- [ ] Training Logs & Metrics
- [ ] Halluzinations-Baseline gemessen

## Erfolgskriterien

- Training abgeschlossen ohne Errors
- Preference Accuracy > Zufallsniveau
- Halluzinationen reduziert vs. SFT-only
- Ehrliche Ablehnungen stabil

## Metriken

- DPO Loss: sollte sinken
- Preference Accuracy: sollte steigen
- Halluzinationsrate (erste Schätzung)
