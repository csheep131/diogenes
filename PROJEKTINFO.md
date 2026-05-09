# Diogenes - The Reliable 32B

## Zweck
Entwicklung eines epistemisch zuverlässigen Sprachmodells auf Basis von **Qwen3-32B**. Das Modell wird optimiert für korrekte Unsicherheitsabschätzung, Minimierung von Halluzinationen und epistemisch korrektes Antwortverhalten in kritischen Domänen (IT, Medizin, Recht, Finanzen). Im Gegensatz zu klassischen LLMs fokussiert Diogenes auf die Erkennung von Wissensgrenzen und ehrliche Kommunikation von Unsicherheit.

## Tech Stack
- **Programmiersprache:** Python 3.10+
- **ML-Frameworks:** PyTorch ≥2.1.0, Transformers ≥4.37.0, Accelerate ≥0.25.0
- **Fine-Tuning:** PEFT (LoRA/QLoRA), bitsandbytes ≥0.42.0 für 4-bit Quantisierung
- **Training-Pipelines:** Axolotl ≥0.4.0 und/oder Unsloth ≥2024.1
- **Datenverarbeitung:** Datasets ≥2.14.0, Pandas ≥2.0.0, NumPy ≥1.24.0
- **Evaluation:** Evaluate ≥0.4.1, rouge-score, sacrebleu, custom epistemische Metriken
- **Experiment-Tracking:** Weights & Biases ≥0.16.0
- **Basismodell:** Qwen3-32B (mit Option für Qwen2.5-3B in Entwicklungsphasen)
- **Hardware:** RTX 3050 8GB (Entwicklung), H100 80GB (Produktionstraining)

## Architektur
- **Epistemischer Routing Head:** 7-Klassen-Classifier für Antwortmodi (DIRECT_ANSWER, CAUTIOUS_LIMIT, ABSTAIN, CLARIFY, REJECT_PREMISE, REQUEST_TOOL, PROBABILISTIC)
- **Kalibrierung:** Temperature Scaling für korrekte Confidence-Schätzung
- **Training:** Zweistufig - Supervised Fine-Tuning (SFT) gefolgt von Direct Preference Optimization (DPO)
- **Evaluations-Suite:** TruthfulQA, HaluEval, WildBench, GPQA sowie custom epistemische Metriken

## Entwicklungsstrategie
- **Phase 0-6:** Lokale Entwicklung und Validierung auf RTX 3050 mit kleineren Modellen (0.6B-3B)
- **Phase 7:** Finales Training des 32B-Modells auf H100-Infrastruktur
- **Shadow Loop:** Paralleler Custom-Training-Loop für experimentelle epistemische Regularization

## Projektstatus
- **Aktuelle Phase:** Phase 2 (SFT Testing auf RTX 3050 mit 3B-Modell) — **READY TO START**
- **Letztes Update:** 9. Mai 2026 — Project Restart nach Pause seit März 2026
- **Environment:** Python 3.11.8, PyTorch 2.10.0, CUDA 13.0, RTX 3050 8GB erkannt ✅
- **Tests:** 41 neue Tests hinzugefügt (test_model, test_inference, test_train_sft, test_train_dpo) — alle grün ✅
- **Datasets:** SFT 80k, DPO 60k, SUA 25k generiert und validiert ✅
- **Smoke Test:** Pipeline-Validierung mit Qwen3-0.6B (100 Samples, 1 Epoche) läuft
- **Roadmap:** Vollständige 20-Tage-Planung mit klaren Exit-Kriterien und Decision Gates
- **Repository:** Strukturiertes Python-Paket mit vollständiger CI/CD und Testing-Infrastruktur
- **Bekannte Issues:** test_attnres.py hat Import-Bug (`nn` nicht importiert in `full.py`), test_dataset_generator.py verwendet pytest-subtests falsch, einige test_pass1_protection.py Assertions (numpy bool vs Python bool)