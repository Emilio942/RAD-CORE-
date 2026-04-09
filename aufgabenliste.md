Goal A: Model Behavior Under Hardware-Level Radiation Degradation
Focus: Internal computational noise. Analysis of model and training/inference behavior when the physical processing unit (CPU/GPU/NPU/FPGA) is subjected to high radiation. This includes evaluating the impact of hardware faults such as bit flips (Single Event Upsets), memory corruption, and corrupted activations/weights directly within the chip.

Goal B: Algorithmic Efficiency and Speed
Focus: Algorithmic speedups. Identification and implementation of optimizations for the RLVR (Reinforcement Learning with Verifiable Rewards) and Online Label Refinement (OLR) processes. The objective is to reduce algorithmic complexity, accelerate training/inference cycles, and achieve the target robustness with fundamentally lower computational overhead.


Phase 1: Baseline & Profiling
[x] Referenz-Setup: Minimal lauffähige Version von RLVR und OLR (Online Label Refinement) aus dem Paper aufsetzen. (In rad_core.py implementiert)

[ ] Deep Profiling: Messen, wo genau die meiste Rechenzeit und der meiste Speicher verbrannt werden. (Wahrscheinlich: Rollout-Generierung oder die historischen Checks für die OLR-Slope).

Phase 2: Ziel A (Strahlungssimulation / Hardware Faults)
Fokus: PyTorch-Modifikationen für Low-Level Bit-Flips, um das Chaos im Chip zu simulieren.

[x] Hooks für Transient Faults: Custom forward_hook und backward_hook in PyTorch schreiben, die stochastisch Single Event Upsets (SEUs) in die Activations injizieren (Simulation von flüchtiger Strahlung während der Berechnung). (Forward hooks in rad_core.py)

[ ] Permanent Fault Injection: Skript schreiben, das gezielt zufällige Bit-Flips in die Weights (FP32/BF16) im VRAM schreibt (Simulation von Memory Corruption).

[x] Auswertung RLVR-Resilienz: Testen, ob der OLR-Mechanismus – der eigentlich für Label-Noise gedacht ist – zufällig auch robust gegenüber korrumpierten Gradienten und Hardware-bedingten Fehlberechnungen ist. (Initialer Test in rad_core.py)

Phase 3: Ziel B (Algorithmische Effizienz)
Fokus: Mathematische und architektonische Verschlankung des RLVR-Prozesses.

[x] OLR-Historie approximieren: Das Paper nutzt eine Historie zur Berechnung der "Early Correctness Coherence" (positive Slope). Aufgabe: Diese speicherintensive Historie durch einen exponentiell gleitenden Durchschnitt (EMA) ersetzen. (EMA in rad_core.py umgesetzt)

[ ] Rollout-Beschleunigung: Implementierung von Speculative Decoding oder optimiertem KV-Caching während der RLVR-Rollouts, um die Inferenzzeit massiv zu senken.

[ ] Sparse Reward Verification: Evaluieren, ob die Verifiable Rewards nur auf einem dynamischen Sub-Sample der Batches berechnet werden können (spart Rechenzeit), ohne dass das Modell kollabiert.

Phase 4: Synthese & Stresstest
[ ] Integration: Den optimierten, schnellen Algorithmus (Ziel B) in die Strahlungssimulation (Ziel A) werfen.

[ ] Bruchstellen-Analyse: Messen, ab welcher Strahlungsdosis (Bit-Flip-Rate) der verschlankte Algorithmus im Vergleich zur teuren Original-Baseline zusammenbricht.