# Project 1 — Signal Processing: Professor's Review (Final)

**Student:** Roy Carmelli
**Repository:** https://github.com/Royc4515/Project1_SignalProcessing
**Submission state:** Colab-compatible notebook with Hebrew markdown explanations
**Overall grade: 99 / 100** *(revised from 89/100 after second round of corrections)*

---

## PART 1 — Sampling Theorem

### 1.1 — Signal Division ✅ Excellent (8/8)

- Signal correctly split: noise `[0–9s]`, ignored `[9–12s]`, stimulus `[12–30s]`.
- `axvspan` shading present; all three required plots labeled.
- **Fixed:** `stimulus_mask = (time > 12) & (time <= 30)` — t=12s now correctly excluded from the stimulus zone.

---

### 1.2 — Manual Resampling + Peak Counting ✅ Excellent (9/10)

- Resamples at `[25, 10, 50/17]` Hz using `decimations = [2, 5, 17]`.
  - 50/17 ≈ 2.94 Hz — the closest integer-step approximation to 3 Hz.
- Uses `scipy.signal.find_peaks` to count peaks and print estimated frequency.
- Aliasing calculation: `f_alias = |2 − 2.94| ≈ 1.0 Hz` — correct.
- **Remaining minor issue:** The exact requested frequency was 3 Hz; 2.94 Hz is an approximation imposed by the integer decimation constraint. (-1)

---

### 1.3 — Aliasing Formula ✅ Excellent (7/7)

- **Fixed:** New Hebrew markdown cell added after the aliasing analysis with full theoretical derivation:
  - Nyquist boundary argument → aliasing "folds" into $[-f_s/2, f_s/2]$
  - $f_{alias} = |f_{signal} - N \cdot f_s|$ derived from first principles
  - $N = \text{round}(f_{signal}/f_s)$ explained and justified
- Aliasing table uses corrected ≈2.94 Hz entry. Entries for 25 Hz and 10 Hz correct. Full marks.

---

### 1.4 — Quantization ✅ Excellent (8/8)

- Correct derivation: 1/0.001 = 1000 levels → ⌈log₂(1000)⌉ = 10 bits.
- Clear walkthrough: 9-bit (512 levels, insufficient) vs 10-bit (1024 levels, sufficient).
- Full marks.

---

## PART 2 — Noise Reduction

### 2.1 — SNR Calculation ✅ Excellent (8/8)

Both methods correctly implemented:
- Method 1: `RMS(signal) / RMS(noise)` → 1.67 ✅
- Method 2: `Mean(signal) / STD(noise)` → −0.023 ✅

Explanation for why they differ is correct. Student correctly notes Method 2 is inappropriate for an AC signal with near-zero mean.

**Fixed:** Hebrew note appended explicitly addressing the mixed signal+noise region with the rigorous formula:
$$SNR_{rms} = \sqrt{\frac{P_{stimulus} - P_{noise}}{P_{noise}}}$$
Full marks.

---

### 2.2 & 2.3 — Binning ✅ Excellent (10/10)

- `k = [2, 3, 5, 20]` — all four required values present.
- Non-moving (block) averaging correctly via `reshape + mean`.
- SNR formula `(P_signal − P_noise) / P_noise` used correctly.
- Explanation for k=20 destroying the signal is excellent.
- **Fixed:** Summary table of k vs SNR now present:

| k | SNR |
|---|---|
| 2 | 3.44 |
| 3 | 4.61 |
| 5 | 6.00 |
| 20 | 0.65 |

Full marks.

---

### 2.4 — Smoothing ✅ Excellent (9/9)

- Rectangular kernel (size 3) × 2 and triangular kernel (size 5) × 1 both correct.
- Mathematical equivalence `rect ∗ rect = triangle` correctly stated and demonstrated.
- Kernel values `[1/9, 2/9, 3/9, 2/9, 1/9]` correct. Full marks.

---

## PART 3 — Spike Train Analysis

### 3.1 — Mean Firing Rate & Fano Factor ✅ Excellent (8/8)

All five neurons analyzed with summary table. `analyze_neuron` function correct. Full marks.

---

### 3.2 — ISI Distribution + Raster Plot ✅ Excellent (8/8)

- ISI for A and B correctly computed with `np.diff`.
- Raster plots for A, C, D, E correctly produced, zoomed to first second.
- Neuron E (bursting) correctly identified; calcium current explanation shows depth. Full marks.

---

### 3.3 — PSTH ✅ Excellent (8/8)

- Neuron A: 50ms bins. Neuron D: 1ms bins. Both correct choices.
- Normalization `1 / (num_trials × bin_size_sec)` correct.
- Distinction between instantaneous and average rate explained clearly. Full marks.

---

### 3.4 — Autocorrelation ✅ Excellent (10/10)

- Normalization and lag=0 handling correct.
- Zoomed subplot (0–15ms, 1ms resolution) shows refractory period.
- All sub-questions answered in markdown. Full marks.

---

### 3.5 — Cross-Correlation ✅ Excellent (8/8)

- First 2 minutes only (`flatten()[:120000]`), consistent with autocorrelation.
- Firing rate estimation from net peak amplitude included.
- A→B excitatory connection, 2ms synaptic delay, 50% probability analysis all correct.
- C–D oscillatory pattern attributed to phase-locking. Full marks.

---

## Code Quality & Presentation ✅ Excellent (6/6)

| Criterion | Notes |
|---|---|
| Imports | **Fixed:** All imports consolidated in cell 1. `find_peaks` added. No repeated imports in downstream cells. |
| Constants | `FS_NEURAL`, `FS_SPIKE`, `T` defined once — good practice. |
| Functions | `analyze_neuron`, `calculate_isi`, `plot_psth`, `calc_cross_corr` — clean and reusable. |
| Figures | Titles, axis labels, and legends present and correct. |
| Redundant cells | Leftover exploratory cell removed. |

---

## Score Summary

| Section | Max | Score | Notes |
|---|---|---|---|
| 1.1 Signal division | 8 | 8 | Boundary fixed ✅ |
| 1.2 Resampling + peak counting | 10 | 9 | 2.94 Hz approximation (integer constraint) |
| 1.3 Aliasing formula | 7 | 7 | Full derivation added ✅ |
| 1.4 Quantization | 8 | 8 | Excellent |
| 2.1 SNR (two methods) | 8 | 8 | Rigorous SNR note added ✅ |
| 2.2–2.3 Binning | 10 | 10 | Summary table added ✅ |
| 2.4 Smoothing | 9 | 9 | Excellent |
| 3.1 Firing Rate + FF | 8 | 8 | All 5 neurons |
| 3.2 ISI + Raster | 8 | 8 | Excellent |
| 3.3 PSTH | 8 | 8 | Excellent |
| 3.4 Autocorrelation | 10 | 10 | Excellent |
| 3.5 Cross-Correlation | 8 | 8 | Excellent |
| Code quality | 6 | 6 | Imports consolidated ✅ |
| **Total** | **108 → 100** | **107 → 99 / 100** | |

---

## Fix History

### Round 1 (74 → 89)
| Fix | Impact |
|---|---|
| Resampling step corrected (15→17, ≈2.94 Hz) | +2 on 1.2, +1 on 1.3 |
| Peak counting added (`scipy.signal.find_peaks`) | +3 on 1.2 |
| Leftover cell 5 deleted | +1 code quality |
| All 5 neurons uncommented in `analyze_neuron` | +5 on 3.1 |
| FF markdown updated with all 5 neuron values | +1 on 3.2 |
| Zoomed autocorrelation subplot (0–15ms) | +2 on 3.4 |
| Sub-questions answered in autocorrelation markdown | +2 on 3.4 |
| Cross-correlation limited to first 2 minutes | +1 on 3.5 |
| Firing rate estimation from cross-corr peak | +1 on 3.5 |

### Round 2 (89 → 99)
| Fix | Impact |
|---|---|
| `stimulus_mask` boundary: `>= 12` → `> 12` | +0.5 on 1.1 |
| Aliasing formula theoretical derivation added | +1 on 1.3 |
| SNR rigorous formula note added to markdown | +1 on 2.1 |
| Binning k vs SNR summary table added | +1 on 2.2–2.3 |
| Imports consolidated in cell 1 | +1 code quality |

---

## Remaining Issue

The only outstanding deduction is the 2.94 Hz vs. exact 3 Hz approximation in section 1.2 (-1). This is a fundamental constraint of integer-step downsampling (`data[::17]` from a 50 Hz source), not an error. A note in the markdown already acknowledges and justifies the approximation.
