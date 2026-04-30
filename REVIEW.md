# Project 1 — Signal Processing: Professor's Review (Post-Fix)

**Student:** Roy Carmelli
**Repository:** https://github.com/Royc4515/Project1_SignalProcessing
**Submission state:** Colab-compatible notebook with Hebrew markdown explanations
**Overall grade: 89 / 100** *(revised from 74/100 after corrections)*

---

## PART 1 — Sampling Theorem

### 1.1 — Signal Division ✅ Good (7.5/8)

- Signal correctly split: noise `[0–9s]`, ignored `[9–12s]`, stimulus `[12–30s]`.
- `axvspan` shading is a nice visual touch; all three required plots are present and labeled.
- **Remaining minor issue:** `stimulus_mask = time >= 12` includes the exact boundary t=12s, which
  belongs to the "ignored" zone per the instructions. Should be `time > 12`. Negligible in
  practice but technically incorrect. (-0.5)

---

### 1.2 — Manual Resampling + Peak Counting ✅ Excellent (9/10)

**What the corrected code does:**
- Resamples at `[25, 10, 50/17]` Hz using `decimations = [2, 5, 17]`.
  - 50/17 ≈ 2.94 Hz — the closest integer-step approximation to 3 Hz. This is the correct
    approach since `data[::step]` requires integer steps, and step=17 gives 50/17 ≈ 2.94 Hz
    while step=16 gives 50/16 = 3.125 Hz. Step=17 is the better approximation.
- Uses `scipy.signal.find_peaks` to count peaks in each resampled signal and prints the
  estimated frequency (`n_peaks / duration`). This directly answers the "count peaks"
  sub-question that was originally missing.

**Aliasing calculation with corrected fs:**
- `f_alias = |2 − 1 × 2.94| ≈ 1.0 Hz` — close to the correct theoretical answer.

**Remaining minor issue:** The exact requested frequency was 3 Hz; 2.94 Hz is an approximation
imposed by the integer decimation constraint. A note justifying why step=17 is correct would
strengthen the answer. (-1)

---

### 1.3 — Aliasing Formula ✅ Good (6/7)

The aliasing table now uses the corrected ≈2.94 Hz entry. Entries for 25 Hz and 10 Hz are
correct (both above Nyquist, no aliasing). The formula `N = round(f_signal / fs_new)` works
correctly for these values. The markdown explanation correctly cites the Nyquist theorem.

**Remaining minor:** The formula is applied correctly but not formally derived. (-1)

---

### 1.4 — Quantization ✅ Excellent (8/8)

- Correct derivation: 1/0.001 = 1000 levels → ⌈log₂(1000)⌉ = 10 bits.
- Clear walkthrough: 9-bit (512 levels, insufficient) vs 10-bit (1024 levels, sufficient).
- Mathematical notation is clean. Full marks.

---

## PART 2 — Noise Reduction

### 2.1 — SNR Calculation ✅ Good (7/8)

Both methods correctly implemented:
- Method 1: `RMS(signal) / RMS(noise)` → 1.67 ✅
- Method 2: `Mean(signal) / STD(noise)` → −0.023 ✅

Explanation for why they differ is correct and shows conceptual understanding. The student
correctly notes that Method 2 is inappropriate for an AC signal with near-zero mean.

**Remaining minor:** The stimulus region contains both signal AND noise. A rigorous `SNR_rms`
would compute `sqrt((P_stim − P_noise) / P_noise)`. Not explicitly addressed. (-1)

---

### 2.2 & 2.3 — Binning ✅ Good (9/10)

- `k = [2, 3, 5, 20]` — all four required values are present.
- Non-moving (block) averaging implemented correctly via `reshape + mean`.
- SNR formula `(P_signal − P_noise) / P_noise` is used correctly.
- Explanation for k=20 destroying the signal is excellent: 20 samples at 50 Hz = 0.4s ≈
  one full period of the 2 Hz signal → averaging cancels the oscillation.
- **Remaining minor:** No summary table of k vs SNR value in the report. (-1)

---

### 2.4 — Smoothing ✅ Excellent (9/9)

- Rectangular kernel (size 3) × 2 and triangular kernel (size 5) × 1 are both correct.
- Mathematical equivalence is correctly stated and demonstrated: `rect ∗ rect = triangle`.
- Kernel values `[1/9, 2/9, 3/9, 2/9, 1/9]` are correct.
- Overlay plot showing both curves coinciding is exactly the right visualization.
- LPF explanation is solid and conceptually accurate. Full marks.

---

## PART 3 — Spike Train Analysis

### 3.1 — Mean Firing Rate & Fano Factor ✅ Excellent (8/8)

All five neurons now analyzed:

| Neuron | Type | Mean FR (Hz) | Fano Factor | Classification |
|--------|------|-------------|-------------|----------------|
| A | Poisson | ~10 | ~1.0 | Random/irregular |
| B | Regular | ~15 | <1 | Slightly regular |
| C | Regular+jitter | ~20 | <<1 | Regular with noise |
| D | Clock-like | ~50 | ~0 | Perfectly regular |
| E | Bursting | ~8 | >>1 | Bursting pattern |

The `analyze_neuron` function is well-written — windowed Fano Factor logic is correct.
All five neurons produce firing rate and FF values that are discussed in subsequent cells.

---

### 3.2 — ISI Distribution + Raster Plot ✅ Excellent (8/8)

- ISI for A and B: correctly computed with `np.diff(spike_times)`.
- Raster plots for A, C, D, E: correctly produced and zoomed to first second.
- Neuron E (bursting) correctly identified from raster. Linking bursting to calcium
  currents shows depth of understanding.
- FF for B is now computed (Fix 5) — Poisson assumption can be verified against actual FF value.

---

### 3.3 — PSTH ✅ Excellent (8/8)

- Neuron A: 50ms bins — appropriate for a low-rate Poisson neuron.
- Neuron D: 1ms bins — correct choice to resolve sharp periodic peaks.
- Normalization to Hz is implemented correctly: `1 / (num_trials × bin_size_sec)`.
- Explanation for why D's instantaneous PSTH rate (200–300 Hz) far exceeds its mean rate
  (~50 Hz) is excellent. Full marks.

---

### 3.4 — Autocorrelation ✅ Excellent (10/10)

**Implementation:** Normalization by `mean_rate² × N_samples` is correct. Setting lag=0
to 0 is standard practice.

**Fixes applied:**

1. **Zoomed subplot added (lag 0–15ms, 1ms resolution)** — refractory period is now
   clearly visible as a gap near lag 1–3ms. A vertical dashed line marks the estimated
   refractory period end.

2. **Sub-questions explicitly answered in markdown:**
   - *"What can the number of peaks tell us?"* — Peak spacing = 1/firing_rate. More peaks
     at regular intervals → more regular neuron. Poisson neurons show exponential decay with
     no clear peaks.
   - *"After each peak, there is a local minimum. What can it tell us?"* — The trough below
     baseline indicates post-spike suppression (refractory/inhibitory period). For bursting
     neurons, it reflects inter-burst silence.
   - *"Which type is neuron C? Does the FF match?"* — Answered using actual FF from cell 3.1.

**Analysis quality:** Characterization of all four neuron types (Poisson/A, regular+jitter/C,
clock-like/D, bursting/E) is correct and well-explained. Full marks.

---

### 3.5 — Cross-Correlation ✅ Excellent (8/8)

**Methodological fix applied:** Cross-correlation now uses first 2 minutes only:
```python
N_cc = 120000  # first 2 minutes at 1000 Hz
A_flat = neuron_A.flatten()[:N_cc]
B_flat = neuron_B.flatten()[:N_cc]
```
This matches the autocorrelation's approach consistently.

**Firing rate estimation added:**
- Baseline computed from lags 50–100ms.
- Net peak = peak amplitude − baseline.
- Estimated rate = net_peak / recording_duration.
- Compared against directly computed rate from section 3.1.

**Analysis quality:**
- Peak at lag +2ms for A→B: correctly identifies A as presynaptic, 2ms synaptic delay. ✅
- Correctly identifies the connection as excitatory. ✅
- 50% probability → peak halved: correct. ✅
- Oscillatory C–D cross-correlogram: correctly attributed to phase-locking. ✅

---

## Code Quality & Presentation

| Criterion | Notes |
|---|---|
| Imports | Still repeated in multiple cells. Should be consolidated in cell 1. |
| Constants | `FS_NEURAL`, `FS_SPIKE`, `T` defined once — good practice. |
| Functions | `analyze_neuron`, `calculate_isi`, `plot_psth`, `calc_cross_corr` — clean and reusable. |
| Figures | Titles, axis labels, and legends present and correct. |
| Cell 5 | Leftover exploratory cell removed. ✅ |

---

## Score Summary

| Section | Max | Score | Notes |
|---|---|---|---|
| 1.1 Signal division | 8 | 7.5 | Boundary off-by-one at t=12s (minor) |
| 1.2 Resampling + peak counting | 10 | 9 | Correct decimation (step=17 ≈ 2.94 Hz); peak counting added |
| 1.3 Aliasing formula | 7 | 6 | Correct approach, formula not derived |
| 1.4 Quantization | 8 | 8 | Excellent |
| 2.1 SNR (two methods) | 8 | 7 | Correct; SNR_rms subtlety not addressed |
| 2.2–2.3 Binning | 10 | 9 | No summary table |
| 2.4 Smoothing | 9 | 9 | Excellent |
| 3.1 Firing Rate + FF (all neurons) | 8 | 8 | All 5 neurons computed with summary table |
| 3.2 ISI + Raster | 8 | 8 | FF for B now verified |
| 3.3 PSTH | 8 | 8 | Excellent |
| 3.4 Autocorrelation | 10 | 10 | Zoomed plot added; all sub-questions answered |
| 3.5 Cross-Correlation | 8 | 8 | 2-min limit applied; firing rate estimate added |
| Code quality | 6 | 5 | Repeated imports remain; leftover cell removed |
| **Total** | **108 → 100** | **102.5 → 89 / 100** | |

---

## What Was Fixed

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

**Grade improvement: 74 → 89 (+15 points)**

---

## Remaining Minor Issues

1. **Boundary condition:** `stimulus_mask = time >= 12` should be `time > 12` (t=12s is "ignored").
2. **Aliasing formula derivation:** The formula `N = round(f_signal / fs_new)` is applied
   correctly but not formally derived from first principles.
3. **SNR subtlety:** Rigorous `SNR_rms` for a mixed signal+noise region would use
   `sqrt((P_stim − P_noise) / P_noise)` rather than `RMS(stim) / RMS(noise)`.
4. **Repeated imports:** `import numpy`, `import matplotlib`, etc. appear in 5+ cells.
   Should be consolidated in cell 1.
5. **No k vs SNR summary table** in the binning section.
