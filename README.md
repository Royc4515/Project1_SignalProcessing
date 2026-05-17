<div dir="rtl">

# פרויקט 1 — עיבוד אותות ביולוגיים

**קורס:** עיבוד אותות לחקר המוח · **מוסד:** אוניברסיטת בר-אילן · **שנה:** תשפ"ו

🔗 [פתיחת המחברת ב-Google Colab](https://colab.research.google.com/github/Royc4515/Project1_SignalProcessing/blob/main/Project1_SignalProcessing.ipynb)

---

## חלק 1 — משפט הדגימה, Aliasing וקוונטיזציה

### 1.1 חלוקת האות לקטעים

הנתון: הקלטת VSDI באורך 30 שניות בתדר דגימה $F_s = 50\,\text{Hz}$ (סה"כ 1500 דגימות), עם גירוי של תנודות 2 Hz בין שניות 12–30. תשע השניות הראשונות מוגדרות כ-baseline (רעש), ושניות 9–12 מתעלמים מהן (תקופת מעבר).

חלקתי את האות לפי המסכות:
- **רעש:** `time < 9`
- **גירוי:** `(time > 12) & (time <= 30)`

```python
import pandas as pd
neural_df = pd.read_csv('Neural_data.csv')
dff    = neural_df['ΔF/F (%)'].values
time   = neural_df['Frame'].values / 50  # Fs = 50 Hz

noise_signal    = dff[time < 9]
stimulus_signal = dff[(time > 12) & (time <= 30)]
```

![האות המלא](graphs/VSDI_full_signal.png)

האות המלא מציג את כל ה-30 השניות עם סימון אזורי הרעש (אדום), המעבר המתעלם (אפור), והגירוי (ירוק). אפשר לראות בבירור את שינוי האופי בשנייה 12 — מעבר מרעש רקע סטוכסטי לאוסצילציה מחזורית.

![קטע הרעש](graphs/noise_segment.png)

קטע הרעש (0–9 שניות) — האות סביב ערך בסיס יחסית קבוע עם תנודות קצרות-טווח אקראיות. זהו מודל הרעש שישמש לחישובי SNR בהמשך.

![קטע הגירוי](graphs/stimulus_segment.png)

קטע הגירוי (12–30 שניות) — תנודות מחזוריות ברורות סביב ערך בסיס, בהתאם לגירוי של 2 Hz.

---

### 1.2 דגימה מחדש ידנית וחישוב Aliasing

ההוראה: לבצע resampling ידני על קטע הגירוי לפי `data[::int]` בתדרים 25, 10 ו-3⅓ Hz. החישוב: כדי לקבל $F_s' = 10/3\,\text{Hz}$ מתוך 50 Hz, צריך פקטור $50/(10/3) = 15$, כלומר `data[::15]`. עבור 25 ו-10 Hz: `data[::2]` ו-`data[::5]`.

```python
new_rates   = [25, 10, 50/15]   # = [25, 10, 10/3] Hz
decimations = [2,   5,  15]
for fs_new, dec in zip(new_rates, decimations):
    resampled = stimulus_signal[::dec]
```

תדר נייקוויסט הנדרש כדי שלא יהיה aliasing הוא $2 \cdot f_{\text{signal}} = 4\,\text{Hz}$. נוסחת הקיפול: $f_{\text{alias}} = |f_{\text{signal}} - N \cdot F_s'|$, כאשר $N = \text{round}(f_{\text{signal}} / F_s')$.

| $F_s'$ (Hz) | $F_s' > 2 f_{\text{signal}}$? | $N$ | $f_{\text{alias}}$ (Hz) | תדר נצפה |
|:---:|:---:|:---:|:---:|:---:|
| 25 | כן | 0 | 2.00 | 2 Hz |
| 10 | כן | 0 | 2.00 | 2 Hz |
| 10/3 ≈ 3.33 | לא | 1 | 1.33 | 1.33 Hz |

![השוואת תדרי דגימה](graphs/resampling_comparison.png)

בגרף רואים שעבור 25 Hz ו-10 Hz האות שומר על תדר של 2 Hz (כ-36 שיאים ב-18 שניות). ב-3⅓ Hz תדר הדגימה נמוך מ-$2 f_{\text{signal}}$, ולכן האות מתקפל ונראה כאילו תדרו ~1.33 Hz (כ-24 שיאים) — בהתאם לחישוב.

---

### 1.3 קוונטיזציה

**שאלה:** כמה ביטים מינימום נדרשים כדי להבחין בשינויים של 0.1% בעוצמה?

שינוי של 0.1% משמעו שצריך לפחות $1/0.001 = 1000$ רמות בדידות. עם $n$ ביטים יש $2^n$ רמות, ולכן $2^n \geq 1000 \Rightarrow n \geq \log_2(1000) \approx 9.97$.

מכאן: $\boxed{n_{\min} = 10}$ ביטים (1024 רמות). 9 ביטים בלבד נותנים 512 רמות — לא מספיק.

---

## חלק 2 — הפחתת רעשים

### 2.1 חישוב SNR בשתי דרכים

לפי ההרצאה (Lec2_2, slide 14) יש שתי הגדרות מקובלות ל-SNR:

**הגדרה #1 — יחס הספקים (Power ratio):**
$$\text{SNR}_1 = \frac{\sum s_i^2}{\sum n_i^2} = \frac{\text{RMS}(s)^2}{\text{RMS}(n)^2}$$

כאשר $\text{Power} = \frac{1}{n}\sum x_i^2$ (slide 7). בגרסה RMS: $\text{SNR}_{\text{rms}} = \text{RMS}(s) / \text{RMS}(n)$.

**הגדרה #2 — מקרה פרטי (mean / std):**
$$\text{SNR}_2 = \frac{\text{mean}(s)}{\text{std}(n)}$$

תקף כאשר האות חיובי והרעש בעל ממוצע אפס (slide 11).

```python
rms_signal   = np.sqrt(np.mean(stimulus_signal**2))
rms_noise    = np.sqrt(np.mean(noise_signal**2))
snr_method_1 = rms_signal / rms_noise

mean_signal  = np.mean(stimulus_signal)
std_noise    = np.std(noise_signal)
snr_method_2 = mean_signal / std_noise
```

**למה התוצאות שונות?** שתי השיטות מודדות תכונות סטטיסטיות שונות:
- שיטה 1 (RMS/RMS) מודדת את **ההספק הכולל** של האות — סכום ריבועי כל הערכים, כולל ה-DC (התוחלת) וגם ה-AC (התנודות סביבו).
- שיטה 2 (mean/std) מבדילה בין רכיב ה-DC (ממוצע האות) לעוצמת רעש ה-AC (std הרעש).

כאשר לאות הגירוי יש ערך בסיס חיובי (DC offset), שיטה 2 מחזירה ערך שונה משיטה 1 כי היא לוקחת רק את ה-DC של האות ולא את כל ההספק. זה תואם את ההערה ב-slide 13 — הוספת קבוע לאות **לא** משנה את SNR לפי הגדרה #1 (כי שני האותות יוכפלו באותו קבוע פקטור), אבל **כן** משפיעה על הגדרה #2.

---

### 2.2 + 2.3 — Binning (Oversampling)

לפי ההוראה: ממצעים כל $k = 2, 3, 5, 20$ נקודות (לא חלון נע), ומחשבים SNR לפי הנוסחה $\text{SNR} = (P_s - P_n) / P_n$. הפעולה מבוצעת על קטע הגירוי וקטע הרעש בנפרד, וההספק מחושב לפי הגדרת ההרצאה: $P = \frac{1}{n} \sum x_i^2$.

```python
for k in [2, 3, 5, 20]:
    stim_binned  = stim_trimmed.reshape(-1, k).mean(axis=1)
    noise_binned = noise_trimmed.reshape(-1, k).mean(axis=1)

    p_stim  = np.mean(stim_binned  ** 2)
    p_noise = np.mean(noise_binned ** 2)
    snr = (p_stim - p_noise) / p_noise
```

**מקדם השיפור:** לפי ההרצאה (Lec2_2, slides 18, 25), עבור רעש בלתי-תלוי, מיצוע של $k$ דגימות מקטין את סטיית התקן של הרעש בפקטור $\sqrt{k}$. לכן SNR צפוי להשתפר עם $k$.

![Binning בערכי k שונים](graphs/binning_plots.png)

בגרף רואים את האות לאחר binning ב-k=2, 3, 5, 20. עבור k=2-5 צורת התנודה נשמרת והרעש דועך — SNR עולה. **ב-k=20** המצב הפוך: חלון המיצוע הוא $20/50 = 0.4$ שניות, קרוב למחזור שלם של אות הגירוי ($1/2 = 0.5$ שניות). המיצוע ממצע אז את התנודה עצמה לקראת אפס, וצורת האות נהרסת. ה-SNR לפי הנוסחה אולי נראה מסוים, אבל האות הביולוגי כבר אבוד.

---

### 2.4 החלקה מלבנית כפולה (Rectangular ×2)

מסנן מלבני בגודל 3: $h_{\text{rect}} = [1/3, 1/3, 1/3]$. החלקה כפולה מבוצעת על קטע הגירוי בלבד:

```python
rect_window = np.ones(3) / 3
smoothed_rect_1 = np.convolve(stimulus_signal, rect_window, mode='same')
smoothed_rect_2 = np.convolve(smoothed_rect_1, rect_window, mode='same')
```

![החלקה מלבנית כפולה](graphs/rectangular_smoothing.png)

האות הכחול (מקור) מוחלף באות אדום מוחלק. ניתן לראות שתנודות הרעש המהיר נחלשו, בעוד שצורת התנודה של 2 Hz נשמרת.

---

### 2.5 החלקה משולשת (Triangular)

מסנן משולשי בגודל 5: $h_{\text{tri}} = [1, 2, 3, 2, 1] / 9$.

```python
tri_window = np.array([1, 2, 3, 2, 1]) / 9
smoothed_tri = np.convolve(stimulus_signal, tri_window, mode='same')
```

![החלקה משולשת](graphs/triangular_smoothing.png)

האות הירוק (מוחלק במשולש) — דומה מאוד לאות האדום מ-2.4.

---

### 2.6 השוואה והסבר LPF

![השוואה: מלבני כפול מול משולש](graphs/smoothing_comparison.png)

**שני הגרפים זהים.** זו לא במקרה — לפי הנלמד ב-Tirgul 3 וב-Lec2_2 (slide 28):

> "Smoothing twice with a rectangular is the same as once with a triangular"

הסיבה: קונבולוציה היא פעולה אסוציאטיבית, ו-

$$\left[\tfrac{1}{3}, \tfrac{1}{3}, \tfrac{1}{3}\right] * \left[\tfrac{1}{3}, \tfrac{1}{3}, \tfrac{1}{3}\right] = \tfrac{1}{9}[1, 2, 3, 2, 1]$$

כלומר קונבולוציה של שני חלונות מלבניים בגודל 3 שווה בדיוק לחלון המשולשי. לכן `Signal * rect * rect = Signal * (rect*rect) = Signal * tri`. בקוד אכן רואים שההפרש המקסימלי בין שתי השיטות הוא בקנה מידה של $10^{-15}$ — שגיאת עיגול מספרית בלבד.

**למה זה LPF?** לפי משפט הקונבולוציה, החלקה במרחב הזמן שקולה לכפל בפונקציית ההעברה $H(f)$ במרחב התדר. עבור חלון ממוצע נע, $|H(f)|$ דומה ל-sinc — שואף ל-1 ב-$f=0$ ודועך עם תדר. תדרים גבוהים (הרעש) נחלשים, ותדרים נמוכים (אות הגירוי האיטי של 2 Hz) עוברים כמעט בלי שינוי — בדיוק התנהגות של מסנן מעביר נמוכים.

---

## חלק 3 — ניתוח ספייקים (Spike Trains)

קובץ `spike_trains.csv` מכיל מטריצות בינאריות עבור 5 נוירונים (A, B, C, D, E), 100 ניסויים בכל אחד, 30 שניות בתדר 1000 Hz. הטעינה:

```python
df_spikes = pd.read_csv('spike_trains.csv', index_col=0)
def get_neuron_data(df, letter):
    return df[df.index.str.endswith(f'_{letter}')].values
neuron_A = get_neuron_data(df_spikes, 'A')  # shape: (100, 30000)
```

### 3.1 קצב ירי ומדד Fano

קצב ירי ממוצע: $\bar{r} = \langle N_{\text{spikes}} / T \rangle_{\text{trials}}$ ב-Hz. מדד Fano: $\text{FF} = \text{Var}(N) / \text{E}(N)$ של ספירת ספייקים בחלונות של שנייה אחת לרוחב 100 הניסויים.

| נוירון | קצב ירי (Hz) | Fano | סוג נוירון |
|:---:|:---:|:---:|:---|
| A | ראה פלט קוד | ≈1.0 | Poisson |
| B | ראה פלט קוד | ≈1.0 | Poisson (בתדר שונה) |
| C | ראה פלט קוד | <1 | Regular עם jitter |
| D | ראה פלט קוד | ≪1 | Pacemaker (שעון) |
| E | ראה פלט קוד | >1 | Bursting |

הקריטריון: עבור תהליך Poisson $\text{Var} = \text{Mean}$, ולכן FF=1. נוירונים סדירים יורים ב-ISI כמעט קבוע, ולכן הספירה כמעט קבועה והשונות קטנה ($\text{FF}<1$). נוירוני Bursting יורים בפרצות מקובצות, מה שמגדיל את השונות מעבר לתוחלת ($\text{FF}>1$).

**נוירון A:** הוא **Poisson** כי FF≈1 — סימן ישיר לחוסר תלות זמנית בין ספייקים.

---

### 3.2 התפלגות ISI (נוירונים A ו-B)

```python
def calculate_isi(neuron_matrix):
    isis = []
    for trial in neuron_matrix:
        spike_times = np.where(trial == 1)[0]
        if len(spike_times) > 1:
            isis.extend(np.diff(spike_times))
    return np.array(isis)
```

![התפלגות ISI](graphs/isi_distributions.png)

שני הנוירונים מציגים התפלגות מעריכית (exponential decay) — סימן ל-ירי Poisson לפי Tirgul 4: $P(\text{ISI}=t) = \lambda e^{-\lambda t}$. ההבדל ביניהם הוא בקצב הירי: B "דחוס" יותר (ISI עד ~60 ms עיקרי) לעומת A (ISI עד ~250 ms), כלומר B יורה בקצב גבוה יותר. סביב $\text{ISI}=0$ ניתן לזהות תקופה רפרקטורית מוחלטת — אין ספייקים סמוכים.

---

### 3.3 Raster Plots

![Raster Plots](graphs/raster_plots.png)

מציג ספייקים על פני 100 הניסויים בשנייה הראשונה. כל קו אנכי הוא ספייק; ציר Y הוא מספר הניסוי.

- **A:** פיזור אקראי — ירי Poisson.
- **C:** עמודות אנכיות עם פיזור מסוים — ירי סדיר עם jitter קל.
- **D:** עמודות אנכיות צרות וחדות — "שעון ביולוגי".
- **E:** קבוצות צפופות של ספייקים (פרצות) מופרדות בתקופות שתיקה — **נוירון E הוא Bursting**, ההסבר תואם את FF>1 שמצאנו ב-3.1.

---

### 3.4 PSTH (נוירונים A ו-D)

PSTH (Peri-Stimulus Time Histogram) מנורמל ל-Hz: סכום ספייקים בכל bin לחלק ב-$N_{\text{trials}} \cdot \Delta t$.

![PSTH](graphs/psth_plots.png)

- **נוירון A** (bin = 50 ms): קצב ירי יציב סביב 10–14 Hz לאורך כל הזמן — אין זמנים מועדפים, מאחר שהוא Poisson.
- **נוירון D** (bin = 1 ms): פיקים צרים וגבוהים (עד 200–300 Hz) ביני אזורי שתיקה (0 Hz).

**מדוע פיקי D גבוהים מהקצב הממוצע?** קצב הירי הממוצע (~50 Hz) הוא הממוצע על פני כל הניסוי, כולל תקופות שקטות בין ספייקים. ה-PSTH ב-bin של 1 ms מודד **קצב ירי רגעי**: אם D יורה באותה אלפית-שנייה ב-30 מתוך 100 ניסויים (כי הוא מסונכרן), אז $\text{PSTH} = 30 / (100 \cdot 0.001) = 300\,\text{Hz}$. ערך זה משקף **דיוק זמני** של נוירון D, לא קצב ירי גבוה מהממוצע באותה אלפית-שנייה.

---

### 3.5 אוטוקורלציה (A, C, D, E)

חישבתי אוטוקורלציה על 2 דקות (120,000 דגימות) שטוחות, מנורמלת ביחס לציפייה של תהליך Poisson בלתי-תלוי:
$$\text{expected\_coincidences} = \bar{\lambda}^2 \cdot N_{\text{samples}}$$

כך שקו ה-baseline בגרף הוא 1.0.

![אוטוקורלציה מלאה](graphs/autocorr_full.png)

![אוטוקורלציה מוגדלת 0-15ms](graphs/autocorr_zoomed.png)

**זיהוי תקופה רפרקטורית:** בגרף המוגדל ברזולוציית 1 ms, רואים שעבור כל הנוירונים יש אזור של ערך ~0 קרוב ל-lag=0 — זו התקופה הרפרקטורית המוחלטת. הערך המספרי מודפס בפלט הקוד (`Refractory end ≈ X ms`). באופן טיפוסי 1–3 ms.

**מספר הפיקים:** עבור נוירון סדיר בקצב $r$ Hz, הפיקים מופיעים כל $1/r$ שניות (כי ספייק לאחר ISI ידוע). ספירת פיקים מאפשרת אומדן לקצב הירי.

**מינימום מקומי אחרי כל פיק:** ערך מתחת ל-1.0 משמעו שב-lag הזה יש פחות ספייקים מהציפייה האקראית — דיכוי פוסט-ספייק (לאחר ספייק, הנוירון פחות צפוי לירות מיד אחר כך).

**C מול D:** שניהם סדירים, אבל ל-D פיקים חדים וצרים (jitter מינימלי), ול-C פיקים רחבים יותר (jitter גדול יותר). זה תואם את FF הנמוך מאוד של D לעומת FF נמוך-מ-1-אבל-לא-אפס של C.

**צורת נוירון E סביב lag 0:** התקופה הרפרקטורית קצרה, ומיד אחריה יש פיק גבוה (יחסית ל-baseline) המייצג את ירי הספייקים בתוך הפרצה. אחר כך יש שקיעה ארוכה עד מתחת ל-baseline — זמן השתיקה בין הפרצות.

---

### 3.6 קורלציה צולבת (CCG) — ללא נרמול

לפי ההוראה (וכמודגש בהבהרת המתרגלת), הגרפים מוצגים **ללא נרמול** — ספירות גולמיות של צמדי-ספייקים.

```python
def calc_cross_corr_flat(ref, target, max_lag=100):
    N = len(ref); lags = np.arange(-max_lag, max_lag + 1)
    cc = np.zeros(len(lags))
    for j, lag in enumerate(lags):
        if lag > 0:    cc[j] = np.sum(ref[:N-lag] * target[lag:])
        elif lag < 0:  cc[j] = np.sum(ref[-lag:] * target[:N+lag])
        else:          cc[j] = np.sum(ref * target)
    return lags, cc
```

#### A → B

![CCG: A → B](graphs/crosscorr_AB.png)

**ניתוח:** הפיק החיובי המובהק ב-lag חיובי קטן (~2 ms) מעיד על:
- **קיים קשר** (פיק חורג מה-baseline משמעותית).
- **קשר מעורר (Excitatory)** — הפיק חיובי.
- **כיוון:** lag חיובי → B יורה **אחרי** A → **A הוא פרה-סינפטי**, A מעורר את B.
- **Latency** ≈ ערך peak_lag המודפס בפלט (~2 ms).

**גובה הפיק וקצב הירי של A:** אם A מעורר את B באמינות 100%, אז כל ספייק של A מייצר ספייק ב-B בלאג קבוע. גובה הפיק הנקי (peak − baseline) שווה למספר הספייקים של A בתקופת ההקלטה. לכן:
$$\text{Estimated firing rate}(A) = \frac{\text{net peak}}{\text{recording duration}}$$
ניתן להשוות את האומדן הזה לקצב הירי שמדדנו ישירות בסעיף 3.1.

**אמינות 50% במקום 100%:** אם רק חצי מהספייקים של A מעוררים את B, אז גובה הפיק הנקי יורד **בדיוק לחצי**. הקשר ליניארי: $\text{Peak}(p) = p \cdot \text{Peak}(1)$.

#### C → D

![CCG: C → D](graphs/crosscorr_CD.png)

**ניתוח:** הגרף מציג פיקים מחזוריים בהפרשים קבועים — סימן לסינכרון מחזורי בין שני pacemakers. שיאים מחזוריים בלבד **לא מבטיחים** קשר ישיר; ייתכן שמדובר ב-**קלט משותף** (common input) משלישי. כדי להבחין: גרף סימטרי לחלוטין סביב 0 מצביע על קלט משותף; אסימטריה (פיק עיקרי באחד הצדדים) מצביעה על כיוון קשר ישיר.

---

## סיכום

- **חלק 1:** $F_s = 50\,\text{Hz}$, חלוקה לקטעים, resampling ידני (`data[::int]`) ל-25, 10, 3⅓ Hz; ניתוח aliasing לפי $f_{\text{alias}} = |f_{\text{signal}} - N F_s'|$; קוונטיזציה מינימלית ל-10 ביט.
- **חלק 2:** שתי הגדרות SNR (יחס הספקים ו-mean/std); binning עם $\sqrt{k}$ improvement עד שהחלון מכסה מחזור של הגירוי; הוכחת זהות בין החלקה מלבנית כפולה לבין החלקה משולשת.
- **חלק 3:** סיווג חמשת הנוירונים לפי Fano (Poisson, Pacemaker, Bursting); זיהוי תקופה רפרקטורית באוטוקורלציה; זיהוי קשר A→B עם latency וכיוון; דיון בקשר C-D.

🔗 [Notebook ב-Colab](https://colab.research.google.com/github/Royc4515/Project1_SignalProcessing/blob/main/Project1_SignalProcessing.ipynb)

</div>
