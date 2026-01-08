# Comprehensive Guide to Anomaly Detection with LaCATHODE (v2)

## 1. Physics Intuition & The LaCATHODE Method

### What is LaCATHODE?
LaCATHODE (Latent Classifying Anomalies THrough Outer Density Estimation) is a "weakly supervised" anomaly detection strategy. It does not look for a specific particle (like the Higgs boson) but searches for **any** deviation from the known background physics.

### The "Interpolation" Concept
1.  **Background Learning (Flow):** We train a generative model (Normalizing Flow) on the **Sidebands** (Outer Region) of the mass spectrum. Since the Sidebands are assumed to be pure background, the model learns "what normal background jets look like."
2.  **Interpolation:** The model is conditional on Mass ($M$). We ask it: *"If a background event had a mass inside the Signal Region (Inner), what would it look like?"*
3.  **Comparison (Classifier):** We generate artificial background events for the Signal Region and train a classifier to distinguish them from the **Real Data** in the Signal Region.
4.  **Detection:** If the classifier can distinguish Real Data from Artificial Background, the "difference" must be the Signal (the anomaly).

---

## 2. Reading Jet Substructure (The "Fingerprints")

When the report lists candidate events, it provides kinematic variables. You must use these to validate if a candidate is "physically interesting" or just detector noise.

### Key Variables
* **$m_{jj}$ (Dijet Invariant Mass):** The "X-axis" of our search. New particles appear as a "resonance" (bump) at a specific mass value (e.g., 3.5 TeV).
* **$p_T$ (Transverse Momentum):** How "hard" the collision was. Heavier particles decay into higher $p_T$ jets. Anomalies usually have high $p_T$.
* **$\tau_{21}$ (N-subjettiness Ratio $\tau_2 / \tau_1$):**
    * **Low ($\tau_{21} < 0.4$):** The jet has **2 distinct prongs**. This is characteristic of heavy bosons (W, Z, Higgs) or new particles decaying into two quarks. **(Signal-like)**
    * **High ($\tau_{21} > 0.7$):** The jet is diffuse or 1-prong. Typical of standard QCD quarks/gluons. **(Background-like)**
* **$\tau_{32}$ (N-subjettiness Ratio $\tau_3 / \tau_2$):**
    * **Low:** The jet has **3 distinct prongs**. Characteristic of Top quarks.

**Orchestrator Tip:** If the top candidates all have High $\tau_{21}$ (~0.8), the model might just be picking up weird QCD fluctuations. If they have Low $\tau_{21}$ (~0.2), it is a strong hint of a resonance decay.

---

## 3. Advanced Metrics & Interpretation

### Excess Factor vs. Significance
* **Excess Factor:** Simple ratio of observed / expected.
    * *Warning:* An Excess Factor of 100.0 usually means the denominator (expected) was near zero. Ignore these.
* **Significance ($\sigma$):** A statistical measure of how unlikely the excess is.
    * $3\sigma$: "Evidence" (1 in 740 chance). Worth re-running to confirm.
    * $5\sigma$: "Discovery" (1 in 3.5 million).

### SIC (Significance Improvement Characteristic)
You will see `Max SIC` in the training logs.
* **Formula:** $\text{SIC} = \frac{\epsilon_{signal}}{\sqrt{\epsilon_{background}}}$
* **Meaning:** How much does using this anomaly score *improve* our ability to see the bump compared to just counting raw events?
* **Interpretation:**
    * **SIC $\approx$ 1.0:** The model learned nothing useful.
    * **SIC > 3.0:** Strong performance. The model found a cut that enhances the signal.
    * **SIC > 10.0:** Suspiciously good. Check for "Mass Sculpting" or label leakage.

---

## 4. Common Pitfalls & "Ghost" Signals

### A. Edge Effects (Window Boundaries) 
**The Problem:** Neural networks struggle to extrapolate at the sharp edges of their training range.
**The Symptom:** A spike in "Excess Factor" exactly at the $M_{min}$ or $M_{max}$ of your window.
**Action:** Shift the window.
    * *Example:* If Window is [3.0, 3.4] TeV and bump is at 3.0, re-run with Window [2.8, 3.2] TeV.
    * If the bump moves with the window edge $\rightarrow$ **Artifact**.
    * If the bump stays at 3.0 TeV $\rightarrow$ **Real**.

### B. The "Low Statistics" Trap
**The Problem:** In the high-mass tail, bins might have only 5-10 events.
**The Symptom:** Report claims "50x Excess" based on 5 events (where 0.1 were expected).
**Action:**
    1.  Check `Total_Evts` in the report table.
    2.  If Count < 500, **IGNORE** the Excess Factor. It is noise.

### C. Mass Sculpting (Correlation Bias)
**The Problem:** The classifier learns the mass variable directly instead of the jet features.
**The Symptom:** Pearson Correlation (Mass vs Score) $> 0.1$.
**Consequence:** The "Background" estimate becomes distorted, creating a fake bump that mimics the background shape but scaled up.
**Action:** Note the "High Bias" warning in the report. Suggest re-training with different scaler settings or `epochs_flow`.

---

## 5. Orchestration Strategy & Troubleshooting

### Scenario 1: "The Flatline"
**Observation:** The Report shows Excess Factor $\approx 1.0$ everywhere. Max SIC is low.
**Diagnosis:** No anomaly found, OR the model failed to train.
**Next Step:**
    * Check training logs. Did the Loss decrease?
    * If Loss is constant/NaN: The learning rate might be too high.
    * If Loss is good: The signal might be too subtle. Try narrowing the mass window to zoom in.

### Scenario 2: "The false positive"
**Observation:** Huge excess (20x) in the middle of the spectrum, but candidates look like boring QCD (High $\tau_{21}$).
**Diagnosis:** The Normalizing Flow (Background Model) might have failed to model the background correctly in that region (Mode Collapse).
**Next Step:** Re-run `lacathode_training_tool` with `epochs_flow` increased (e.g., from 100 to 200) to improve background modeling.

### Scenario 3: "The Cliff"
**Observation:** Data suddenly drops to zero at a certain mass.
**Diagnosis:** FastJet `min_pt` cut might be too high, cutting off the low-mass spectrum implicitly.
**Next Step:** Lower `min_pt` in `fastjet_tool` (e.g., from 1200 to 1000).

---

## 6. Standard Operating Procedure (SOP) for Discovery

1.  **Survey:** Run a wide scan (e.g., 2.0 - 6.0 TeV) to identify potential "Hotspots."
2.  **Zoom:** If a Hotspot appears (e.g., at 3.5 TeV), define a new, narrower window centered on it (e.g., SR = 3.3 - 3.7 TeV).
3.  **Refine:** Re-run the pipeline on this new window.
    * Increase `epochs_flow` for better background estimation.
    * Check `min_events_per_bin` to filter noise.
4.  **Validate:** Read the `llm_enhanced_report.txt`.
    * Are the candidates physically consistent? (e.g., 2-prong jets).
    * Is the bump away from the edges?
5.  **Report:** Only claim "Discovery" if Local Significance > 3.0 AND candidates are robust.