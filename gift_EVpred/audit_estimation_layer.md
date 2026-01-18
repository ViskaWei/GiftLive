# Estimation Layer Audit Report

> **Experiment ID:** EXP-20260118-gift_EVpred-08  
> **MVP:** MVP-1.6  
> **Date:** 2026-01-18T15:04:42.588078  
> **Author:** Viska Wei

---

## 1. Estimation Layer Target Definition

### 1.1 r_rev (Short-term Revenue) - MUST IMPLEMENT

**Definition:**
$$$r^{rev}_t(a) = \mathbb{E}[\text{gift\_amount}_{t:t+H} \mid u, s, \text{ctx}_t, a]$$$

**Scope:** This round: MUST implement

**Action Mapping:** action (a) = "expose/guide user u to streamer s in live_id"

**Data Fields:**
- User: user_id
- Streamer: streamer_id
- Live: live_id
- Context: timestamp, hour, day_of_week, is_weekend
- Outcome: gift_price within H hours after click

### 1.2 r_usr (User Long-term Proxy) - DEFINITION ONLY

**Definition:**
$$$r^{usr}_t(a) = \mathbb{E}[\text{return}_{t+1d} \mid u, \text{history}, a]$ or $\mathbb{E}[\Delta \text{watch\_time}, \Delta\text{engagement} \mid a]$$$

**Scope:** This round: Definition only, NOT training

### 1.3 r_eco (Ecosystem/Externality) - DEFINITION ONLY

**Definition:**
$$$r^{eco}_t(a) = -\lambda \cdot \text{concentration}(\text{exposure/revenue})$ or concave utility $U_s(x)$$$

**Scope:** This round: Definition only, NOT training

---

## 2. Sample Unit Definition

### 2.1 Click-Level (IMPLEMENTED)

**Definition:** One click = one opportunity (decision proxy)

**Input:** (u_i, s_i, live_i, ctx_i, past_only_features_at_t_i)

**Label:** Y_i = sum of gift_amount within [t_i, t_i+H] for matching (u,s,live)

**Includes Zero:** True

**Note:** Y=0 means no gift within H hours after click

### 2.2 Session-Level (DEFINITION ONLY)

**Definition:** session = continuous viewing segment of same live_id (spliced by time gap threshold)

**Note:** Requires session construction logic (time gap threshold)

### 2.3 Impression-Level (DEFINITION ONLY)

**Definition:** One impression = one exposure opportunity (includes non-clicks)

**Data Requirement:** Requires impression/non-click logs (not available in KuaiLive)

**Note:** Strictest definition for allocation decisions

---

## 3. Train/Val/Test Time Split Boundaries

### 3.1 Split Method

**Method:** Temporal split by timestamp (70/15/15 ratio)

**Is Week1/2/3:** Train spans 13 days, Val spans 3 days, Test spans 3 days

### 3.2 Train Split

- **Samples:** 3,436,660
- **Time Range:** 2025-05-04T16:00:00.022000 to 2025-05-18T09:12:23.408000
- **Unique Users:** 23,360
- **Unique Streamers:** 371,441
- **Unique Lives:** 1,319,817
- **Positive Rate:** 1.42%
- **Y P50/P90/P99:** 0.00 / 0.00 / 1.00
- **Watch Time P50:** 4.0s

### 3.3 Val Split

- **Samples:** 736,427
- **Time Range:** 2025-05-18T09:12:23.445000 to 2025-05-22T01:45:01.666000
- **Positive Rate:** 1.72%
- **Watch Time P50:** 4.6s

### 3.4 Test Split

- **Samples:** 736,428
- **Time Range:** 2025-05-22T01:45:01.947000 to 2025-05-25T14:59:58.515000
- **Positive Rate:** 1.64%
- **Watch Time P50:** 4.3s

### 3.5 Verification

- **Train-Val Gap:** 0.0 hours
- **Val-Test Gap:** 0.0 hours
- **No Overlap:** True

---

## 4. Watch Time vs Label Window Consistency

### 4.1 Comparison Results

**Label Window:** 1h

**Overall Difference Ratio:** 16.51%

**Total Fixed Window:** 6,555,716

**Total Watch Time Cap:** 5,473,569

**Samples with Different Labels:** 7,041 (0.14%)

### 4.2 Bucket Analysis


**<5s:**
- Samples: 2,646,662
- Difference Ratio: 65.57%
- Total Fixed: 112,513
- Total Cap: 38,739


**5-30s:**
- Samples: 1,193,391
- Difference Ratio: 68.32%
- Total Fixed: 168,336
- Total Cap: 53,322


**30-300s:**
- Samples: 794,859
- Difference Ratio: 32.33%
- Total Fixed: 1,301,552
- Total Cap: 880,732


**>300s:**
- Samples: 274,603
- Difference Ratio: 9.50%
- Total Fixed: 4,973,315
- Total Cap: 4,500,776


### 4.3 Verdict

**Verdict:** SIGNIFICANT_IMPACT

🔴 Impact is SIGNIFICANT - Watch_time truncation is recommended.

---

## 5. Frozen Past-Only Baseline

### 5.1 Feature System

**Version:** Frozen (train window statistics only)

**Lookup Statistics:**
- Pairs: 37,595
- Users: 18,402
- Streamers: 25,890

**Feature List:**
- pair_gift_count_past
- pair_gift_sum_past
- pair_gift_mean_past
- pair_last_gift_time_gap_past
- user_total_gift_7d_past
- user_budget_proxy_past
- streamer_recent_revenue_past
- streamer_recent_unique_givers_past

### 5.2 Verification

**Method:** Frozen: Only use train window statistics

**Leakage Risk:** LOW (no future information)

---

## 6. Minimal Baseline Models Results

### 6.1 Logistic Regression (Binary Classification)

#### Set-0 (Time Context Only)

**Test Metrics:**
- PR-AUC: 0.0165
- LogLoss: 0.0840
- ECE: 0.0026
- RevCap@1%: 0.0026

#### Set-1 (Time Context + Past-Only)

**Test Metrics:**
- PR-AUC: 0.0570
- LogLoss: 0.1077
- ECE: 0.0135
- RevCap@1%: 0.1781

### 6.2 Ridge Regression (Regression)

#### Set-0 (Time Context Only)

**Test Metrics:**
- MAE_log: 0.0576
- RMSE_log: 0.3222
- Spearman: -0.0012
- RevCap@1%: 0.0103

#### Set-1 (Time Context + Past-Only)

**Test Metrics:**
- MAE_log: 0.0451
- RMSE_log: 0.3221
- Spearman: 0.0342
- RevCap@1%: 0.2594

### 6.3 Sanity Checks

**Train vs Test Comparison:**
- Logistic PR-AUC: Train=0.3126 > Test=0.0570 ✅
- Linear MAE_log: Train=0.0396 < Test=0.0451 ✅

**Leakage Check:**
- No suspiciously perfect scores (AUC < 0.999) ✅

---

## 7. Conclusions

### 7.1 Can Current Definition Serve Online Allocation?

**Answer:** YES

**Reasoning:**
- Prediction target (r_rev) is clearly defined ✅
- Sample unit (click-level) includes zeros ✅
- Time split is properly implemented ✅
- Features are leakage-free (Frozen past-only) ✅
- Simple models can learn meaningful patterns: RevCap@1% = 25.94%

### 7.2 Next Steps

**Recommended Changes:**

1. **Label Window:** Consider watch_time truncation or shorter H (10min/30min)

2. **Feature Engineering:** Current features are sufficient for baseline

3. **Model Complexity:** Simple models work, can proceed to LightGBM

---

**Report Generated:** 2026-01-18T15:04:55.710795
