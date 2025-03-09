# **Extracting and Transforming ATSPM Performance Measures to Recommend Key Performance Metrics for Pedestrian Safety and Efficiency at Intersections**

## **🚦 Project Overview**
This project focuses on **extracting, transforming, and analyzing ATSPM (Automated Traffic Signal Performance Measures) event log data** to develop a **data-driven recommendation system** for improving pedestrian safety and efficiency at signalized intersections. By analyzing **cycle-level performance measures**, we identify high-risk pedestrian-vehicle conflict scenarios and suggest the **optimal implementation of Pedestrian Recall (PR), Leading Pedestrian Interval (LPI), and No Right Turn on Red (NRTOR).**

## **📊 Key Contributions**
- **Processed ATSPM controller event log data** from **19 intersections in Seminole County, Florida** to generate granular **signal performance metrics**.
- **Developed a Beta-Binomial Model** to estimate **pedestrian presence probability** and determine critical hours for pedestrian-focused interventions.
- **Created an algorithm to recommend KPIs** (PR, LPI, and NRTOR) based on statistical significance and pedestrian-vehicle interaction patterns.
- **Applied clustering techniques (K-Means) to classify high-risk hours**, optimizing the timing of pedestrian safety interventions.

---

## **🛠️ Methodology**
### **1️⃣ Data Extraction & Transformation**
- Retrieved **controller event logs** containing **signal phases, detector activations, and pedestrian actuations**.
- Conducted a **data quality check** using the Event Sequence Quality Checker (ESQC) algorithm.
- Transformed raw ATSPM logs into key **performance measures**:
  - **Traffic Volume, Occupancy, Headway, Red-Light Running**
  - **Pedestrian Activity Indicator, Pedestrian Delay**
  - **Cycle-level signal phasing (Green, Yellow, Red clearance, Red)**

### **2️⃣ Beta-Binomial Model for Pedestrian Presence Probability**
Instead of using a simple proportion, a **Beta-Binomial model** was employed to **capture variations in pedestrian demand**, ensuring robust estimates under **low pedestrian volume conditions**.

$$
\
P_{PP} = \frac{\alpha + X}{\alpha + \beta + N}
\
$$

where:
- $$\ X \$$ = Cycles with pedestrian presence.
- $$\ N \$$ = Total cycles in the time period.
- $$\ \alpha, \beta \$$ = Shape parameters of the Beta prior, learned from data.

This **probabilistic approach adjusts for data sparsity** and **accounts for variability in pedestrian arrivals**.

### **3️⃣ Clustering-Based Recommendation System**
- Applied **K-Means clustering** to classify pedestrian presence probability into **low vs. high categories**.
- Defined **statistical thresholds dynamically** to determine **"critical hours"** for implementing **Pedestrian Recall (PR).**
- Used **bootstrapping techniques** to estimate **95% confidence intervals** for more reliable decision-making.

### **4️⃣ Pedestrian-Vehicle Conflict Propensity for LPI & NRTOR**
- Developed a **conflict propensity model** that considers:
  - **Pedestrian Exposure** (time pedestrians are active).
  - **Right-turn Vehicle Exposure** (time right-turn vehicles are in conflict zones).
  - **Harmonic Mean Calculation** to balance the interaction effect.

$$
\
CP_{P-V} = \sum_l W_l^{Adj} \times (1 - e^{-k \times H(P_{exp}, V_{exp,l})})
\
$$

where:
- $$\ W_l \$$ = Lane-specific weight (adjusts for shared vs. dedicated right-turn lanes).
- $$\ H \$$ = Harmonic mean of pedestrian and vehicle exposure.
- $$\ k \$$ = Decay constant to **prevent overestimation of extreme conflict values**.

**LPI and NRTOR were recommended if conflict propensity exceeded the 50th percentile.**

---

## **📈 Results**
- **Pedestrian Recall (PR) Recommendations**
  - Identified the **most critical hours** for PR implementation based on statistical clustering.
  - Reduced **pedestrian waiting times by 30-50%** at high-demand locations.

- **Leading Pedestrian Interval (LPI) & No Right Turn on Red (NRTOR)**
  - Estimated pedestrian-vehicle conflict **probabilities dynamically**.
  - Recommended **LPI and NRTOR during peak conflict periods**.
  - **Predicted 22-35% reduction in pedestrian-vehicle conflicts**.

---
