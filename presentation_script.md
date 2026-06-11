# Presentation Script — Fraud and Abnormal Order Detection in Supply Chain ERP Transactions

**Total target time: 14–15 minutes of speaking, followed by Q&A.**

This script matches the slide order in `project_progress_presentation.tex` (15 slides).
Each section gives: the time budget, what to say (you can read it almost verbatim or
paraphrase), a transition line into the next slide, and likely questions for that slide.

> Tip: to compile a version of the deck with the embedded notes visible, open
> `project_progress_presentation.tex` and replace `\setbeameroption{hide notes}` with
> `\setbeameroption{show notes on second screen=right}`.

---

## Slide 1 — Title (45–60 seconds)

**Say:**

"Good morning everyone. My name is Varad Dharmadhikari, from the Information Technology
branch. Today I am presenting the progress of my major project, titled *Fraud and
Abnormal Order Detection in Supply Chain ERP Transactions*, carried out under the
guidance of Ms. Ramyashree as my internal guide.

The project applies machine learning to detect suspicious orders in ERP-style supply
chain transaction data. It is not only a model — it is a complete working system, with a
web interface, a backend API, and optional cloud storage on Microsoft Azure."

**Transition:** "Let me first give you an outline of the presentation."

**Likely question — Why did you choose this topic?**
Fraud and abnormal transaction detection is a practical, high-impact problem in ERP and
supply chain systems, and it let me combine machine learning, backend development,
frontend integration, and cloud deployment in one end-to-end project.

---

## Slide 2 — Presentation Outline (30 seconds)

**Say:**

"I will begin with the motivation and the literature survey, then define the problem and
objectives. After that I will explain the methodology and system architecture, show the
work completed so far and the current experimental results, and finish with the
remaining work, the conclusion, and references."

**Transition:** "Starting with why this problem matters."

---

## Slide 3 — Introduction and Motivation (1 minute)

**Say:**

"ERP and supply chain systems process very large volumes of orders, refunds, returns,
and shipping transactions every day. Fraudulent or abnormal behaviour usually hides
inside records that look perfectly legitimate — for example, a high-value order from a
brand-new account, an unusual discount, or a mismatch between the billing and shipping
country.

Reviewing these manually is slow, inconsistent, and impossible to scale. Machine
learning can help by assigning every transaction a fraud risk score, so analysts can
focus on the most suspicious records first. And for the solution to be practical, it has
to run both on a local machine and in an online cloud environment — which is exactly
what this project does."

**Transition:** "Before building anything, I surveyed the existing literature."

**Likely question — What do you mean by an "abnormal order"?**
An abnormal order is not necessarily proven fraud. It is a transaction that deviates
from expected business behaviour and deserves review — for example, a very high-value
order from a newly created account with urgent shipping. It may be legitimate, but it is
suspicious enough to flag.

---

## Slide 4 — Literature Survey and Research Gap (1.5 minutes)

**Say:**

"From the fraud-detection surveys, four recurring challenges stand out: severe class
imbalance, delayed labels, concept drift, and the complexity of actually deploying these
systems. The literature consistently shows that ensemble methods such as Random Forest
are strong baselines for structured tabular fraud data. Hybrid approaches — for example
an autoencoder combined with a Random Forest — can handle more complexity, but at a
much higher implementation cost.

Most published work concentrates on credit card fraud. ERP-style supply chain order
fraud is comparatively under-explored, and very few academic works present a complete
upload-to-prediction-to-cloud pipeline. That is the gap my project addresses: an
interpretable, modular, Azure-compatible fraud detection system for ERP order data."

**Transition:** "Based on that gap, I defined the problem formally."

**Likely question — Why Random Forest instead of deep learning?**
Random Forest is a strong, well-established baseline for structured tabular data. It is
easier to train, interpret, and deploy. Deep models need more data and tuning and reduce
explainability. I plan to compare stronger models after establishing this baseline.

---

## Slide 5 — Problem Definition (1 minute)

**Say:**

"The research problem, in one sentence: design and implement an accurate, explainable,
and cloud-capable machine learning system that detects suspicious ERP-style supply chain
transactions from uploaded CSV data.

The problem is difficult for two kinds of reasons. On the data side: the classes are
imbalanced, values can be missing or inconsistent, and the fields mix numeric and
categorical types, while the output still needs to be interpretable. On the system
side: the solution must ingest CSV files, perform feature engineering, serve predictions
through a web interface, and run both locally and on Azure."

**Transition:** "From this problem statement I derived concrete objectives."

**Likely question — Is this fraud detection or anomaly detection?**
The current implementation is supervised fraud detection, because training uses a
labelled `fraud_label` field. But the engineered features capture suspicious behaviour
patterns, so the system conceptually supports abnormal-order detection and could be
extended with anomaly-detection methods.

---

## Slide 6 — Objectives (45 seconds)

**Say:**

"The objectives are technical milestones. First, build the fraud detection pipeline
itself: preprocess and validate uploaded CSVs, engineer business-interpretable fraud
features, and train a baseline model that produces a fraud risk score. Second, make it
usable: expose predictions through a FastAPI backend and a browser frontend. Third, make
it deployable: support both local execution and Azure-based deployment, with optional
persistence of outputs in Azure Blob Storage.

The key point is that the project is not just about the model — it is about a complete
working pipeline."

**Transition:** "This slide shows how that pipeline actually works, end to end."

---

## Slide 7 — Methodology: End-to-End Pipeline (1.5 minutes)

**Say:**

"Walking through the diagram from left to right: the user uploads a CSV of
transactions. The preprocessing stage standardises column names, validates the required
schema, removes duplicates, and imputes missing values. Feature engineering then
converts raw ERP fields into fraud-relevant indicators — billing–shipping mismatch,
discount behaviour, the refund-to-order ratio, the customer's return ratio, and odd-hour
ordering patterns.

The trained model then scores every record, producing a `fraud_risk_score` between 0 and
1 and a binary `suspicious_flag`. The results are saved as a CSV. Finally, there is a
decision point: if Azure storage is configured, the output file is also uploaded to Blob
Storage; if not, the system simply keeps the local output. This makes the cloud step
optional by design — the system never breaks just because Azure is not configured."

**Transition:** "That optionality is clearer when comparing the two deployment modes."

**Likely question — Why is feature engineering important here?**
Raw ERP fields alone rarely expose suspicious intent. Engineered features such as
shipping mismatch and refund-to-order ratio convert raw values into fraud-relevant
business signals, which both improves learning and makes the output explainable.

---

## Slide 8 — Architecture: Local and Azure Versions (1.5 minutes)

**Say:**

"In local mode, the frontend is served on the local machine, the FastAPI backend runs at
localhost, the model is loaded from the local `models` folder, and results go to a local
`outputs` folder. This is the development and testing configuration.

In Azure mode, the frontend calls a backend deployed on Azure App Service, the storage
connection is configured through App Service application settings, and the output CSV is
uploaded to Azure Blob Storage.

One deliberate security decision: only the backend ever holds the Azure storage
credential. The frontend remains completely secret-free, because anything in the browser
is visible to the user."

**Transition:** "Here is what has been completed so far against that design."

**Likely question — Why shouldn't the frontend hold the Blob connection string?**
The frontend runs in the browser, so any secret stored there can be extracted. Secrets
belong server-side; in Azure App Service the connection string is injected as an
environment variable and used only by the backend.

---

## Slide 9 — Work Done So Far (1 minute)

**Say:**

"All foundational modules are implemented and working together: the synthetic ERP
dataset with fraud labels, the preprocessing and validation module, the engineered fraud
features, the trained Random Forest baseline, the FastAPI backend with health and
predict endpoints, the CSV-upload frontend, and the optional Azure Blob upload of the
generated prediction file.

So the current status is: the core end-to-end workflow is fully functional in local
mode, and Azure storage support is integrated."

**Transition:** "Now, the part you are probably most interested in — the numbers."

**Likely question — Which modules are complete and which are partial?**
Dataset generation, preprocessing, feature engineering, training, backend, frontend, and
local output generation are complete. Azure Blob integration works, but user feedback
and monitoring around the cloud upload can still be improved.

---

## Slide 10 — Current Results (1.5 minutes)

**Say:**

"On a 1000-row test with 97 fraud-positive records, the baseline achieves an accuracy of
0.905 and a ROC-AUC of 0.933. The high AUC means the model ranks risky transactions
well — genuinely fraudulent records tend to receive higher risk scores than legitimate
ones.

But I want to be honest about the weakness: recall at the default threshold is only
0.053, with precision at 0.50 and an F1 of 0.095. In other words, the model is a good
*ranker* but a poor *classifier* at the default cut-off — it misses most fraud cases.

Two caveats frame these numbers. First, the dataset is synthetic, so these results
validate the pipeline rather than prove real-world performance. Second, the low recall
is a threshold and imbalance issue, not a broken pipeline — and fixing it is explicitly
part of the remaining work."

**Transition:** "Let me interpret what these numbers actually tell us."

**Likely question — Why is accuracy high but recall low?**
The dataset is imbalanced and the default 0.5 decision threshold is conservative.
Classifying the many legitimate records correctly keeps accuracy high, while only a
small fraction of true fraud cases cross the threshold, which keeps recall low. That is
exactly why threshold tuning and imbalance handling are the next steps.

---

## Slide 11 — Key Observations (1 minute)

**Say:**

"Three observations. First, the most important features according to the model are
shipping mismatch, refund-to-order ratio, and customer return ratio — which are also the
features a human fraud analyst would consider intuitive, and that agreement increases
confidence in the model. Second, ranking quality and classification threshold are
separate issues: the ranking is already strong, the threshold is what needs work. Third,
even in its current form, the system is useful as a fraud-screening prototype that
prioritises records for analyst review.

The main takeaway: the baseline is a strong starting point, but recall and operational
robustness must improve."

**Transition:** "Which brings me to the remaining work."

**Likely question — How do you know which features are important?**
From the Random Forest feature importances. The top signals — shipping mismatch,
refund-to-order ratio, return ratio, order amount, account age — are also intuitive from
a domain perspective.

---

## Slide 12 — Remaining Work and Scope (1.25 minutes)

**Say:**

"The remaining work has a clear path: tune the decision threshold, apply
class-imbalance techniques, compare the baseline against stronger models such as
XGBoost and anomaly-detection methods, add cross-validation, improve frontend feedback
on the upload result, and strengthen monitoring on the Azure side.

In terms of scope, the system is relevant to manufacturing and retail ERP environments,
audit and compliance teams, supply chain analytics teams, and small or medium
enterprises that need low-cost fraud screening — as well as academic and teaching use."

**Transition:** "To conclude."

**Likely question — What is the most important next step?**
Improving recall while preserving reasonable precision: threshold tuning, imbalance
strategies, and comparing the Random Forest baseline with stronger models.

---

## Slide 13 — Conclusion (1 minute)

**Say:**

"To summarise: a complete fraud detection pipeline for ERP-style supply chain
transactions has been designed and implemented. It supports CSV upload, preprocessing,
feature engineering, prediction, result display, and optional Azure Blob upload. The
baseline model shows strong ranking ability, and its low recall defines the next phase
of improvement rather than a dead end. Overall, the project combines machine learning,
software engineering, and cloud deployment into one working system."

**Transition:** "These are the key references that grounded the work."

**Likely question — What is the core contribution?**
The integration of an explainable fraud-detection pipeline with a usable web
architecture supporting both local and Azure deployment — a complete working system,
not just a model experiment.

---

## Slide 14 — References (20–30 seconds)

**Say:**

"The work is grounded in fraud-detection surveys, ensemble-learning literature, and
supply-chain-specific fraud research — Abdallah et al.'s fraud detection survey,
Sorournejad et al. on credit card fraud techniques, Ahmed et al. on anomaly detection in
the financial domain, Sulaiman et al.'s review of machine learning for credit card
fraud, and Seify et al. on fraud detection in supply chains."

---

## Slide 15 — Thank You / Q&A

**Say:** "Thank you. I am happy to take questions."

### Prepared answers for common viva questions

**1. Why did you use synthetic data?**
Real ERP fraud datasets are very hard to obtain due to privacy, access restrictions, and
confidentiality. Synthetic data gave me a controlled dataset with meaningful
fraud-related patterns, so I could validate the architecture, preprocessing, feature
engineering, and baseline model end to end. Evaluating on more realistic data is part of
the remaining work.

**2. Why is Azure Blob Storage used?**
For persistent cloud storage of the generated prediction outputs. It lets the system
store results outside the local machine and supports the online deployment scenario
where outputs can be saved and shared reliably. It is optional by design — without the
connection string, the system still works locally.

**3. What are the limitations of the current work?**
Synthetic data, low recall at the default decision threshold, and limited
production-level monitoring and security features. All three are explicitly identified
as remaining work.

**4. How can the model be improved?**
Threshold tuning, feature refinement, class-imbalance techniques, hyperparameter
optimisation, and comparison with gradient-boosted trees or anomaly-detection methods.

**5. Is this solution explainable?**
Yes, to a useful extent: the features are business-interpretable and Random Forest
provides feature importances. SHAP could be added later for record-level explanations.

**6. What happens if the uploaded CSV has missing or wrong columns?**
The preprocessing module validates the required schema before prediction. Missing
required columns produce a clear error response; missing values within valid columns are
imputed during cleaning.

**7. Does the system upload the user's original file to the cloud?**
No. Only the generated prediction output CSV is uploaded after `/predict` completes.
The original upload is processed in memory for prediction.

---

## Timing summary

| Slide | Topic | Time |
|---|---|---|
| 1 | Title | 0:45–1:00 |
| 2 | Outline | 0:30 |
| 3 | Introduction and motivation | 1:00 |
| 4 | Literature survey and gap | 1:30 |
| 5 | Problem definition | 1:00 |
| 6 | Objectives | 0:45 |
| 7 | Methodology pipeline | 1:30 |
| 8 | Local vs Azure architecture | 1:30 |
| 9 | Work done so far | 1:00 |
| 10 | Current results | 1:30 |
| 11 | Key observations | 1:00 |
| 12 | Remaining work and scope | 1:15 |
| 13 | Conclusion | 1:00 |
| 14 | References | 0:30 |
| 15 | Thank you / Q&A | — |
| | **Total** | **≈ 14:45** |
