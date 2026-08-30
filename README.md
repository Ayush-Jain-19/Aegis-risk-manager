# 🛡️ AI Risk Manager (Razorpay Buildathon — Track 2)

> **Core Philosophy:** Most fraud detection models are built for a perfect world. This system is designed with production-style failure handling in mind. 

The AI Risk Manager is a highly available, cost-optimized fraud detection engine. It doesn't just predict fraud; it actively minimizes financial loss through a custom threshold optimizer, conditionally explains its decisions via SHAP, and features a bulletproof **3-tier fallback architecture** to ensure the merchant checkout flow never hangs—even if the ML pipeline goes down or a malformed payload is received.

---

## 🚀 1. Business Impact & Cost Optimization

We do not optimize for abstract academic metrics like raw Accuracy or F1-score using a naive 0.5 decision boundary. We optimize for **dollars**.

By applying a custom cost matrix—estimating a $5 friction cost for a False Positive (declining a good customer) vs. eating a 100% chargeback on a False Negative (missing actual fraud)—our optimizer dynamically anchored the decision threshold at **0.130**. 

*   **The Trade-off:** We intentionally sacrificed precision (28.9%) to achieve a staggering **96.88% Recall**. 
*   **The Bottom Line:** On the held-out test set, this model saved **$8,983.65 (a 19.1% reduction in loss)** compared to a standard baseline. We prioritize catching $5,000 chargebacks over avoiding $5 friction costs.

---

## 🏗️ 2. System Architecture & Data Flow

```mermaid
graph TD
    %% Define styles
    classDef ui fill:#0B0E14,stroke:#0262FF,stroke-width:2px,color:#fff;
    classDef api fill:#1A1F2B,stroke:#2D3748,stroke-width:2px,color:#E2E8F0;
    classDef ml fill:#0262FF,stroke:#fff,stroke-width:2px,color:#fff;
    classDef fallback fill:#d62728,stroke:#fff,stroke-width:2px,color:#fff;

    A[Merchant Checkout / Streamlit UI]:::ui -->|1. JSON Payload| B(FastAPI Endpoint):::api
    
    B --> C{Layer 1: Pydantic Validation}:::api
    C -->|Invalid| D[422 Graceful Error + REVIEW]:::fallback
    C -->|Valid| E[In-Memory Feature Store]:::api
    
    E -->|Compute Velocity| F{Layer 2: ML Inference Threadpool}:::ml
    
    F -->|Success| G[LightGBM Scoring]:::ml
    G -->|Probability| H{Cost-Optimized Threshold}:::ml
    H -->|Action: REVIEW / BLOCK| I[SHAP TreeExplainer]:::ml
    H -->|Action: APPROVE| J[Bypass SHAP to save CPU]:::api
    
    F -->|ML Pipeline Fails| K[Layer 3: Hardcoded Fallback Rules]:::fallback
    
    I --> L[PredictionOutput JSON]:::api
    J --> L
    K --> L
    D --> L
    
    L -->|200 OK| A

🛡️ 3. The 3-Tier Graceful Fallback
A risk engine going down should not take the merchant's checkout page down with it. We implemented a strict defense-in-depth strategy:

Tier 1 (Validation Layer): Bad payloads (e.g., missing fields, negative amounts) are caught by Pydantic. Instead of a hard 500 crash, it returns a 422 with a safe REVIEW decision and structured error logs.

Tier 2 (Asynchronous ML): LightGBM inference and SHAP generation are offloaded to a threadpool to prevent blocking the FastAPI async event loop. Furthermore, SHAP is only computed for REVIEW or BLOCK decisions, saving massive CPU cycles on the 99% of transactions that are safely approved.

Tier 3 (Rules Engine): If the ML model crashes, the API seamlessly routes the transaction to fallback_rules.py—a zero-dependency, hardcoded rules engine (e.g., automatically blocking any unscored transaction > $5,000).

🧠 4. Machine Learning & Feature Engineering
Leakage-Free Engineering
Time-Series Velocity Metrics: Rolling averages (amt_sum_24h, txn_count_1h) are engineered securely by keeping train and test sets in a chronological stream, ensuring no forward-looking data leakage while accurately reflecting live transaction history.

Haversine Distance: We calculate the physical curve of the Earth between the merchant and the customer to detect card-present anomalies.

Leakage Protection: We strictly use Frequency Encoding for high-cardinality categoricals (Merchant, Job, City) to prevent in-sample memorization (target leakage).

Chronological Split: Hyperparameters and thresholds were tuned on a strictly time-ordered validation tail, not a random shuffle.

🔬 The Ablation Test (Transparency)
This model was trained on the synthetic Sparkov dataset. To prove the model learned actual behavioral fraud (velocity & distance) rather than merely memorizing the synthetic data generator's clock patterns, we executed an ablation test removing all temporal identifiers (txn_hour, day of week, etc.).

Original PR-AUC: 0.9716

Ablated PR-AUC: 0.9202
The model retains massive predictive power without time features, proving its structural integrity.

📊 5. Final Evaluation Metrics (Held-Out Test Set)
Operating Threshold: 0.130

Precision: 28.94%

Recall (Fraud Capture Rate): 96.88%

PR-AUC: 0.9716 (Full Model)

ROC-AUC: 0.9982

💻 6. How to Run Locally
1. Clone the repository and install dependencies:
    git clone <your-repo-link>
    cd Fraud_Detection_System
    pip install -r requirements.txt
2. Start the FastAPI Backend:
    uvicorn src.api.main:app --reload
3. Start the Streamlit Risk Console:
    streamlit run src/ui/app.py