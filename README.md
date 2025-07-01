## Credit Scoring Business Understanding

### 1. Basel II and Model Interpretability

The Basel II Capital Accord emphasizes the importance of robust risk measurement and management for credit risk. It requires financial institutions to use internal models that are transparent, well-documented, and interpretable by both internal auditors and external regulators. This regulatory environment demands that credit risk models not only deliver predictive performance but also offer a clear rationale behind their predictions. 

For example, models like Logistic Regression with Weight of Evidence (WoE) are favored in many banking applications because they provide direct insight into how each feature affects the predicted probability of default. This interpretability builds trust and enables compliance with audit and regulatory requirements. In contrast, overly complex or black-box models may pose challenges in explaining decision logic, leading to regulatory pushback or operational risks.

---

### 2. Necessity and Risks of Proxy Variable

The dataset provided does not contain a direct label indicating whether a customer defaulted. To address this, we construct a proxy variable that estimates default risk based on customer behavioral patterns such as Recency, Frequency, and Monetary (RFM) values. This approach allows us to proceed with supervised learning techniques.

However, this introduces risks. Proxy labels may not perfectly align with actual default behavior, potentially leading to biased or unfair predictions. A disengaged customer might not necessarily default, and vice versa. If not carefully validated, the model could systematically disadvantage certain customer segments. Therefore, business understanding, domain expertise, and sensitivity analysis are essential when interpreting results derived from a proxy variable.

---

### 3. Model Complexity vs Interpretability Trade-offs

In regulated financial contexts, there's a critical trade-off between using simple, interpretable models and complex, high-performance ones. Simple models like Logistic Regression with WoE are easy to understand, audit, and justify. They allow financial institutions to maintain transparency and comply with regulatory standards.

On the other hand, advanced models such as Gradient Boosting Machines (GBM) often yield better predictive accuracy by capturing complex non-linear relationships. However, their black-box nature makes them harder to explain, which can be problematic in highly regulated environments. While tools like SHAP can enhance interpretability, they add complexity to the deployment and audit processes.

In practice, organizations often use a hybrid approach—leveraging complex models internally for decision support while using simpler models for regulatory reporting and risk governance.

