# Module Phân tích Python: Trực quan hóa SHAP & Phân tích Căn chỉnh Bằng chứng (RQ3)

Tệp này giải thích chi tiết các bước thực hiện và kết quả phân tích học máy giải thích được (Explainable AI - XAI) cho **Câu hỏi Nghiên cứu 3 (RQ3)**:
> *"Các giải thích được tạo ra bởi mô hình học máy tốt nhất (XGBoost) có nhất quán về mặt thống kê với bằng chứng thu được từ các kiểm định giả thuyết truyền thống hay không?"*

Quá trình phân tích XAI được triển khai trong kịch bản [shap_analysis.py].

---

## 1. Phương pháp luận & Tải mô hình Động

1. **Tải Mô hình Khóa:** Đọc trực tiếp `results/modeling/final_model.joblib` và `results/modeling/model_selection.json` từ Phase 4.
2. **Mẫu Kiểm thử Độc lập:** Sử dụng mẫu 10,000 bản ghi từ tập kiểm thử độc lập (Holdout Test Set) để tính toán post-hoc SHAP values với `shap.TreeExplainer`.
3. **Thuật ngữ Chuẩn hóa:** Gọi từng đối tượng được khảo sát là `survey respondent` hoặc `example profile` (không dùng thuật ngữ bệnh nhân lâm sàng).

---

## 2. Phân nhóm Căn chỉnh Bằng chứng (Statistical–SHAP Evidence Alignment)

Không dựa vào các giá trị p-value ngây thơ ($p < 0.05$), chúng tôi phân chia 21 đặc trưng thành 4 nhóm theo ngưỡng Kích thước Ảnh hưởng (Effect Size: Cramér's V $\ge 0.05$ hoặc $|r_{rb}| \ge 0.10$) và độ quan trọng Top-10 SHAP:

### Group 1 — Consistent high evidence (Đồng thuận cao - 9 đặc trưng)
*   **Danh sách:** `['GenHlth', 'HighBP', 'Age', 'BMI', 'HighChol', 'Income', 'CholCheck', 'HeartDiseaseorAttack', 'HvyAlcoholConsump']`
*   **Diễn giải:** Các đặc trưng này thể hiện liên hệ biên có ý nghĩa trong mẫu BRFSS được phân tích (within the analyzed BRFSS sample) và đồng thời đóng góp đáng kể trong mô hình đa biến.

### Group 2 — Meaningful marginal association, lower model salience (Liên hệ biên ý nghĩa, độ nổi bật mô hình thấp hơn - 7 đặc trưng)
*   **Danh sách:** `['DiffWalk', 'Education', 'Stroke', 'PhysHlth', 'PhysActivity', 'Smoker', 'Veggies']`
*   **Diễn giải:** Mức độ nổi bật SHAP thấp hơn của một số biến có thể phản ánh thông tin dùng chung, tương quan giữa các biến hoặc đóng góp gia tăng hạn chế sau khi các biến khác đã được xét đến (The lower SHAP salience of some features may reflect shared information, correlation, or limited incremental contribution after other predictors are considered).

### Group 3 — Model-salient, weak marginal association (Đóng góp mô hình cao, liên hệ biên yếu - 1 đặc trưng)
*   **Danh sách:** `['Sex']`
*   **Diễn giải:** Biến thể hiện đóng góp có điều kiện trong mô hình đa biến mặc dù liên hệ biên yếu. A feature with high SHAP salience but weak marginal association may contribute conditionally within the multivariable model; this result does not by itself prove a statistical interaction.

### Group 4 — Weak evidence (Bằng chứng hạn chế - 4 đặc trưng)
*   **Danh sách:** `['MentHlth', 'Fruits', 'AnyHealthcare', 'NoDocbcCost']`
*   **Diễn giải:** Cung cấp ít thông tin dự báo cả theo phân tích biên lẫn đóng góp trong mô hình đa biến.

---

## 3. Exploratory Rank-Alignment Diagnostics

*   **Primary Multivariable Alignment (SHAP vs Nested LR $\chi^2$ Deviance):**
    *   **Top-10 Overlap Count:** 10 / 10 đặc trưng (100%).
    *   **Top-10 Jaccard Similarity:** **1.0000**
    *   **Spearman Rank Correlation:** **$\rho_s = 0.9338$** ($p = 6.38 \times 10^{-10}$)
*   **Degree-of-Freedom-Aware Sensitivity Alignment:**
    *   **$\text{LR } \chi^2 - \text{df}$:** Top-10 Overlap = 10/10, Jaccard = 1.0000, $\rho_s = 0.9338$ ($p = 6.38 \times 10^{-10}$).
    *   **$\text{LR } \chi^2 / \text{df}$:** Top-10 Overlap = 9/10, Jaccard = 0.8182, $\rho_s = 0.8818$ ($p = 1.27 \times 10^{-7}$).

---

## 4. Tệp tin và Biểu đồ Kết quả

*   `results/xai/explanation_consistency.csv`: Dữ liệu phân nhóm và thứ hạng căn chỉnh.
*   `results/xai/shap_summary_bar.png` & `docs/figures/shap_global_importance.png`: Biểu đồ thanh ngang tầm quan trọng SHAP.
*   `results/xai/shap_summary_dot.png` & `docs/figures/shap_summary_beeswarm.png`: Biểu đồ beeswarm thể hiện hướng tác động.
*   `results/xai/consistency_quadrant.png` & `docs/figures/effect_size_shap_alignment.png`: Biểu đồ phân tán 4 nhóm căn chỉnh bằng chứng.
*   `results/xai/shap_local_class1.png` & `docs/figures/shap_local_high_risk.png`: Biểu đồ thác nước giải thích hồ sơ cá nhân có nguy cơ cao.
