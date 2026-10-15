# Module Phân tích Python: Kiểm định Giả thuyết Thống kê (RQ1)

Thư mục này chứa mã nguồn Python giải quyết **Câu hỏi Nghiên cứu 1 (RQ1)**:
> *"Những yếu tố nhân khẩu học, lối sống và liên quan đến sức khỏe nào có mối liên quan có ý nghĩa thống kê với bệnh tiểu đường trong tập dữ liệu CDC BRFSS 2015?"*

Giai đoạn này tuân theo quy trình CRISP-DM, kết hợp suy luận thống kê và phân tích khám phá trước khi thực hiện mô hình hóa học máy.

---

## 1. Cơ sở Toán học và Thống kê

Để hiểu mối quan hệ giữa các chỉ số sức khỏe và nguy cơ mắc bệnh tiểu đường, chúng tôi sử dụng một số kiểm định giả thuyết thống kê tùy thuộc vào kiểu dữ liệu của biến.

### 1.1 Kiểm định Chi-Square về tính độc lập (Biến phân loại & biến thứ tự)
Đối với các biến nhị phân và biến thứ tự, chúng tôi đánh giá mối liên quan của chúng với `Diabetes_binary` bằng bảng chéo (contingency table). Gọi $O_{ij}$ là tần số quan sát được ở hàng $i$ và cột $j$, và $E_{ij}$ là tần số kỳ vọng dưới giả thuyết không ($H_0$) về tính độc lập:

$$E_{ij} = \frac{\sum_{k} O_{ik} \cdot \sum_{k} O_{kj}}{N}$$

Thống kê Chi-Square được tính như sau:

$$\chi^2 = \sum_{i} \sum_{j} \frac{(O_{ij} - E_{ij})^2}{E_{ij}}$$

*   **Giả thuyết Không ($H_0$)**: Chỉ số sức khỏe (ví dụ: `HighBP`) và tình trạng tiểu đường độc lập với nhau (không liên quan).
*   **Giả thuyết Đối ($H_1$)**: Chỉ số sức khỏe và tình trạng tiểu đường có mối liên quan với nhau (không độc lập).
*(Tệp kết quả tạo ra ở đây: chi_square_results.csv)*

### 1.2 Hệ số Cramér's V (Kích thước ảnh hưởng đối với biến phân loại)
Vì kích thước mẫu lớn ($N = 253,680$), các kiểm định thống kê có sức mạnh kiểm định tiệm cận vô hạn khiến p-value của gần như mọi biến đều tiến về 0 ($p < 0.05$). Do đó, việc xếp hạng quan trọng không dựa vào p-value mà dựa vào **Cramér's V** để đo lường mức độ liên quan thực tế ở cấp độ quần thể:

$$V = \sqrt{\frac{\chi^2}{N \cdot (k - 1)}}$$

Trong đó $N$ là kích thước mẫu, và $k = \min(r, c) = 2$. Do đó $k - 1 = 1$:

$$V = \sqrt{\frac{\chi^2}{N}}$$

**Thang đo diễn giải Cramér's V:**
*   $V < 0.05$: Không đáng kể (Negligible)
*   $0.05 \le V < 0.10$: Yếu (Weak)
*   $0.10 \le V < 0.20$: Nhỏ (Small)
*   $0.20 \le V < 0.30$: Trung bình (Moderate)
*   $V \ge 0.30$: Mạnh (Strong)

---

### 1.3 Kiểm định t-Test độc lập hai mẫu (Welch's t-Test) & Cohen's d
Đối với các biến liên tục/số (`BMI`, `MentHlth`, `PhysHlth`), chúng tôi so sánh giá trị trung bình của nhóm không tiểu đường ($X_1$) và nhóm tiểu đường ($X_2$) bằng kiểm định Welch's t-test và tính toán Cohen's d:

$$t = \frac{\bar{X}_1 - \bar{X}_2}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}}$$

---

### 1.4 Kiểm định Mann-Whitney U & Hệ số tương quan Rank-Biserial
Bởi vì các biến số có độ lệch và phân phối không chuẩn, kiểm định phi tham số **Mann-Whitney U** và hệ số **Absolute Rank-Biserial Correlation ($|r_{rb}|$)** được chọn làm thước đo hiệu ứng chính cho biến số:

$$\text{CLES} = \frac{U_1}{n_1 \cdot n_2}, \quad r_{rb} = 2 \cdot \text{CLES} - 1$$

**Thang đo diễn giải Absolute Rank-Biserial:**
*   $|r_{rb}| < 0.10$: Không đáng kể (Negligible)
*   $0.10 \le |r_{rb}| < 0.30$: Nhỏ (Small)
*   $0.30 \le |r_{rb}| < 0.50$: Trung bình (Moderate)
*   $|r_{rb}| \ge 0.50$: Lớn (Large)

---

### 1.5 Hiệu chỉnh Holm–Bonferroni cho kiểm định đa giả thuyết
Để kiểm soát sai sót họ (family-wise error rate) trên toàn bộ 21 chỉ số sức khỏe, chúng tôi áp dụng quy trình hiệu chỉnh p-value Holm–Bonferroni:

$$p_{(i)}^{\text{Holm}} = \min\left(1, \max_{j \le i} \left( (m - j + 1) \cdot p_{(j)} \right)\right)$$

---

## 2. Cấu trúc Kết quả Đầu ra

Khi chạy `python_analysis/statistical_analysis.py`, hệ thống tạo ra:
* `results/statistical_analysis/chi_square_results.csv`: Kết quả Chi-Square, p-value thô, p-value hiệu chỉnh Holm, và Cramér's V.
* `results/statistical_analysis/numerical_results.csv`: Kết quả Welch t-test, Mann-Whitney U, p-value Holm, Cohen's d và Rank-Biserial.
* `results/statistical_analysis/cramers_v_ranking.png` & `docs/figures/effect_size_ranking.png`: Biểu đồ dạng lollipop xếp hạng Kích thước Ảnh hưởng (Effect Size) toàn bộ 21 biến số có phân vùng ngưỡng hiệu ứng.
* `docs/statistical_analysis.md`: Báo cáo học thuật biên soạn tự động.
