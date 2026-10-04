# HOUSE PRICES - TV2 → TV3 HANDOFF

## 1. Mục đích

File README này là tài liệu bàn giao công việc từ TV2 cho TV3.

TV3 cần đọc README này trước khi bắt đầu làm phần của mình.

TV2 đã hoàn thành phần Machine Learning bằng Scikit-learn/XGBoost. TV3 sẽ tiếp tục với mô hình MLP bằng PyTorch và so sánh kết quả với TV2.

---

# 2. Phần việc TV2 đã hoàn thành

TV2 phụ trách:

- Xây dựng các mô hình Machine Learning.
- Thử nghiệm nhiều feature sets.
- So sánh các mô hình.
- Thử nghiệm XGBoost.
- Tuning XGBoost.
- Chọn mô hình tốt nhất.
- Train mô hình cuối cùng trên toàn bộ training data.
- Tạo file submission.
- Submit kết quả lên Kaggle.

Các mô hình TV2 đã thử:

- Linear Regression
- Decision Tree
- Random Forest
- XGBoost
- XGBoost Tuned

---

# 3. Dữ liệu TV3 cần sử dụng

TV3 không cần preprocessing lại dữ liệu gốc.

TV3 sử dụng dữ liệu đã được TV1 preprocessing và feature selection.

Dữ liệu nằm tại:

```text
HousePrices_TV1/results/member1/processed/
```

Các file dữ liệu:

```text
X_train_full.csv
X_val_full.csv
X_test_full.csv

X_train_top_20.csv
X_val_top_20.csv
X_test_top_20.csv

X_train_top_40.csv
X_val_top_40.csv
X_test_top_40.csv

X_train_top_60.csv
X_val_top_60.csv
X_test_top_60.csv

y_train.csv
y_val.csv
test_ids.csv
```

---

# 4. Kích thước dữ liệu

Số lượng dòng:

| Dataset | Số dòng |
|---|---:|
| Train | 1168 |
| Validation | 292 |
| Test | 1459 |

Số lượng features:

| Feature Set | Số features |
|---|---:|
| Full | 279 |
| Top 20 | 20 |
| Top 40 | 40 |
| Top 60 | 60 |

---

# 5. Quy định về dữ liệu

TV3 phải giữ nguyên train/validation split của TV1.

Không được tự chia lại train/validation.

Sử dụng:

```text
X_train_* → Training
X_val_*   → Validation
X_test_*  → Test
```

Target:

```text
y_train.csv
y_val.csv
```

Random state:

```python
RANDOM_STATE = 42
```

Không được fit preprocessing hoặc feature selection lại trên validation/test.

---

# 6. Phần việc của TV3

TV3 phụ trách xây dựng mô hình:

**MLP - Multi-Layer Perceptron**

Framework sử dụng:

**PyTorch**

Mục tiêu là xây dựng Neural Network để dự đoán:

```text
SalePrice
```

Sau đó đánh giá và so sánh kết quả MLP với các mô hình mà TV2 đã thực hiện.

---

# 7. Các feature sets TV3 phải thử

TV3 cần thử tối thiểu 4 feature sets:

```text
Full
Top 20
Top 40
Top 60
```

Mục đích:

- Kiểm tra ảnh hưởng của số lượng features.
- Xác định feature set phù hợp nhất với MLP.
- So sánh với kết quả của TV2.

---

# 8. Metric đánh giá

Metric sử dụng:

**Log RMSE**

Càng thấp càng tốt.

Công thức:

```python
rmse_log = np.sqrt(
    mean_squared_error(
        np.log1p(y_val),
        np.log1p(predictions)
    )
)
```

Prediction không được âm:

```python
predictions = np.maximum(predictions, 0)
```

---

# 9. Target

Target là:

```text
SalePrice
```

Có thể train MLP trên:

```python
np.log1p(SalePrice)
```

Sau khi model prediction:

```python
predictions = np.expm1(predictions)
```

Sau đó tính Log RMSE trên validation set.

---

# 10. Mô hình MLP

TV3 bắt đầu bằng một MLP cơ bản.

Kiến trúc ban đầu có thể là:

```text
Input
  ↓
Linear
  ↓
ReLU
  ↓
Dropout
  ↓
Linear
  ↓
ReLU
  ↓
Dropout
  ↓
Linear
  ↓
Output
```

TV3 có thể thay đổi kiến trúc dựa trên kết quả thực nghiệm.

---

# 11. Các hyperparameters cần thử

TV3 nên thử một số cấu hình khác nhau về:

- Số hidden layers
- Số neurons mỗi layer
- Learning rate
- Batch size
- Dropout
- Số epochs
- Optimizer

Không cần thử quá nhiều cấu hình.

Mục tiêu là tìm được một cấu hình MLP hợp lý và có kết quả tốt.

---

# 12. Kết quả TV2 - Model cơ bản

TV2 đã thử nghiệm các model:

| Model | Feature Set | Log RMSE |
|---|---|---:|
| Random Forest | Full | 0.151819 |
| Random Forest | Top 60 | 0.152101 |
| Random Forest | Top 40 | 0.157294 |
| Linear Regression | Top 60 | 0.158576 |
| Linear Regression | Full | 0.160900 |
| Random Forest | Top 20 | 0.164018 |
| Linear Regression | Top 20 | 0.174479 |
| Linear Regression | Top 40 | 0.174945 |
| Decision Tree | Top 60 | 0.208696 |
| Decision Tree | Top 40 | 0.211511 |
| Decision Tree | Full | 0.213573 |
| Decision Tree | Top 20 | 0.225467 |

Model cơ bản tốt nhất:

```text
Random Forest + Full
Log RMSE = 0.151819
```

---

# 13. XGBoost Baseline của TV2

XGBoost baseline có kết quả:

| Feature Set | Log RMSE |
|---|---:|
| Full | 0.137608 |
| Top 60 | 0.142951 |
| Top 40 | 0.150651 |
| Top 20 | 0.162683 |

Kết quả tốt nhất:

```text
XGBoost + Full
Log RMSE = 0.137608
```

---

# 14. XGBoost Tuned - Model tốt nhất của TV2

TV2 đã tuning XGBoost bằng RandomizedSearchCV.

Best parameters:

```python
{
    "subsample": 0.9,
    "n_estimators": 700,
    "min_child_weight": 5,
    "max_depth": 3,
    "learning_rate": 0.08,
    "gamma": 0,
    "colsample_bytree": 1.0
}
```

Model:

```python
XGBRegressor(
    subsample=0.9,
    n_estimators=700,
    min_child_weight=5,
    max_depth=3,
    learning_rate=0.08,
    gamma=0,
    colsample_bytree=1.0,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)
```

Kết quả:

```text
Validation Log RMSE = 0.135426
```

Đây là **baseline chính để TV3 so sánh**.

---

# 15. So sánh XGBoost trước và sau tuning

| Model | Feature Set | Validation Log RMSE |
|---|---|---:|
| XGBoost Baseline | Full | 0.137608 |
| XGBoost Tuned | Full | 0.135426 |

Tuning giúp cải thiện Log RMSE từ:

```text
0.137608
```

xuống:

```text
0.135426
```

---

# 16. Kaggle Result của TV2

Sau khi chọn được XGBoost Tuned, TV2 train lại model trên toàn bộ:

```text
Train + Validation
= 1168 + 292
= 1460 rows
```

Feature set:

```text
Full
```

Số features:

```text
279
```

Test:

```text
1459 rows
```

File submission:

```text
submission.csv
```

Kết quả Kaggle:

```text
0.13357
```

---

# 17. Kết quả quan trọng nhất của TV2

```text
Best Model:
XGBoost Tuned

Feature Set:
Full

Number of Features:
279

Validation Log RMSE:
0.135426

Kaggle Score:
0.13357
```

TV3 sử dụng:

```text
Validation Log RMSE = 0.135426
```

làm baseline chính để so sánh MLP.

---

# 18. Mục tiêu của TV3

TV3 cần thực hiện các bước:

```text
1. Load dữ liệu từ TV1
        ↓
2. Kiểm tra dữ liệu
        ↓
3. Xây dựng MLP bằng PyTorch
        ↓
4. Train MLP
        ↓
5. Evaluate trên validation
        ↓
6. Thử Full / Top20 / Top40 / Top60
        ↓
7. Thử một số hyperparameters
        ↓
8. Chọn MLP tốt nhất
        ↓
9. So sánh với TV2
        ↓
10. Bàn giao kết quả cho nhóm
```

---

# 19. Kết quả TV3 cần có

TV3 cần tạo bảng kết quả:

| Model | Feature Set | Log RMSE |
|---|---|---:|
| MLP | Full | ... |
| MLP | Top 20 | ... |
| MLP | Top 40 | ... |
| MLP | Top 60 | ... |

Sau đó ghi rõ model tốt nhất:

```text
Best MLP:
Feature Set:
Architecture:
Learning Rate:
Batch Size:
Dropout:
Epochs:
Optimizer:
Validation Log RMSE:
```

---

# 20. So sánh với TV2

TV3 cần tạo bảng tổng hợp:

| Model | Feature Set | Log RMSE |
|---|---|---:|
| Random Forest | Full | 0.151819 |
| XGBoost Baseline | Full | 0.137608 |
| XGBoost Tuned | Full | 0.135426 |
| MLP | Best Feature Set | ... |

Sau đó kết luận:

- MLP có tốt hơn XGBoost Tuned hay không?
- Feature set nào tốt nhất cho MLP?
- Model nào có Log RMSE thấp nhất?

---

# 21. Deliverables của TV3

TV3 cần bàn giao:

### 1. Notebook

```text
notebook_MLP.ipynb
```

Notebook cần có:

- Load dữ liệu.
- Kiểm tra dữ liệu.
- Xây dựng MLP.
- Train model.
- Validation.
- Thử nghiệm Full / Top20 / Top40 / Top60.
- Hyperparameter tuning.
- So sánh kết quả.
- Chọn model tốt nhất.

### 2. Kết quả

Cần ghi rõ:

```text
Best MLP
Best Feature Set
Best Architecture
Best Hyperparameters
Best Validation Log RMSE
```

### 3. Bảng so sánh

So sánh MLP với kết quả của TV2.

### 4. Kết luận

Nêu rõ:

```text
MLP có vượt XGBoost Tuned hay không?
Feature set nào tốt nhất?
Model nào tốt nhất?
```

---

# 22. Những phần TV3 KHÔNG cần làm lại

TV3 không cần làm lại:

- Preprocessing
- Feature Engineering
- Feature Selection
- Linear Regression
- Decision Tree
- Random Forest
- XGBoost Baseline
- XGBoost Tuning
- Kaggle submission của TV2

TV3 tập trung vào:

```text
PyTorch
    ↓
MLP
    ↓
Experiment
    ↓
Validation
    ↓
Comparison
```

---

# 23. Quy tắc quan trọng

1. Không chia lại train/validation.
2. Không thay đổi preprocessing của TV1.
3. Sử dụng đúng các feature sets được bàn giao.
4. Giữ `random_state = 42` khi có thể.
5. Dùng Log RMSE để đánh giá.
6. Ghi lại đầy đủ hyperparameters của model tốt nhất.
7. Có bảng kết quả để nhóm dễ tổng hợp.
8. So sánh trực tiếp với XGBoost Tuned của TV2.

---

# 24. Baseline cuối cùng cần nhớ

```text
TV2 Best Model:
XGBoost Tuned

Feature Set:
Full

Features:
279

Validation Log RMSE:
0.135426

Kaggle Score:
0.13357
```

## Mục tiêu của TV3

Không bắt buộc MLP phải vượt XGBoost.

Điều quan trọng là TV3 phải:

- Xây dựng được MLP hoàn chỉnh.
- Thực hiện đúng thực nghiệm.
- Có kết quả rõ ràng.
- Chọn được cấu hình MLP tốt nhất.
- So sánh được với TV2.
- Bàn giao kết quả cho phần tổng hợp cuối cùng của nhóm.