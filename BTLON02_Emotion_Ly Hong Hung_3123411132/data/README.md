# Dữ liệu Kaggle FER2013

Tệp dữ liệu gốc `fer2013.csv` (~301 MB) vượt quá giới hạn dung lượng 100 MB/tệp của GitHub nên được loại trừ trong `.gitignore`.

## Nguồn tải dữ liệu
- **Kaggle (khuyên dùng)**: `deadskull7/fer2013` hoặc `msambare/fer2013` (hoặc trang cuộc thi gốc: `https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data`)
- **Cấu trúc chuẩn (35.887 ảnh xám `48×48`, 7 lớp cảm xúc)**:
  - `Training`: 28.709 ảnh
  - `PublicTest` (Validation): 3.589 ảnh
  - `PrivateTest` (Test độc lập): 3.589 ảnh

## Cách sử dụng
- **Chạy trên máy cá nhân (Local)**: Giải nén và đặt tệp `fer2013.csv` (hoặc thư mục `Training/`, `PublicTest/`, `PrivateTest/`) trực tiếp vào thư mục `data/` này — `src/dataset.py` sẽ tự động nhận diện.
- **Chạy trên Kaggle Notebook**: Bấm **Add Input → Datasets → `deadskull7/fer2013`**, không cần chép dữ liệu vào thư mục này.
