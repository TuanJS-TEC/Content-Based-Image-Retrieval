# BÀI TOÁN: XÂY DỰNG HỆ CSDL TÌM KIẾM ẢNH CHIM

## Phần 1: Yêu cầu

Xây dựng hệ CSDL lưu trữ và tìm kiếm ảnh chim.

### 1. Bộ dữ liệu ảnh
- Thu thập ít nhất **500 ảnh chim khác nhau**
- Điều kiện:
  - Cùng kích thước
  - Cùng tỷ lệ khung hình
  - Chim đang đậu (không bay)
  - Góc chụp ngang
- Định dạng ảnh: tùy chọn

---

### 2. Xây dựng bộ thuộc tính

- Xây dựng các đặc trưng:
  - Thể hiện **sự tương đồng**
  - Thể hiện **sự khác biệt**
- Yêu cầu:
  - Trình bày lý do lựa chọn
  - Phân tích giá trị thông tin

---

### 3. Xây dựng hệ CSDL

- Lưu trữ:
  - Siêu dữ liệu (metadata)
- Trình bày:
  - Cơ chế tìm kiếm ảnh tương đồng

---

### 4. Xây dựng hệ thống tìm kiếm

- **Input**: 1 ảnh chim mới  
- **Output**: 5 ảnh giống nhất (giảm dần độ tương đồng)

#### a. Sơ đồ khối
- Trình bày pipeline hệ thống

#### b. Kết quả trung gian
- Hiển thị các bước xử lý

---

### 5. Demo & đánh giá

- Demo hệ thống
- Đánh giá:
  - Độ chính xác
  - Tốc độ

---

## Phần 2: Các điểm cần chú ý

- Không phải bài toán:
  - Gán nhãn
  - AI / Machine Learning

→ Mục tiêu: **Xây dựng bộ đặc trưng phong phú**

---

# Gợi ý triển khai

## 1. Xây dựng đặc trưng ảnh

### 1.1. Đặc trưng màu sắc

- **Color Histogram**
- **PWH (Histogram có trọng số)**
- Chia vùng ảnh (spatial partitioning)
- Đổi không gian màu:
  - RGB → CIE Luv / CIE Lab / HSI

---

### 1.2. Đặc trưng hình dạng

- Trục chính / trục phụ
- Bounding box
- Độ lệch tâm

#### Grid-based:
- Lưới nhị phân
- Ngưỡng: 15%

#### Nâng cao:
- Moment bất biến
- Fourier descriptors

---

### 1.3. Đặc trưng kết cấu

- Độ thô/mịn
- Độ tương phản
- Đường viền
- Bố cục

---

### 1.4. Miền tần số

- DFT / FFT
- DCT
- Wavelet

→ Sinh ra hệ số đặc trưng độc lập

---

### 1.5. Phân đoạn ảnh

- Chia ảnh thành vùng đồng nhất
- Dựa trên:
  - Mức xám
  - Màu sắc

→ Dùng để trích xuất đặc trưng chính xác hơn

---

## 2. Phương pháp tìm kiếm ảnh

### 2.1. Attribute-based

- Dựa trên metadata
- Nhược điểm: không phản ánh nội dung ảnh

---

### 2.2. Text-based

- Gán từ khóa
- Nhược điểm:
  - Chủ quan
  - Tốn công

---

### 2.3. CBIR (Content-Based Image Retrieval)

### Quy trình:

#### Bước 1: Query Formulation
- Query by Example (QBE)
- Trích xuất vector đặc trưng

---

#### Bước 2: Similarity Computation

- L1-norm
- L2-norm (Euclidean)

→ Khoảng cách nhỏ = giống nhau

---

#### Bước 3: Retrieval & Ranking

- Trả về danh sách ảnh
- Sắp xếp theo độ tương đồng

---

#### Bước 4: Relevance Feedback

- Người dùng chọn ảnh phù hợp
- Hệ thống cập nhật lại truy vấn

---

## Tối ưu hóa

- Sử dụng:
  - R-tree
  - Chỉ mục đa chiều
- Áp dụng:
  - Triangle Inequality

→ Giảm không gian tìm kiếm, tăng tốc độ
