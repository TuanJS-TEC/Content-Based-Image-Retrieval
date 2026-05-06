# Xây dựng hệ thống tìm kiếm ảnh chim tương đồng

## 1. Xây dựng bộ thuộc tính nhận diện ảnh chim

Dựa trên đặc thù bộ dữ liệu (chim đang đậu, góc ngang, cùng tỷ lệ/kích thước khung hình), hình dáng tổng thể của các bức ảnh khá giống nhau. Vì vậy, cần trích xuất các **đặc trưng bậc thấp (low-level features)** để phục vụ việc phân biệt và tìm kiếm tương đồng.

### 1.1. Đặc trưng màu sắc (Color Features)

- **Thuộc tính sử dụng**:  
  - Biểu đồ tần suất màu (*Color Histogram*)
  - Không gian màu: RGB → HSI / CIE Luv / CIE Lab

- **Lý do & giá trị**:
  - Màu sắc là yếu tố quan trọng nhất để phân biệt các loài chim.
  - Biểu đồ màu toàn ảnh dễ bị nhiễu bởi nền (cành cây, trời, lá).
  - Giải pháp:
    - Phân đoạn ảnh (tách chim và nền)
    - Hoặc dùng **PWH (Perceptual Weighted Histogram)**: tăng trọng số vùng trung tâm (vị trí con chim)

---

### 1.2. Đặc trưng hình dạng (Shape Features)

- **Thuộc tính sử dụng**:
  - Tâm sai (Eccentricity)
  - Moment bất biến (Invariant Moments)
  - Biểu diễn dạng lưới (Grid-based binary encoding)

- **Lý do & giá trị**:
  - Các loài chim khác nhau về:
    - Độ dài đuôi
    - Kích thước mỏ
    - Độ cong thân
  - Trục chính/phụ → đo tỷ lệ hình dạng
  - Moment bất biến → ổn định với biến đổi hình học nhỏ

---

### 1.3. Đặc trưng kết cấu (Texture Features)

- **Thuộc tính sử dụng**:
  - Độ thô/mịn (*Coarseness*)
  - Độ tương phản (*Contrast*)

- **Lý do & giá trị**:
  - Lông chim có nhiều kiểu:
    - Trơn
    - Có đốm
    - Có vằn
  - Giúp phân biệt các loài có màu sắc tương tự nhưng khác cấu trúc lông

---

## 2. Cách thức lưu trữ bộ đặc trưng

### 2.1. Tách biệt dữ liệu

- Ảnh gốc:
  - Lưu dưới dạng **BLOB** hoặc file riêng
- Vector đặc trưng:
  - Lưu trong bảng riêng (metadata)

---

### 2.2. Cấu trúc dữ liệu đa chiều

Để tối ưu tìm kiếm (tránh quét toàn bộ 500 ảnh):

#### a. Cây R (R-tree, R+ tree)

- Sử dụng **MBR (Minimum Bounding Rectangle)**
- Nhóm các vector gần nhau thành cụm
- Nút lá chứa con trỏ đến ảnh

#### b. Cây k-d (k-d tree)

- Cây nhị phân
- Phân chia theo từng chiều đặc trưng
- Giúp giảm mạnh không gian tìm kiếm

---

## 3. Cơ chế tìm kiếm 5 ảnh chim tương đồng nhất

### Bước 1: Trích xuất đặc trưng (Query Formulation)

- Ảnh đầu vào → trích xuất:
  - Màu sắc
  - Hình dạng
  - Kết cấu
- Tạo thành **vector truy vấn**

---

### Bước 2: Đo khoảng cách (Similarity Computation)

- Không tìm exact match → tìm **xấp xỉ**
- Các phép đo phổ biến:
  - **L1-norm (Manhattan distance)**
  - **L2-norm (Euclidean distance)**

- Nguyên tắc:
  - Khoảng cách càng nhỏ → ảnh càng giống

---

### Bước 3: Tìm kiếm k-NN

- Không so sánh toàn bộ dữ liệu
- Duyệt cây chỉ mục (k-d tree / R-tree)
- Áp dụng:
  - **Triangle Inequality**
- Thuật toán:
  - **k-Nearest Neighbor (k-NN)**
  - Với bài toán: **k = 5**

---

### Bước 4: Trả kết quả & xếp hạng

- Lấy ra 5 ảnh gần nhất
- Sắp xếp:
  - Tăng dần theo khoảng cách
- Ảnh giống nhất → đứng đầu

---

## Tổng kết

Hệ thống sử dụng:

- **Đặc trưng**: màu sắc + hình dạng + kết cấu  
- **Lưu trữ**: vector đa chiều + cây chỉ mục  
- **Tìm kiếm**: k-NN + khoảng cách toán học  

→ Đảm bảo tìm nhanh và chính xác 5 ảnh chim tương đồng nhất.
