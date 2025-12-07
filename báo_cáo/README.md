# Báo Cáo Đồ Án - AI Service

## Cấu trúc

Folder này chứa các file LaTeX cho báo cáo đồ án:

- `chuong4.tex`: Chương 4 - Hệ gợi ý việc làm

## Cách sử dụng

### 1. Tích hợp vào báo cáo chính

Để sử dụng Chương 4 trong báo cáo của bạn, bạn cần:

1. Copy file `chuong4.tex` vào thư mục chứa các chapter của báo cáo

2. Trong file main.tex, thêm dòng:
```latex
\input{chuong4.tex}
```

3. Đảm bảo các package cần thiết đã được include:
```latex
\usepackage{tikz}
\usepackage{amsmath}
\usepackage{float}
```

### 2. Template main.tex mẫu

Xem file `main_mau.tex` để tham khảo cách tích hợp.

### 3. Customization

Bạn có thể điều chỉnh:
- Số section/subsection
- Các bảng và hình vẽ
- Nội dung chi tiết các phần
- Style và format theo yêu cầu của trường

## Nội dung Chương 4

Chương 4 bao gồm các phần:

1. **Tóm tắt**: Tổng quan về hệ thống gợi ý việc làm
2. **Giới thiệu vấn đề**: Bài toán và thách thức
3. **Tổng quan công nghệ**: Two-Tower Architecture, Sentence Transformers
4. **Chọn mô hình AI**: Lý do chọn PhoBERT-based SimCSE
5. **Benchmark và đánh giá**: Kết quả và so sánh
6. **Kiến trúc hệ thống**: Chi tiết các thành phần và luồng xử lý
7. **Kết luận chương**: Tóm tắt và hướng phát triển

## Lưu ý

- Tất cả thông tin trong chương này được lấy từ codebase thực tế
- Các số liệu benchmark là từ file `evaluation_results.json`
- Các mô hình được đề cập đều có trong code
- Kiến trúc hệ thống dựa trên code thực tế

## Yêu cầu packages

Các package LaTeX cần thiết:
- `tikz`: Để vẽ sơ đồ
- `amsmath`: Để viết công thức toán
- `float`: Để đặt vị trí hình vẽ và bảng
- `graphicx`: Nếu cần chèn hình ảnh
- `booktabs`: Để tạo bảng đẹp hơn (tùy chọn)

## Template sử dụng

File này được viết để tương thích với:
- Template nd-hung/thesis-template
- Template HCMUIT
- Các template LaTeX thesis khác (có thể cần điều chỉnh nhỏ)

## Hỗ trợ

Nếu có câu hỏi hoặc cần điều chỉnh, vui lòng:
1. Kiểm tra lại thông tin trong codebase
2. Tham khảo các file documentation trong folder `docs/`
3. Xem các file markdown báo cáo: `report_architecture.md`, `report_recommendation.md`


