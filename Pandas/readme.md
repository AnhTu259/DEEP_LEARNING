# CÔNG CỤ CHÍNH ĐƯỢC SỬ DỤNG
- Ngôn ngữ lập trình chính: Python
- Dùng để xử lý dữ liệu, thao tác mảng, phân tích thống kê
- Pandas
- Thư viện xử lý dữ liệu dạng bảng (DataFrame)
- Phù hợp cho dữ liệu CSV, Excel
- Cung cấp các hàm: đọc dữ liệu, làm sạch, thống kê, nhóm dữ liệu (groupby)
- NumPy
- Thư viện xử lý mảng số (array)
- Dùng để:
chuyển dữ liệu từ DataFrame sang dạng mảng
# Bài tập 1:
#In ra 2 dòng đầu tiên và 2 dòng cuối cùng của DataFrame.
head(2) lấy 2 dòng đầu tiên
tail(2) lấy 2 dòng cuối cùng
Giúp kiểm tra:
dữ liệu có đọc đúng không
cấu trúc bảng dữ liệu

#In ra kích thước (shape) của DataFrame.
trả về số cột và số dòng

#In ra tên các đặc trưng (các cột) của DataFrame.
Trả về danh sách tên cột

#In ra bảng thống kê mô tả bằng hàm .describe().
Tính các thống kê:
mean (trung bình)
std (độ lệch chuẩn)
min, max
quartile (25%, 50%, 75%)
Chỉ áp dụng cho cột sôS

#Vì cột Hepatitis B có rất nhiều giá trị thiếu (NaN) và có mức tương quan cao với Diphtheria, hãy xóa cột Hepatitis B.
Đồng thời xóa cột Population do có quá nhiều giá trị NaN.
Loại bỏ cột:
nhiều giá trị NaN
gây nhiễu mô hình

#Chuyển đổi cột Status sang dạng số:

    0 cho Developing

    1 cho Developed
Chuyển dữ liệu dạng chữ -> số

#Đổi tên cột thinness 1-19 years thành thinness 10-19 years.
rename()Thay đổi tên cột

#Lấy tất cả các cột ngoại trừ Life Expectancy, chuyển sang mảng NumPy và lưu vào biến X.
Loại bỏ cột mục tiêu (Life expectancy)
Chuyển DataFrame -> NumPy array
X dùng làm đầu vào mô hình

#Lấy cột Life Expectancy, chuyển sang mảng NumPy và lưu vào biến y.
Lấy riêng cột Life expectancy
Chuyển sang mảng NumPy
y là giá trị cần dự đoán

# Bài tập 2:
#Kiểm tra số lượng giá trị bị thiếu (NaN) của mỗi cột
isna() -> đánh dấu NaN = True
sum() -> đếm số True theo từng cột

#Thay thế toàn bộ NaN bằng giá trị trung bình (mean)
Pandas: fillna(), mean()
Tính mean cho từng cột số
Thay NaN bằng mean tương ứng

#Groupby theo Country
Pandas: groupby(), mean()
groupby("Country") -> gom dữ liệu theo quốc gia
mean() -> tính tuổi thọ trung bình
idxmin() / idxmax() -> lấy tên quốc gia

#Groupby theo Status (Developed / Developing)
Pandas: groupby()
Status = 0 -> Developing
Status = 1 -> Developed

#Tạo DataFrame mới thủ công
Pandas: DataFrame
NumPy: random.rand()
ID lấy giống cột Country
Noise_level là dữ liệu ngẫu nhiên
Mục đích: luyện merge DataFrame

#Gộp (merge) hai DataFrame dựa trên ID
Pandas: merge()
Ghép dữ liệu giống như JOIN trong SQL
left_on="Country" <-> right_on="ID"
Kết quả: DataFrame đầy đủ hơn