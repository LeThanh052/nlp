BÀI TẬP THỰC HÀNH
Chủ đề: One Pipeline, Many Classifiers cho bài toán phân loại văn bản tiếng Việt
1. Tên bài tập
Phân loại bài báo tiếng Việt thuộc chủ đề Kinh tế/Kinh doanh hay không
2. Bối cảnh
Xây dựng một hệ thống phân loại bài báo tiếng Việt thành hai nhóm:
 Nhóm 1: Bài báo thuộc chủ đề Kinh tế/Kinh doanh
 Nhóm 0: Bài báo không thuộc chủ đề Kinh tế/Kinh doanh
3. Mục tiêu học tập
Sau khi hoàn thành bài này, sinh viên cần đạt được các mục tiêu sau:
1. Hiểu và triển khai được pipeline cơ bản cho bài toán phân loại văn bản.
2. Biết cách chuẩn bị dữ liệu, tiền xử lý và biểu diễn văn bản bằng BoW và TF-IDF.
3. Huấn luyện và so sánh hiệu quả của các mô hình:
o Multinomial Naive Bayes
o Logistic Regression
o SVM
4. Biết đánh giá mô hình bằng nhiều chỉ số thay vì chỉ nhìn vào accuracy.
5. Nhận ra các vấn đề thường gặp trong thực tế như:
o dữ liệu mất cân bằng,
o không có mô hình nào luôn tốt nhất,
o feature quá nhiều có thể gây nhiễu,
o cải tiến mô hình cần dựa trên kết quả đánh giá.
6. Rèn kỹ năng phân tích kết quả và thực nghiệm.
4. Mô tả bài toán
Sinh viên xây dựng một hệ thống nhận đầu vào là một bài báo tiếng Việt và dự đoán xem bài báo
đó có thuộc nhóm Kinh tế/Kinh doanh hay không.
Bài toán được xem như một bài toán phân loại nhị phân:
 1: Kinh tế/Kinh doanh
 0: Không phải Kinh tế/Kinh doanh
5. Dữ liệu
5.1. Nguồn dữ liệu
Sinh viên sử dụng dữ liệu do chính nhóm tự crawl từ một hoặc nhiều trang báo điện tử tiếng Việt.
5.2. Yêu cầu dữ liệu tối thiểu
Mỗi mẫu dữ liệu nên có ít nhất các trường sau:
 tiêu đề bài báo
 mô tả ngắn/tóm tắt nếu có
 nội dung bài báo
 chuyên mục
 đường dẫn bài viết
5.3. Gán nhãn
Sinh viên gán nhãn dựa trên chuyên mục của bài báo:
 Các bài thuộc các mục như Kinh tế, Kinh doanh, Tài chính, Thị trường, Chứng khoán
được gán nhãn 1
 Các bài thuộc các mục khác như Thể thao, Giải trí, Giáo dục, Đời sống, Sức khỏe, Công
nghệ, Du lịch... được gán nhãn 0
5.4. Quy mô dữ liệu
Khuyến nghị mỗi nhóm chuẩn bị tối thiểu:
 2000 bài báo
 trong đó nên có sự mất cân bằng lớp vừa phải, ví dụ:
o 600–800 bài thuộc lớp 1
o 1200–1600 bài thuộc lớp 0
Mục đích: để quan sát được ảnh hưởng của mất cân bằng dữ liệu đến kết quả phân loại.
5.5. Làm sạch dữ liệu
Sinh viên cần kiểm tra và xử lý:
 bài trùng lặp
 bài rỗng hoặc thiếu nội dung
 bài có nội dung quá ngắn
 lỗi mã hóa ký tự nếu có
6. Yêu cầu thực hiện cụ thể
Ngày 20/03/2026
Phần A. Chuẩn bị dữ liệu
Sinh viên cần:
1. Thu thập dữ liệu
2. Gán nhãn
3. Mô tả tập dữ liệu:
o tổng số mẫu
o số mẫu mỗi lớp
o tỷ lệ giữa hai lớp
o ví dụ minh họa một vài mẫu dữ liệu
Cần nhận xét: dữ liệu có cân bằng hay không, và dự đoán trước điều này có thể ảnh hưởng thế
nào đến mô hình.
Gợi ý cấu trúc CSV tối thiểu:
 id: mã bài viết
 url: link bài báo
 category: chuyên mục gốc của bài báo
 label: nhãn phân loại
 title: tiêu đề
 desc: mô tả ngắn / sapo
 text: nội dung bài báo
 Lưu ý: Ở bước Tiền xử lý, nên copy dữ liệu sang 1 cột riêng
Phần B. Tiền xử lý văn bản
Sinh viên thực hiện các bước tiền xử lý đã học, ví dụ:
 chuẩn hóa chữ
 loại bỏ ký tự đặc biệt không cần thiết
 tách từ
 loại bỏ stopwords nếu thấy phù hợp
 ghép các trường văn bản để tạo đầu vào cuối cùng
Sinh viên phải mô tả rõ:
 đầu vào mô hình là gì
(ví dụ chỉ dùng title, hoặc title + summary, hoặc title + content)
 các bước tiền xử lý đã áp dụng
 lý do chọn cách đó
Phần C. Biểu diễn văn bản
Sinh viên phải thử ít nhất 2 cách biểu diễn đặc trưng:
1. Bag of Words
2. TF-IDF
Có thể bắt đầu với unigram. Nhóm nào muốn mở rộng có thể thử thêm bigram.
Phần D. Xây dựng mô hình
Sinh viên phải huấn luyện và so sánh ít nhất 3 mô hình:
1. Multinomial Naive Bayes
2. Logistic Regression
3. Support Vector Machine (SVM)
Lưu ý:
 Cần giữ quy trình xử lý tương đối thống nhất để việc so sánh công bằng.
 Phải mô tả rõ dữ liệu được chia như thế nào, ví dụ train/test hoặc train/validation/test.
Phần E. Đánh giá mô hình
Sinh viên bắt buộc đánh giá bằng các chỉ số sau:
 Accuracy
 Precision
 Recall
 F1-score
 Confusion Matrix
Ngoài ra, sinh viên phải phân tích riêng cho lớp Kinh tế/Kinh doanh, không chỉ nêu kết quả chung.
Yêu cầu phân tích
Sinh viên cần trả lời các câu hỏi như:
 Mô hình nào cho accuracy cao nhất?
 Mô hình nào nhận diện tốt hơn lớp Kinh tế/Kinh doanh?
 Có trường hợp nào accuracy cao nhưng kết quả thực tế vẫn chưa tốt không?
 Dữ liệu mất cân bằng ảnh hưởng như thế nào?
Ngày 27/03/2026
Phần F. Cải tiến mô hình
Sau khi chạy kết quả ban đầu, sinh viên phải thực hiện ít nhất 2 hướng cải tiến và phân tích tác
động của từng hướng.
Có thể chọn trong các hướng sau:
Hướng 1: Giảm số lượng đặc trưng
Ví dụ giới hạn số feature để xem việc giảm độ thưa có giúp mô hình tốt hơn hay không.
Hướng 2: Xử lý mất cân bằng lớp
Ví dụ:
 class weight
 resampling
Hướng 3: So sánh BoW và TF-IDF
Phân tích xem biểu diễn nào phù hợp hơn với bài toán.
Hướng 4: Thử n-gram
Ví dụ so sánh unigram với unigram + bigram.
Hướng 5: Thử đầu vào khác nhau
Ví dụ:
 chỉ dùng title
 dùng title + summary
 dùng full content
Hướng 6: Điều chỉnh tham số mô hình
Ví dụ tinh chỉnh các tham số cơ bản của Naive Bayes, Logistic Regression hoặc SVM.
Mỗi cải tiến phải có:
 lý do thử
 cách thực hiện
 kết quả trước và sau
 nhận xét
7. Nội dung báo cáo cần nộp
Mỗi nhóm nộp 1 báo cáo với các phần sau:
7.1. Giới thiệu bài toán
 Mô tả bài toán
7.2. Dữ liệu
 Nguồn dữ liệu
 Cách crawl
 Cách gán nhãn
 Quy mô dữ liệu
 Phân bố lớp
 Ví dụ dữ liệu
7.3. Tiền xử lý
 Các bước đã thực hiện
 Giải thích lý do
7.4. Biểu diễn dữ liệu
 BoW
 TF-IDF
 Các thiết lập chính
7.5. Mô hình
 Naive Bayes
 Logistic Regression
 SVM
 Cách chia tập dữ liệu
 Cách huấn luyện
7.6. Kết quả thực nghiệm
 Bảng so sánh kết quả
 Confusion Matrix
 Nhận xét chi tiết
7.7. Cải tiến
 Mô tả ít nhất 2 cải tiến
 So sánh kết quả trước/sau
 Nhận xét
7.8. Kết luận
 Mô hình tốt nhất theo nhóm là mô hình nào
 Vì sao chọn mô hình đó
 Bài học rút ra từ thực nghiệm
8. Sản phẩm cần nộp
Mỗi nhóm nộp:
1. File báo cáo
2. Notebook hoặc source code
3. File dữ liệu đã xử lý hoặc đường dẫn tới dữ liệu
4. File README ngắn hướng dẫn cách chạy