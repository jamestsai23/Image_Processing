import sys
import cv2
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QFileDialog, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QGridLayout, QLineEdit, QGroupBox, QWidget, QDialog, QFormLayout, QSlider

class ResultWindow(QDialog):
    def __init__(self, image, title="Result"):
        super().__init__()
        self.setWindowTitle(title)
        self.setGeometry(300, 300, 600, 400)
        
        layout = QVBoxLayout()
        
        # Display result image
        label = QLabel(self)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        label.setPixmap(QtGui.QPixmap.fromImage(qt_image))
        
        layout.addWidget(label)
        self.setLayout(layout)

class ImageProcessingApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.image = None
        self.image2 = None  # For additional image loading if needed

    def initUI(self):
        # Main Layout
        main_layout = QGridLayout()

        # Center Load Images Buttons (Column 0)
        load_image_layout = QVBoxLayout()
        load_image_layout.addStretch(1)  # Top Spacer
        self.loadButton1 = QPushButton("Load Image 1")
        self.loadButton1.clicked.connect(self.load_image1)
        load_image_layout.addWidget(self.loadButton1)

        self.loadButton2 = QPushButton("Load Image 2")
        self.loadButton2.clicked.connect(self.load_image2)
        load_image_layout.addWidget(self.loadButton2)
        load_image_layout.addStretch(1)  # Bottom Spacer

        main_layout.addLayout(load_image_layout, 0, 0, 3, 1)  # Column 0, spanning 3 rows

        # Image Processing Group (Column 1)
        image_processing_group = QGroupBox("1. Image Processing")
        image_processing_layout = QVBoxLayout()
        
        self.colorSeparationButton = QPushButton("1.1 Color Separation")
        self.colorSeparationButton.clicked.connect(self.color_separation)
        image_processing_layout.addWidget(self.colorSeparationButton)

        self.colorTransformationButton = QPushButton("1.2 Color Transformation")
        self.colorTransformationButton.clicked.connect(self.color_transformation)
        image_processing_layout.addWidget(self.colorTransformationButton)

        self.colorExtractionButton = QPushButton("1.3 Color Extraction")
        self.colorExtractionButton.clicked.connect(self.color_extraction)
        image_processing_layout.addWidget(self.colorExtractionButton)

        image_processing_group.setLayout(image_processing_layout)
        main_layout.addWidget(image_processing_group, 0, 1)

        # Image Smoothing Group (Column 1)
        image_smoothing_group = QGroupBox("2. Image Smoothing")
        image_smoothing_layout = QVBoxLayout()
        
        self.gaussianButton = QPushButton("2.1 Gaussian Blur")
        self.gaussianButton.clicked.connect(self.gaussian_blur)
        image_smoothing_layout.addWidget(self.gaussianButton)

        self.bilateralButton = QPushButton("2.2 Bilateral Filter")
        self.bilateralButton.clicked.connect(self.bilateral_filter)
        image_smoothing_layout.addWidget(self.bilateralButton)

        self.medianButton = QPushButton("2.3 Median Filter")
        self.medianButton.clicked.connect(self.median_filter)
        image_smoothing_layout.addWidget(self.medianButton)

        image_smoothing_group.setLayout(image_smoothing_layout)
        main_layout.addWidget(image_smoothing_group, 1, 1)

        # Edge Detection Group (Column 1)
        edge_detection_group = QGroupBox("3. Edge Detection")
        edge_detection_layout = QVBoxLayout()
        
        self.sobelXButton = QPushButton("3.1 Sobel X")
        self.sobelXButton.clicked.connect(self.sobel_x)
        edge_detection_layout.addWidget(self.sobelXButton)

        self.sobelYButton = QPushButton("3.2 Sobel Y")
        self.sobelYButton.clicked.connect(self.sobel_y)
        edge_detection_layout.addWidget(self.sobelYButton)

        self.combinationButton = QPushButton("3.3 Combination and Threshold")
        self.combinationButton.clicked.connect(self.combination_threshold)
        edge_detection_layout.addWidget(self.combinationButton)

        self.gradientButton = QPushButton("3.4 Gradient Angle")
        self.gradientButton.clicked.connect(self.gradient_angle)
        edge_detection_layout.addWidget(self.gradientButton)

        edge_detection_group.setLayout(edge_detection_layout)
        main_layout.addWidget(edge_detection_group, 2, 1)

        # Transforms Group (Column 2)
        transforms_group = QGroupBox("4. Transforms")
        transforms_layout = QFormLayout()  # Using QFormLayout for alignment
        transforms_layout.setContentsMargins(10, 10, 10, 5)  # Set margins to reduce bottom space

        # Define a fixed width for all input fields for consistency
        input_width = 80

        self.rotationInput = QLineEdit()
        self.rotationInput.setFixedWidth(input_width)
        rotation_layout = QHBoxLayout()
        rotation_layout.addWidget(self.rotationInput)
        rotation_layout.addWidget(QLabel("deg"))
        transforms_layout.addRow("Rotation:", rotation_layout)

        self.scaleInput = QLineEdit()
        self.scaleInput.setFixedWidth(input_width)
        transforms_layout.addRow("Scaling:", self.scaleInput)

        self.txInput = QLineEdit()
        self.txInput.setFixedWidth(input_width)
        tx_layout = QHBoxLayout()
        tx_layout.addWidget(self.txInput)
        tx_layout.addWidget(QLabel("pixel"))
        transforms_layout.addRow("Tx:", tx_layout)

        self.tyInput = QLineEdit()
        self.tyInput.setFixedWidth(input_width)
        ty_layout = QHBoxLayout()
        ty_layout.addWidget(self.tyInput)
        ty_layout.addWidget(QLabel("pixel"))
        transforms_layout.addRow("Ty:", ty_layout)

        self.transformButton = QPushButton("4. Transforms")
        self.transformButton.clicked.connect(self.transforms)
        transforms_layout.addRow(self.transformButton)

        transforms_group.setLayout(transforms_layout)
        main_layout.addWidget(transforms_group, 0, 2, 2, 1)  # Only spans two rows in Column 2

        # Adjust column stretch to control widths
        main_layout.setColumnStretch(0, 1)
        main_layout.setColumnStretch(1, 2)
        main_layout.setColumnStretch(2, 1)

        # Set background color to light gray
        self.setStyleSheet("""
            QWidget {
                background-color: #d3d3d3;  /* Light gray background */
                color: black;
            }
            QGroupBox {
                border: 1px solid gray;
                margin-top: 10px;
                color: black;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 3px;
            }
            QPushButton {
                background-color: #f0f0f0;
                border: 1px solid #a9a9a9;
                padding: 5px;
                color: black;
            }
            QLineEdit {
                background-color: #ffffff;
                color: black;
                padding: 2px;
                border: 1px solid gray;
            }
            QLabel {
                color: black;
            }
        """)

        self.setLayout(main_layout)
        self.setWindowTitle('Hw1 - Image Processing GUI')
        self.setGeometry(200, 200, 800, 600)

    def load_image1(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Image Files (*.png *.jpg *.bmp)")
        if fileName:
            self.image = cv2.imread(fileName)

    def load_image2(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Image Files (*.png *.jpg *.bmp)")
        if fileName:
            self.image2 = cv2.imread(fileName)

    # def show_result(self, result_image, title="Result"):
    #     result_window = ResultWindow(result_image, title)
    #     result_window.exec_()

    def show_result(self, result_image, title="Result"):
        # 創建新的非模態視窗
        result_window = QDialog(self)
        result_window.setWindowTitle(title)
        result_window.setGeometry(300, 300, 600, 400)
    
        layout = QVBoxLayout(result_window)
        
        # 顯示圖片
        label = QLabel(result_window)
        rgb_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        label.setPixmap(QtGui.QPixmap.fromImage(qt_image))
        
        layout.addWidget(label)
        result_window.setLayout(layout)
        
        # 使用非模態方式顯示
        result_window.setWindowModality(QtCore.Qt.NonModal)
        result_window.show()
        
        # 保存視窗引用，防止被垃圾回收
        if not hasattr(self, 'result_windows'):
            self.result_windows = []
        self.result_windows.append(result_window)


    def color_separation(self):
        if self.image is None:
            return

        # 分割 RGB 通道
        b, g, r = cv2.split(self.image)
        zeros = np.zeros(b.shape, dtype=np.uint8)

        # 創建單色圖片
        b_image = cv2.merge([b, zeros, zeros])
        g_image = cv2.merge([zeros, g, zeros])
        r_image = cv2.merge([zeros, zeros, r])

        # 分別顯示三個通道圖片，每個通道在一個新視窗中
        self.show_result(b_image, "Blue Channel")
        self.show_result(g_image, "Green Channel")
        self.show_result(r_image, "Red Channel")


    def color_transformation(self):
        if self.image is None:
            return

        #cv_gray
        cv_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.show_result(cv_gray, "cv_gray")

        #avg_gray
        b, g, r = cv2.split(self.image)
        avg_gray=(b/3 + g/3 + r/3).astype(np.uint8)
        self.show_result(avg_gray, "avg_gray")


    def color_extraction(self):
        if self.image is None:
            return
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        lower_bound = np.array([18, 0, 25])
        upper_bound = np.array([85, 255, 255])
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        mask_inv = cv2.bitwise_not(mask)
        extracted_image = cv2.bitwise_and(self.image, self.image, mask=mask_inv)
        self.show_result(mask, "Yellow-Green Mask")
        self.show_result(extracted_image, "Extracted Image")


    def gaussian_blur(self):
        if self.image is None:
            return

        # 創建彈出窗口
        dialog = QDialog(self)
        dialog.setWindowTitle("img 1")
        dialog.setGeometry(300, 300, 400, 400)

        # 設置垂直布局
        layout = QVBoxLayout()

        # 創建水平布局，用於放置 m 值標籤和滑動條
        slider_layout = QHBoxLayout()
        
        # 創建 QLabel 用於顯示當前的 m 值，初始設為 1
        m_label = QLabel("m: 1", dialog)
        slider_layout.addWidget(m_label)

        # 創建分段滑動條
        slider = QSlider(QtCore.Qt.Horizontal)
        slider.setMinimum(1)
        slider.setMaximum(5)
        slider.setTickInterval(1)  # 設置滑動條的刻度間隔為 1
        slider.setSingleStep(1)    # 設置步進為 1
        slider.setTickPosition(QSlider.TicksBelow)
        slider_layout.addWidget(slider)

        # 將水平布局（包含 m 標籤和滑動條）添加到主布局中
        layout.addLayout(slider_layout)

        # 創建 QLabel 用於顯示圖像
        image_label = QLabel(dialog)
        layout.addWidget(image_label)

        # 初次顯示模糊結果
        self.update_blur(1, image_label, m_label)

        # 當滑動條值變化時，更新圖像模糊和 m 值顯示
        slider.valueChanged.connect(lambda m: self.update_blur(m, image_label, m_label))

        # 設置彈出窗口的布局
        dialog.setLayout(layout)
        dialog.exec_()

    def update_blur(self, m, image_label, m_label):
        # 更新 m 值標註
        m_label.setText(f"m: {m}")

        # 計算高斯模糊的核大小
        kernel_size = (2 * m + 1, 2 * m + 1)
        blurred_image = cv2.GaussianBlur(self.image, kernel_size, 0)

        # 將圖像轉換為 Qt 格式並顯示
        rgb_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        image_label.setPixmap(QtGui.QPixmap.fromImage(qt_image))

    def bilateral_filter(self):
        if self.image is None:
            return

        # 創建彈出窗口
        dialog = QDialog(self)
        dialog.setWindowTitle("img 1")
        dialog.setGeometry(300, 300, 400, 400)

        # 設置垂直布局
        layout = QVBoxLayout()

        # 創建水平布局，用於放置 m 值標籤和滑動條
        slider_layout = QHBoxLayout()
        
        # 創建 QLabel 用於顯示當前的 m 值，初始設為 1
        m_label = QLabel("m: 1", dialog)
        slider_layout.addWidget(m_label)

        # 創建分段滑動條
        slider = QSlider(QtCore.Qt.Horizontal)
        slider.setMinimum(1)
        slider.setMaximum(5)
        slider.setTickInterval(1)  # 設置滑動條的刻度間隔為 1
        slider.setSingleStep(1)    # 設置步進為 1
        slider.setTickPosition(QSlider.TicksBelow)
        slider_layout.addWidget(slider)

        # 將水平布局（包含 m 標籤和滑動條）添加到主布局中
        layout.addLayout(slider_layout)

        # 創建 QLabel 用於顯示圖像
        image_label = QLabel(dialog)
        layout.addWidget(image_label)

        # 初次顯示雙邊濾波結果
        self.update_bilateral_filter(1, image_label, m_label)

        # 當滑動條值變化時，更新圖像濾波和 m 值顯示
        slider.valueChanged.connect(lambda m: self.update_bilateral_filter(m, image_label, m_label))

        # 設置彈出窗口的布局
        dialog.setLayout(layout)
        dialog.exec_()

    def update_bilateral_filter(self, m, image_label, m_label):
        # 更新 m 值標註
        m_label.setText(f"m: {m}")

        # 計算直徑 d (2 * m + 1)
        d = 2 * m + 1
        sigmaColor = 90
        sigmaSpace = 90

        # 應用雙邊濾波
        bilateral_image = cv2.bilateralFilter(self.image, d, sigmaColor, sigmaSpace)

        # 將圖像轉換為 Qt 格式並顯示
        rgb_image = cv2.cvtColor(bilateral_image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        image_label.setPixmap(QtGui.QPixmap.fromImage(qt_image))

    def median_filter(self):
        if self.image2 is None:
            return

        # 創建彈出窗口
        dialog = QDialog(self)
        dialog.setWindowTitle("img 2")
        dialog.setGeometry(300, 300, 400, 400)

        # 設置垂直布局
        layout = QVBoxLayout()

        # 創建水平布局，用於放置 m 值標籤和滑動條
        slider_layout = QHBoxLayout()
        
        # 創建 QLabel 用於顯示當前的 m 值，初始設為 1
        m_label = QLabel("m: 1", dialog)
        slider_layout.addWidget(m_label)

        # 創建分段滑動條
        slider = QSlider(QtCore.Qt.Horizontal)
        slider.setMinimum(1)
        slider.setMaximum(5)
        slider.setTickInterval(1)  # 設置滑動條的刻度間隔為 1
        slider.setSingleStep(1)    # 設置步進為 1
        slider.setTickPosition(QSlider.TicksBelow)
        slider_layout.addWidget(slider)

        # 將水平布局（包含 m 標籤和滑動條）添加到主布局中
        layout.addLayout(slider_layout)

        # 創建 QLabel 用於顯示圖像
        image_label = QLabel(dialog)
        layout.addWidget(image_label)

        # 初次顯示中值濾波結果
        self.update_median_filter(1, image_label, m_label)

        # 當滑動條值變化時，更新圖像濾波和 m 值顯示
        slider.valueChanged.connect(lambda m: self.update_median_filter(m, image_label, m_label))

        # 設置彈出窗口的布局
        dialog.setLayout(layout)
        dialog.exec_()

    def update_median_filter(self, m, image_label, m_label):
        # 更新 m 值標註
        m_label.setText(f"m: {m}")

        # 計算中值濾波的核大小 (2m + 1)
        kernel_size = 2 * m + 1

        # 應用中值濾波
        median_image = cv2.medianBlur(self.image2, kernel_size)

        # 將圖像轉換為 Qt 格式並顯示
        rgb_image = cv2.cvtColor(median_image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        image_label.setPixmap(QtGui.QPixmap.fromImage(qt_image))

    def compute_sobel_x(self):
        if self.image is None:
            return

        # 將彩色圖像轉換為灰度圖像
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # 設定高斯模糊的參數
        kernel_size = 3  # 固定使用 3x3 的核大小
        sigmaX = 0
        sigmaY = 0

        # 應用高斯模糊
        blur = cv2.GaussianBlur(gray, (kernel_size, kernel_size), sigmaX=sigmaX, sigmaY=sigmaY)

        # 定義自定義的 3x3 Sobel X 卷積核
        sobel_x_kernel = np.array([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]], dtype=np.float32)

        # 手動執行卷積操作
        sobel_x = np.zeros_like(blur, dtype=np.float32)
        rows, cols = blur.shape

        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                region = blur[i - 1:i + 2, j - 1:j + 2]
                sobel_x[i, j] = np.sum(region * sobel_x_kernel)

        # 將結果轉換為可顯示的格式
        # sobel_x = np.abs(sobel_x)
        # sobel_x = np.clip(sobel_x, 0, 255).astype(np.uint8)

        return sobel_x

    def compute_sobel_y(self):
        if self.image is None:
            return

        # 將彩色圖像轉換為灰度圖像
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # 設定高斯模糊的參數
        kernel_size = 3  # 固定使用 3x3 的核大小
        sigmaX = 0
        sigmaY = 0

        # 應用高斯模糊
        blur = cv2.GaussianBlur(gray, (kernel_size, kernel_size), sigmaX=sigmaX, sigmaY=sigmaY)

        # 定義自定義的 3x3 Sobel X 卷積核
        sobel_y_kernel = np.array([[-1, -2, -1],
                                [0, 0, 0],
                                [1, 2, 1]], dtype=np.float32)

        # 手動執行卷積操作
        sobel_y = np.zeros_like(blur, dtype=np.float32)
        rows, cols = blur.shape

        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                region = blur[i - 1:i + 2, j - 1:j + 2]
                sobel_y[i, j] = np.sum(region * sobel_y_kernel)

        # 將結果轉換為可顯示的格式
        # sobel_y = np.abs(sobel_y)
        # sobel_y = np.clip(sobel_y, 0, 255).astype(np.uint8)

        return sobel_y

    def sobel_x(self):
        
        sobel_x = self.compute_sobel_x()
        sobel_x = np.abs(sobel_x)
        sobel_x = np.clip(sobel_x, 0, 255).astype(np.uint8)

        # 使用 OpenCV 顯示結果
        cv2.imshow("sobel x", sobel_x)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def sobel_y(self):
        sobel_y = self.compute_sobel_y()
        sobel_y = np.abs(sobel_y)
        sobel_y = np.clip(sobel_y, 0, 255).astype(np.uint8)

        # 使用 OpenCV 顯示結果
        cv2.imshow("sobel y", sobel_y)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def compute_combination(self):
        if self.image is None:
            return

        # 4. 使用 cv2.Sobel 計算 Sobel X 和 Sobel Y 邊緣檢測結果
        sobel_x = self.compute_sobel_x()
        # sobel_x = np.abs(sobel_x)
        # sobel_x = np.clip(sobel_x, 0, 255).astype(np.uint8)
        
        sobel_y = self.compute_sobel_y()        
        # sobel_y = np.abs(sobel_y)
        # sobel_y = np.clip(sobel_y, 0, 255).astype(np.uint8)

        # 5. 計算 Sobel X 和 Sobel Y 的平方和
        sobel_x_squared = np.square(sobel_x.astype(np.float32))
        sobel_y_squared = np.square(sobel_y.astype(np.float32))
        combination = np.sqrt(sobel_x_squared + sobel_y_squared)

        # 6. 將結果正規化到 0 ~ 255 範圍
        normalized = cv2.normalize(combination, None, 0, 255, cv2.NORM_MINMAX)
        normalized = normalized.astype(np.uint8)

        return normalized

    def combination_threshold(self):
        if self.image is None:
            return

        # 4. 使用 cv2.Sobel 計算 Sobel X 和 Sobel Y 邊緣檢測結果
        sobel_x = self.compute_sobel_x()
        sobel_x = np.abs(sobel_x)
        sobel_x = np.clip(sobel_x, 0, 255).astype(np.uint8)
        
        sobel_y = self.compute_sobel_y()        
        sobel_y = np.abs(sobel_y)
        sobel_y = np.clip(sobel_y, 0, 255).astype(np.uint8)

        # 5. 計算 Sobel X 和 Sobel Y 的平方和
        sobel_x_squared = np.square(sobel_x.astype(np.float32))
        sobel_y_squared = np.square(sobel_y.astype(np.float32))
        combination = np.sqrt(sobel_x_squared + sobel_y_squared)

        # 6. 將結果正規化到 0 ~ 255 範圍
        normalized = cv2.normalize(combination, None, 0, 255, cv2.NORM_MINMAX)
        normalized = normalized.astype(np.uint8)

        self.combination_threshold_result = normalized

        # 7. 設定兩種閾值：128 和 28
        _, threshold_result_128 = cv2.threshold(normalized, 128, 255, cv2.THRESH_BINARY)
        _, threshold_result_28 = cv2.threshold(normalized, 28, 255, cv2.THRESH_BINARY)

        # 8. 顯示合併結果和閾值處理結果
        cv2.imshow("sobel xy", normalized)
        cv2.imshow("threshold", threshold_result_128)
        cv2.imshow("threshold_2", threshold_result_28)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def gradient_angle(self):
        if self.image is None:
            return

        sobel_x = self.compute_sobel_x()
        sobel_y = self.compute_sobel_y()
        combine = self.compute_combination()

        gradient_angle = np.arctan2(sobel_y, sobel_x) * 180 / np.pi
        gradient_angle[gradient_angle < 0 ] += 360
        
        #generate mask
        mask1 = ((gradient_angle >= 170) & (gradient_angle <= 190)).astype(np.uint8) * 255
        mask2 = ((gradient_angle >= 260) & (gradient_angle <= 280)).astype(np.uint8) * 255
        mask1_pic = cv2.bitwise_and(combine, mask1)
        mask2_pic = cv2.bitwise_and(combine, mask2)

        cv2.imshow("angle 1", mask1_pic)
        cv2.imshow("angle 2", mask2_pic)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def transforms(self):
        if self.image is None:
            return

        # 讀取默認或輸入的參數值
        angle = float(self.rotationInput.text()) if self.rotationInput.text() else 0  # 預設角度為 0 度
        scale = float(self.scaleInput.text()) if self.scaleInput.text() else 0.       # 預設縮放比例為 0
        tx = int(self.txInput.text()) if self.txInput.text() else 0                   # 預設平移 X 為 0
        ty = int(self.tyInput.text()) if self.tyInput.text() else 0                   # 預設平移 Y 為 0

        # 取得圖片的中心點作為旋轉中心
        rows, cols, _ = self.image.shape
        center_point = (240, 200)  # 圖片中漢堡的原始中心點

        # 1. 計算旋轉矩陣
        rotation_matrix = cv2.getRotationMatrix2D(center_point, angle, scale)

        # 2. 加入平移參數到旋轉矩陣中
        # 根據提供的平移參數 (Xnew = Xold + 535, Ynew = Yold + 335)
        rotation_matrix[0, 2] += tx
        rotation_matrix[1, 2] += ty

        # 3. 應用仿射變換
        transformed_image = cv2.warpAffine(self.image, rotation_matrix, (cols, rows))

        # 顯示結果
        self.show_result(transformed_image, "Transformed Burger")



app = QtWidgets.QApplication(sys.argv)
window = ImageProcessingApp()
window.show()
sys.exit(app.exec_())
