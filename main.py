import sys 
from PyQt5.QtWidgets import (QApplication, QLabel, QPushButton, QFileDialog, QWidget, QButtonGroup, QRadioButton, QDockWidget, QAction, QLineEdit, QGridLayout, 
                             QVBoxLayout, QHBoxLayout, QDialog, QDialogButtonBox, QToolBar, QMainWindow, QDockWidget, QStyle, QGraphicsDropShadowEffect,
                             QProgressBar, QTextEdit, QSlider, QComboBox, QSpacerItem, QSizePolicy, QTabWidget)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QIcon, QColor
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import numpy as np
from PIL import Image
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler, StableDiffusionInpaintPipeline
import torch
import os
from segment_anything import sam_model_registry, SamPredictor

# 新增 DraggableLabel 类，放在 ImageSegmentationApp 之前
class DraggableLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.is_dragging = False  # 初始化 is_dragging
        self.dragging = False
        self.last_mouse_position = None
        self.offset_x = 0  # 偏移量
        self.offset_y = 0  # 偏移量
        self.scale_factor = 1.0  # 初始化缩放比例
        self.drag_start_position = None
        self.setScaledContents(True)  # 允许内容缩放
        
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.pixmap():
            self.is_dragging = True
            self.last_mouse_position = event.pos()

    def mouseMoveEvent(self, event):
        if self.is_dragging and self.pixmap():
            delta = event.pos() - self.last_mouse_position
            self.offset_x += delta.x()
            self.offset_y += delta.y()
            self.last_mouse_position = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.is_dragging = False

    def paintEvent(self, event):
        if self.pixmap():  # 确保 pixmap 存在
            painter = QPainter(self)
            painter.translate(self.offset_x, self.offset_y)  # 根据偏移量移动图片
            scaled_pixmap = self.pixmap().scaled(
                self.pixmap().size() * self.scale_factor,  # 根据缩放比例调整大小
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            painter.drawPixmap(0, 0, scaled_pixmap)  # 绘制缩放后的 pixmap
            painter.end()

    def wheelEvent(self, event):
        angle = event.angleDelta().y()
        if angle > 0:
            self.scale_factor *= 1.1  # 放大
        else:
            self.scale_factor /= 1.1  # 缩小
        self.scale_factor = max(0.1, min(self.scale_factor, 10))
        self.update()

    def update_pixmap(self):
        """这个方法已经不需要再独立调用，直接通过paintEvent的update来进行"""
        self.update()

class ImageSegmentationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.image_path = None
        self.result_pixmap = None
        self.point_coords = None
        self.mask = None  # 存储分割的蒙版
        self.scale_factor = 1.0 

        # 模型路径字典
        self.model_paths = {
            'majicMIX realistic 麦橘写实_v7': "G:/BaiduNetdiskDownload/ComfyUI_windows_portable/ComfyUI/models/checkpoints/majicMIX realistic 麦橘写实_v7.safetensors",
            'dreamshaper_8': "G:/BaiduNetdiskDownload/ComfyUI_windows_portable/ComfyUI/models/checkpoints/dreamshaper_8.safetensors",
            'realisticVisionV60B1_v51HyperVAE': "G:/BaiduNetdiskDownload/ComfyUI_windows_portable/ComfyUI/models/checkpoints/realisticVisionV60B1_v51HyperVAE.safetensors",
            '麒麟-revAnimated_v122_V1.2.2': "G:/BaiduNetdiskDownload/ComfyUI_windows_portable/ComfyUI/models/checkpoints/麒麟-revAnimated_v122_V1.2.2.safetensors",
            'ChilloutMix_Chilloutmix-Ni': "G:/BaiduNetdiskDownload/ComfyUI_windows_portable/ComfyUI/models/checkpoints/真实感必备模型｜ChilloutMix_Chilloutmix-Ni-pruned-fp32-fix.safetensors"
        }
        self.current_model_path = self.model_paths['majicMIX realistic 麦橘写实_v7']  # 默认选择模型

        self.initUI()
        self.sam = self.load_sam_model()
        self.sd_model = self.load_sd_model()
        self.inpainting_pipeline = self.load_inpainting_pipeline()  # 加载Inpainting模型

    def initUI(self):
        self.setWindowTitle('AI Image Segmentation & Generation Tool')
        self.setGeometry(800, 600, 2100, 1200)


        # 主布局
        main_widget = QWidget()  # 创建一个中央 widget
        main_layout = QHBoxLayout(main_widget)

        # 创建第一个 QDockWidget：用于模型选择和图像上传
        model_upload_dock = QDockWidget("Model & Upload", self)
        model_upload_dock.setAllowedAreas(Qt.LeftDockWidgetArea)  # 允许吸附在左侧
        model_upload_dock.setFloating(False)  # 初始状态不浮动
        model_upload_widget = QWidget()
        model_upload_layout = QVBoxLayout(model_upload_widget)

        # 模型选择
        checkpoint_layout = QHBoxLayout()
        checkpoint_label = QLabel("CheckPoint:", self)
        checkpoint_label.setFixedWidth(130)
        self.model_selector = QComboBox(self)
        self.model_selector.addItems(self.model_paths.keys())
        self.model_selector.currentIndexChanged.connect(self.on_model_selection_change)
        self.model_selector.setFixedWidth(300)
        checkpoint_layout.addWidget(checkpoint_label)
        checkpoint_layout.addWidget(self.model_selector)
        model_upload_layout.addLayout(checkpoint_layout)

        # 灰色窗口显示区域
        self.upload_area = QLabel(self)
        self.upload_area.setFixedSize(480, 270)
        self.upload_area.setStyleSheet("""
            background-color: #2E2F34;  /* 背景颜色 */
            border: 1px solid #444;      /* 边框颜色 */
            border-radius: 15px;         /* 圆角半径 */
            padding: 10px;               /* 内边距 */
            color: white;
            font-size: 16px;
        """)
        self.upload_area.setAlignment(Qt.AlignCenter)
        self.upload_area.mousePressEvent = self.open_selection_window

        # 上传按钮
        self.upload_button = QPushButton('Upload Image', self)
        self.upload_button.clicked.connect(self.upload_image)
        self.upload_button.setFixedSize(270, 30)
        self.upload_area_layout = QVBoxLayout(self.upload_area)
        self.upload_area_layout.addWidget(self.upload_button, alignment=Qt.AlignCenter)
        self.overlay_label = QLabel(self.upload_area)
        self.overlay_label.setFixedSize(480, 270)
        self.overlay_label.setStyleSheet("background-color: rgba(0, 0, 0, 76); color: white; font-size: 16px;")
        self.overlay_label.setText("点击选择范围")
        self.overlay_label.setAlignment(Qt.AlignCenter)
        self.overlay_label.hide()
        model_upload_layout.addWidget(self.upload_area)

        # 重选按钮
        self.reupload_button = QPushButton('Re-upload Image', self)
        self.reupload_button.clicked.connect(self.reupload_image)
        self.reupload_button.setFixedWidth(480)
        model_upload_layout.addWidget(self.reupload_button)

        model_upload_widget.setLayout(model_upload_layout)
        model_upload_dock.setWidget(model_upload_widget)
        self.addDockWidget(Qt.LeftDockWidgetArea, model_upload_dock)

        # 创建第二个 QDockWidget：用于 Prompt 输入、比例选择和生成按钮
        prompt_dock = QDockWidget("Prompt & Settings", self)
        prompt_dock.setAllowedAreas(Qt.LeftDockWidgetArea)  # 允许吸附在左侧
        prompt_dock.setFloating(False)  # 初始状态不浮动
        prompt_widget = QWidget()
        prompt_layout = QVBoxLayout(prompt_widget)

        # 添加比例选择按钮
        self.create_aspect_ratio_buttons()
        prompt_layout.addLayout(self.aspect_ratio_layout)

        prompt_widget.setLayout(prompt_layout)
        prompt_dock.setWidget(prompt_widget)
        self.addDockWidget(Qt.LeftDockWidgetArea, prompt_dock)

        # 设置初始吸附顺序
        self.splitDockWidget(model_upload_dock, prompt_dock, Qt.Vertical)

        # 右侧结果显示区域
        result_widget = QWidget()
        result_layout = QVBoxLayout(result_widget)

        # 创建顶部工具栏，设置为 QDockWidget 以便拖动
        self.top_bar_dock = QDockWidget("Top Bar", self)
        self.top_bar_dock.setAllowedAreas(Qt.TopDockWidgetArea)  # 允许停靠在顶部
        self.top_bar_dock.setFloating(False)

        # 创建工具栏
        top_toolbar = QToolBar(self.top_bar_dock)
        top_toolbar.setMovable(True)  # 允许移动

        # 添加选择框（类似你示例中的下拉菜单+图标）
        self.model_selector = QComboBox(self)
        self.model_selector.addItem(QIcon("F:/path_to_icon1.png"), "Model 1")  # 替换为你的图标路径
        self.model_selector.addItem(QIcon("F:/path_to_icon2.png"), "Model 2")
        top_toolbar.addWidget(self.model_selector)

        # 添加比例按钮
        aspect_ratio_action = QAction("1:1", self)
        top_toolbar.addAction(aspect_ratio_action)

        # 将工具栏添加到 dock 中
        self.top_bar_dock.setWidget(top_toolbar)

        # 将顶部栏插入到 result_layout 的顶部
        result_layout.insertWidget(0, self.top_bar_dock)

        # 将其他现有代码添加到 result_layout
        self.result_area = DraggableLabel(self)
        # 设置结果区域的样式
        self.result_area.setStyleSheet("""
            background-color: #2c2f33; /* 内部的深灰色 */
            border: 1px solid #1e2124; /* 边框颜色，可以略深 */
            border-radius: 15px; /* 圆角半径 */
        """)
        result_layout.setContentsMargins(15, 15, 15, 15)  # 设置外边距，增加卡片感

        result_layout.addWidget(self.result_area)

        # 将悬浮的输入框放入结果显示区域内
        self.createFloatingInputWidget()

        # 设置主布局和中央控件
        main_layout.addWidget(result_widget)
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        # 将 result_area 的 resizeEvent 绑定到更新悬浮输入框位置的方法
        self.result_area.resizeEvent = self.updateFloatingWidgetPosition

        # 在 result_area 内部创建垂直布局
        result_area_layout = QVBoxLayout(self.result_area)
        result_area_layout.setContentsMargins(0, 0, 0, 0)  # 移除边距
        result_area_layout.setSpacing(0)  # 移除组件间的间距

        # 创建进度条并设置样式
        #text-align: center;  /* 将文字居中 */
        self.progress_bar = QProgressBar(self.result_area)
        self.progress_bar.setFixedHeight(4)  # 设置进度条高度
        self.progress_bar.setTextVisible(False)  # 隐藏进度条上的文字
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: none;
                background-color: transparent;
            }
            QProgressBar::chunk {
                background-color: green;
            }
        """)

        # 将进度条添加到布局底部
        result_area_layout.addWidget(self.progress_bar, alignment=Qt.AlignBottom)


        # 添加下载按钮
        self.download_button = QPushButton("Download Image", self)
        self.download_button.clicked.connect(self.save_image)  # 绑定保存图片的方法
        result_layout.addWidget(self.download_button)

        # 工具栏设置【新增代码】
        self.toolbar = QToolBar("Tools", self)
        self.toolbar.setMovable(True)  # 允许工具栏移动

        # 使用图标而不是文字
        zoom_in_icon = QIcon("F:\2024\10\aidraw\img\选择指针.png")  # 替换为图标文件的路径
        zoom_out_icon = QIcon("F:\2024\10\aidraw\img\灰度反转.png")

        # 使用系统的标准图标
        #zoom_in_icon = self.style().standardIcon(QStyle.SP_ArrowUp)
        #zoom_out_icon = self.style().standardIcon(QStyle.SP_ArrowDown)

        # 创建带图标的 QAction 并设置为可选中状态
        zoom_in_action = self.toolbar.addAction(zoom_in_icon, "Zoom In", self.zoom_in)
        zoom_in_action.setCheckable(True)  # 设置为可选中

        zoom_out_action = self.toolbar.addAction(zoom_out_icon, "Zoom Out", self.zoom_out)
        zoom_out_action.setCheckable(True)  # 设置为可选中

        self.toolbar.addAction(zoom_in_icon,"Zoom In",self.zoom_in)  # 添加缩放操作
        self.toolbar.addAction(zoom_in_icon,"Zoom Out",self.zoom_out)

        # 添加工具栏到 QMainWindow，并允许其停靠在顶部、底部、左右位置
        self.addToolBar(Qt.TopToolBarArea, self.toolbar)
        self.addToolBar(Qt.LeftToolBarArea, self.toolbar)
        self.addToolBar(Qt.RightToolBarArea, self.toolbar)
        self.addToolBar(Qt.BottomToolBarArea, self.toolbar)

        # 设置点击事件以实现单选效果
        zoom_in_action.triggered.connect(lambda: self.toggle_action(zoom_in_action, zoom_out_action))
        zoom_out_action.triggered.connect(lambda: self.toggle_action(zoom_out_action, zoom_in_action))


        # 允许工具栏浮动
        self.toolbar.setFloatable(True)

        # 将主布局添加到中央 widget
        main_layout.addWidget(result_widget)

        # 设置伸缩比例
        main_layout.setStretch(0, 2)  # 左侧 QDockWidget 占 2 个单位
        main_layout.setStretch(1, 8)  # 右侧显示结果区域占 8 个单位

        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)  # 设置 central widget

    def createFloatingInputWidget(self):
        """创建悬浮输入框和按钮的容器"""
        self.text_input = QLineEdit()
        self.text_input.setPlaceholderText("Describe the image you want to generate")
        self.text_input.setFixedHeight(60)
        self.text_input.setFixedWidth(420)  # 调整宽度
        self.text_input.setStyleSheet("""
            QLineEdit {
                border-radius: 25px;
                border: 1px solid #ccc;
                padding-left: 15px;
                font-size: 18px;
                background-color: white;
            }
        """)

        self.generate_button = QPushButton("Go")
        self.generate_button.setFixedSize(70, 60)  # 确保按钮足够大
        self.generate_button.setStyleSheet("""
            QPushButton {
                border-radius: 30px;
                background-color: #007bff;
                color: white;
                font-weight: bold;
                font-size: 18px;
                padding: 10px;  /* 增加填充 */
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
        """)

        # 连接按钮的点击事件
        self.generate_button.clicked.connect(self.generate_image_from_right_input)

        # 创建悬浮输入的布局
        floating_input_layout = QHBoxLayout()
        floating_input_layout.addWidget(self.text_input)
        floating_input_layout.addWidget(self.generate_button)
        floating_input_layout.setContentsMargins(10, 10, 10, 10)
        floating_input_layout.setSpacing(5)  # 减少按钮和输入框之间的间距

        # 创建悬浮输入框的容器
        self.floating_input_widget = QWidget(self.result_area)
        self.floating_input_widget.setLayout(floating_input_layout)
        self.floating_input_widget.setFixedSize(550, 80)  # 调整容器宽度

        # 去除外部的边框
        self.floating_input_widget.setStyleSheet("""
            QWidget {
                border: none;
                background-color: transparent;
            }
        """)

        # 添加阴影效果
        shadow_effect = QGraphicsDropShadowEffect()
        shadow_effect.setBlurRadius(15)
        shadow_effect.setOffset(0, 5)
        shadow_effect.setColor(QColor(0, 0, 0, 100))
        self.floating_input_widget.setGraphicsEffect(shadow_effect)

        # 初始位置
        self.updateFloatingWidgetPosition(None)

    # 在 ImageSegmentationApp 类中新增方法
    def generate_image_from_right_input(self): 
        prompt_text = self.text_input.text()  # 获取右侧输入框的文本
        if not prompt_text:
            print("请输入 prompt 内容以生成图片")
            return

        # 判断是否有图像上传以及是否存在蒙版
        if self.image_path and self.mask is not None:
            print("启用局部重绘模式")
            
            # 检查并确保图像和蒙版的尺寸一致，并转换为符合inpainting要求的格式
            mask_image = Image.fromarray((self.mask * 255).astype(np.uint8)).convert("L")  # 转换为二值图像
            if mask_image.size != self.uploaded_image.size:
                mask_image = mask_image.resize(self.uploaded_image.size)
            
            # 使用 inpainting pipeline 进行局部重绘
            inpainted_image = self.inpainting_pipeline(
                prompt=prompt_text,
                image=self.uploaded_image.convert("RGB"),  # 确保输入图像为 RGB 格式
                mask_image=mask_image,
                num_inference_steps=50,
                guidance_scale=7.5
            ).images[0]

            # 显示局部重绘的结果
            self.display_result_image(inpainted_image)
        else:
            print("启用文生图模式")
            # 调用原有的文生图生成逻辑
            generated_image = self.sd_model(
                prompt=prompt_text,
                width=512,  # 按照设定的尺寸或比例
                height=512
            ).images[0]
            self.display_result_image(generated_image)

    def run_inpainting(self, prompt_text):
        # 生成与上传图像一致的蒙版
        mask_image = Image.fromarray((self.mask * 255).astype(np.uint8)).resize(self.uploaded_image.size).convert("L")
        
        # 调用 inpainting pipeline
        inpainted_image = self.inpainting_pipeline(
            prompt=prompt_text,
            image=self.uploaded_image,
            mask_image=mask_image,
            num_inference_steps=50,
            guidance_scale=7.5
        ).images[0]
        
        # 显示生成的图像
        self.display_result_image(inpainted_image)

    def updateFloatingWidgetPosition(self, event):
        """更新悬浮输入框的位置以保持在结果区域的底部居中"""
        if self.floating_input_widget and self.result_area:
            self.floating_input_widget.move(
                (self.result_area.width() - self.floating_input_widget.width()) // 2,
                self.result_area.height() - self.floating_input_widget.height() - 10
            )
        if event:
            super().resizeEvent(event)


    # 添加一个 toggle_action 方法来控制选中状态
    def toggle_action(self, selected_action, other_action):
        if selected_action.isChecked():
            other_action.setChecked(False)
        else:
            selected_action.setChecked(True)

    # 工具栏动作相关方法【新增代码】
    def zoom_in(self):
        self.scale_factor *= 1.1  # 放大
        self.update_result_area()

    def zoom_out(self):
        self.scale_factor /= 1.1  # 缩小
        self.update_result_area()

    def update_result_area(self):
        if self.result_pixmap:
            new_size = self.result_pixmap.size() * self.scale_factor
            scaled_pixmap = self.result_pixmap.scaled(new_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.result_area.setPixmap(scaled_pixmap)
            self.result_area.setAlignment(Qt.AlignCenter)

    def save_image(self):
        if self.result_pixmap:
            print("保存图片...")  # 调试用，确保 save_image 方法被触发
            file_path, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "PNG Files (*.png);;JPEG Files (*.jpg)")
            if file_path:
                print(f"保存路径: {file_path}")  # 打印保存路径
                self.result_pixmap.save(file_path)
            else:
                print("保存路径无效")  # 如果用户取消保存对话框，打印此信息
        else:
            print("没有可保存的图片")  # 如果 result_pixmap 为空，打印此信息

    def display_result_image(self, image):
        # 确保 image 是 PIL 图像，并将其转换为 RGB 模式
        if isinstance(image, Image.Image):
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            # 将 PIL 图像转换为 QImage 显示在 QLabel 中
            q_image = QImage(image.tobytes("raw", "RGB"), image.width, image.height, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
        elif isinstance(image, QPixmap):
            # 如果已经是 QPixmap 格式，则直接使用
            pixmap = image
        else:
            print("不支持的图像格式")
            return

        # 保存结果图片
        self.result_pixmap = pixmap

        # 根据 result_area 大小调整 pixmap 并显示
        result_size = self.result_area.size()
        scaled_pixmap = pixmap.scaled(result_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.result_area.setPixmap(scaled_pixmap)
        self.result_area.setAlignment(Qt.AlignCenter)
        self.result_area.update()  # 确保图片更新

    
    def create_aspect_ratio_buttons(self):
        # 使用 QGridLayout 创建比例选择布局
        self.aspect_ratio_layout = QGridLayout()
        self.aspect_ratio_layout.setSpacing(10)  # 设置按钮之间的间距

        # 定义比例选项和它们的位置
        ratios = [
            ("1:1", 0, 0), ("4:3", 0, 1), ("3:2", 0, 2),
            ("16:9", 0, 3), ("21:9", 0, 4),
            ("3:4", 1, 0), ("2:3", 1, 1), ("9:16", 1, 2)
        ]

        # 保存按钮的字典
        self.ratio_buttons = {}

        # 创建比例按钮并将其添加到布局中
        for text, row, col in ratios:
            button = QPushButton(text, self)
            button.setCheckable(True)  # 设置按钮为可选中状态
            button.setFixedSize(70, 70)  # 设置按钮的固定大小
            button.setStyleSheet("""
                QPushButton {
                    background-color: #2E2F34;
                    color: white;
                    font-size: 14px;
                    border: 2px solid #42464d;
                    border-radius: 10px;
                }
                QPushButton:checked {
                    background-color: #5a5a5f;
                }
            """)
            button.clicked.connect(lambda _, b=button: self.on_aspect_ratio_selected(b))
            self.aspect_ratio_layout.addWidget(button, row, col)
            self.ratio_buttons[text] = button

        # 默认选择 1:1 比例
        self.ratio_buttons["1:1"].setChecked(True)

        # 创建按钮组，使其在单选状态下互相排斥
        self.aspect_ratio_group = QButtonGroup(self)
        for button in self.ratio_buttons.values():
            self.aspect_ratio_group.addButton(button)

    def on_aspect_ratio_selected(self, selected_button):
        # 遍历所有按钮，取消选中其他按钮
        for button in self.ratio_buttons.values():
            if button != selected_button:
                button.setChecked(False)
        # 更新当前选中的比例
        self.current_aspect_ratio = selected_button.text()


    def load_sam_model(self):
        model_type = "vit_h"
        checkpoint = "F:/2024/10/aidraw/models/sam_vit_h_4b8939.pth"
        sam = sam_model_registry[model_type](checkpoint=checkpoint)
        return sam

    def load_sd_model(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = StableDiffusionPipeline.from_single_file(self.current_model_path, torch_dtype=torch.float16 if device == "cuda" else torch.float32)
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.to(device)
        return pipe

    def load_inpainting_pipeline(self):
        # 加载Inpainting模型
        model_path = r"F:\2024\10\aidraw\models\inpainting_model"  # 使用指定路径的inpainting模型
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
        )
        pipe.to("cuda")  # 使用GPU
        return pipe


    def on_model_selection_change(self):
        # 获取当前选择的模型路径
        model_name = self.model_selector.currentText()
        self.current_model_path = self.model_paths[model_name]
        # 重新加载选择的模型
        self.sd_model = self.load_sd_model()

    def upload_image(self):
        image_path, _ = QFileDialog.getOpenFileName(self, 'Open file', '', "Image files (*.jpg *.png *.jpeg)")
        if image_path:
            self.image_path = image_path
            self.upload_button.hide()
            
            # 加载并保存图像
            self.uploaded_image = Image.open(image_path).convert("RGB")  # 将图像转换为RGB格式
            pixmap = QPixmap(image_path)
            
            # 调整图像大小并显示
            scaled_pixmap = pixmap.scaled(self.upload_area.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.upload_area.setPixmap(scaled_pixmap)
            self.overlay_label.show()


    def reupload_image(self):
        self.upload_area.clear()
        self.upload_button.show()
        self.image_path = None
        self.overlay_label.hide()

    def open_selection_window(self, event):
        if self.image_path:
            self.selection_window = SelectionWindow(self.image_path, self.sam, self)
            self.selection_window.exec_()
            self.point_coords = self.selection_window.get_selected_coords()

    def start_segmentation(self):
        if self.image_path and self.point_coords is not None:
            image = Image.open(self.image_path)
            image = np.array(image)
            predictor = SamPredictor(self.sam)
            predictor.set_image(image)

            input_label = np.array([1])
            masks, _, _ = predictor.predict(point_coords=self.point_coords, point_labels=input_label, multimask_output=False)
            self.mask = masks[0]  # 存储分割蒙版
            masked_image = self.apply_mask(image, self.mask)
            self.display_result_image(masked_image)

    def apply_mask(self, image, mask):
        masked_image = np.zeros_like(image)
        masked_image[mask == 1] = image[mask == 1]
        return masked_image

    def display_result_image(self, image):
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)

        self.result_pixmap = pixmap  # 保存结果图片

        result_size = self.result_area.size()
        scaled_pixmap = pixmap.scaled(result_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.result_area.setPixmap(scaled_pixmap)
        self.result_area.setAlignment(Qt.AlignCenter)
        self.result_area.update()  # 确保图片更新

    def generate_image(self, prompt_text):
        # 使用右侧输入框传递的 prompt_text 作为生成图像的输入
        prompt = prompt_text  # 获取 prompt 文本
        if prompt:
            # 获取选中的比例
            selected_button = self.aspect_ratio_group.checkedButton()
            ratio_text = selected_button.text()

            # 设置生成图像的宽高比例
            if ratio_text == "1:1":
                aspect_ratio = (1, 1)
            elif ratio_text == "16:9":
                aspect_ratio = (16, 9)
            elif ratio_text == "4:3":
                aspect_ratio = (4, 3)
            elif ratio_text == "3:2":
                aspect_ratio = (3, 2)
            elif ratio_text == "2:3":
                aspect_ratio = (2, 3)
            elif ratio_text == "3:4":
                aspect_ratio = (3, 4)
            elif ratio_text == "9:16":
                aspect_ratio = (9, 16)

            # 显示进度条弹窗
            self.progress_dialog = ProgressDialog(self)
            self.progress_dialog.show()

            # 启动图片生成任务
            self.worker = ImageGenerationWorker(self.sd_model, prompt, aspect_ratio)
            self.worker.progress.connect(self.update_progress)
            self.worker.finished.connect(self.on_generation_finished)
            self.progress_bar.setValue(0)
            self.worker.start()


    def update_progress(self, value):
        self.progress_bar.setValue(value)
        self.progress_dialog.set_progress(value)

    def on_generation_finished(self, image):
        image.save("generated_image.png")
        image = image.convert("RGB")  # 确保 PIL 图像为 RGB 模式
        data = image.tobytes("raw", "RGB")  # 将 PIL 图像转换为字节
        q_image = QImage(data, image.width, image.height, QImage.Format_RGB888)  # 转为 QImage
        self.result_pixmap = QPixmap.fromImage(q_image)  # 确保将生成的图片存为 QPixmap
        self.display_generated_image(image)
        self.progress_dialog.close()


    def display_generated_image(self, image):
        image = np.array(image)
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        result_size = self.result_area.size()
        scaled_pixmap = pixmap.scaled(result_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.result_area.setPixmap(scaled_pixmap)
        self.result_area.setAlignment(Qt.AlignCenter)
    # 在 ImageSegmentationApp 类中添加以下方法
    # 修改的 wheelEvent 方法，实现只缩放图片，不影响 result_area 窗口
    def wheelEvent(self, event):
        if self.result_pixmap:
            angle = event.angleDelta().y()  # 获取滚轮的旋转角度
            if angle > 0:
                self.scale_factor *= 1.1  # 放大
            else:
                self.scale_factor /= 1.1  # 缩小

            # 限制缩放比例范围
            self.scale_factor = max(0.1, min(self.scale_factor, 10))

            # 根据缩放比例调整图片大小
            new_size = self.result_pixmap.size() * self.scale_factor

            # 缩放图片，不改变 result_area 窗口的大小
            scaled_pixmap = self.result_pixmap.scaled(new_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            
            # 通过设置 QLabel 的样式，确保图片不会超出显示区域
            self.result_area.setPixmap(scaled_pixmap)
            self.result_area.setAlignment(Qt.AlignCenter)

            # 使用 setFixedSize 来锁定 result_area 的大小，防止窗口随着图片缩放
            self.result_area.setFixedSize(self.result_area.size())

# 进度条弹窗
class ProgressDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Generating Image...')
        self.setGeometry(900, 600, 450, 150)

        self.layout = QVBoxLayout(self)
        self.progress_bar = QProgressBar(self)
        self.layout.addWidget(self.progress_bar)

    def set_progress(self, value):
        self.progress_bar.setValue(value)


# 进度条后台生成任务
class ImageGenerationWorker(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(object)

    def __init__(self, model, prompt, aspect_ratio):
        super().__init__()
        self.model = model
        self.prompt = prompt
        self.aspect_ratio = aspect_ratio  # 新增 aspect_ratio 参数

    def run(self):
        def callback(step: int, timestep: int, latents):
            progress = int(step / 50 * 100)
            self.progress.emit(progress)

        # 根据选择的比例生成图像
        width = 512
        height = int(width / self.aspect_ratio[0] * self.aspect_ratio[1])

        # 保证 width 和 height 都是8的倍数
        width = (width // 8) * 8
        height = (height // 8) * 8
        
        # 生成图像
        image = self.model(self.prompt, num_inference_steps=50, width=width, height=height, callback=callback, callback_steps=1).images[0]
        self.progress.emit(100)
        self.finished.emit(image)

# SelectionWindow 类
class SelectionWindow(QDialog):
    def __init__(self, image_path, sam_model, parent):
        super().__init__(parent)
        self.setWindowTitle('Select Region to Segment')

        # 设置窗口标志，确保有关闭、全屏、最小化按钮
        self.setWindowFlags(Qt.Window | Qt.WindowCloseButtonHint | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint)

        # 加载图片
        self.pixmap = QPixmap(image_path)
        self.image_ratio = self.pixmap.width() / self.pixmap.height()  # 计算图片宽高比

        # 根据图片的初始尺寸设置弹出窗口的大小
        initial_width = 1280  # 可以根据实际情况设置初始宽度
        initial_height = initial_width / self.image_ratio
        self.setGeometry(400, 100, int(initial_width), int(initial_height))

        self.sam = sam_model
        self.parent = parent
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setPixmap(self.pixmap)
        self.image_label.setScaledContents(True)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)

        # 添加滑块到弹窗中
        anti_aliasing_layout = QHBoxLayout()
        anti_aliasing_label = QLabel('Anti-aliasing:', self)
        anti_aliasing_label.setFixedWidth(100)
        self.anti_aliasing_slider = QSlider(Qt.Horizontal)
        self.anti_aliasing_slider.setMinimum(1)
        self.anti_aliasing_slider.setMaximum(15)
        self.anti_aliasing_slider.setValue(5)
        self.anti_aliasing_slider.setFixedWidth(220)
        anti_aliasing_layout.addWidget(anti_aliasing_label)
        anti_aliasing_layout.addWidget(self.anti_aliasing_slider)

        # 羽化滑块
        feathering_layout = QHBoxLayout()
        feathering_label = QLabel('Feathering:', self)
        feathering_label.setFixedWidth(100)
        self.feathering_slider = QSlider(Qt.Horizontal)
        self.feathering_slider.setMinimum(1)
        self.feathering_slider.setMaximum(15)
        self.feathering_slider.setValue(5)
        self.feathering_slider.setFixedWidth(220)
        feathering_layout.addWidget(feathering_label)
        feathering_layout.addWidget(self.feathering_slider)

        # 添加滑块和按钮到水平布局中
        sliders_and_button_layout = QHBoxLayout()
        sliders_and_button_layout.addLayout(anti_aliasing_layout)
        sliders_and_button_layout.addLayout(feathering_layout)

        # Run Segmentation 按钮
        self.run_button = QPushButton('Run Segmentation', self)
        self.run_button.setFixedWidth(270)
        self.run_button.clicked.connect(self.run_segmentation)
        sliders_and_button_layout.addWidget(self.run_button)

        layout.addLayout(sliders_and_button_layout)

        # 确认和取消按钮
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept_selection)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.setLayout(layout)
        self.point_coords = None

        self.image_label.mousePressEvent = self.get_click_position

    def get_click_position(self, event):
        # 获取点击的 x, y 位置，直接使用 event.pos()
        x = event.pos().x()
        y = event.pos().y()

        # 保存点击的坐标
        self.point_coords = np.array([[x, y]])

        # 在点击的地方显示红色标记
        self.display_click_marker(x, y)

    def display_click_marker(self, x, y):
        # 确保 pixmap 存在
        if not self.image_label.pixmap():
            return

        # 重新显示原始的 pixmap 以确保每次点击重置为原始图像
        self.image_label.setPixmap(self.pixmap.copy())

        # 创建一个拷贝以进行绘制
        pixmap = self.image_label.pixmap().copy()
        painter = QPainter(pixmap)
        pen = QPen(Qt.red)
        pen.setWidth(3)
        painter.setPen(pen)

        # 在点击位置绘制红圈，x 和 y 直接来源于点击事件的位置
        painter.drawEllipse(x - 5, y - 5, 10, 10)
        painter.end()

        # 更新 QLabel 显示
        self.image_label.setPixmap(pixmap)

    def get_selected_coords(self):
        return self.point_coords

    def run_segmentation(self):
        self.parent.point_coords = self.point_coords
        self.parent.start_segmentation()

    def accept_selection(self):
        self.accept()

    def resizeEvent(self, event):
        """确保窗口按照图片的宽高比例调整大小"""
        new_width = event.size().width()
        new_height = int(new_width / self.image_ratio)
        self.resize(new_width, new_height)
        super().resizeEvent(event)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ImageSegmentationApp()
    window.show()
    sys.exit(app.exec_())
