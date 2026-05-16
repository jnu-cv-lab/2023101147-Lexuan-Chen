#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

// 检查是否有GUI支持
bool hasGuiSupport() {
    try {
        // 尝试创建一个小窗口来测试GUI支持
        // 在WSL/无显示器环境下会失败
        const char* display = getenv("DISPLAY");
        return display != nullptr && strlen(display) > 0;
    } catch (...) {
        return false;
    }
}

int main() {
    bool guiAvailable = hasGuiSupport();
    if (!guiAvailable) {
        cout << "[信息] 未检测到GUI环境，将以无窗口模式运行" << endl;
    }

    // ========== 任务1: 读取测试图片 ==========
    // 获取程序所在目录，用于定位图片
    string imagePath = "../images/test_image.jpg";

    // 如果上面的路径失败，尝试当前目录
    Mat image = imread(imagePath);
    if (image.empty()) {
        imagePath = "images/test_image.jpg";
        image = imread(imagePath);
    }

    if (image.empty()) {
        cerr << "错误: 无法加载图片 " << imagePath << endl;
        cerr << "请确保图片路径正确!" << endl;
        return -1;
    }

    cout << "===== 任务1: 成功读取图片 =====" << endl;

    // ========== 任务2: 输出图像基本信息 ==========
    cout << "\n===== 任务2: 图像基本信息 =====" << endl;
    cout << "图像宽度 (Width): " << image.cols << " 像素" << endl;
    cout << "图像高度 (Height): " << image.rows << " 像素" << endl;
    cout << "图像通道数 (Channels): " << image.channels() << endl;
    cout << "图像数据类型 (Data Type): ";

    // 解析数据类型
    switch (image.depth()) {
        case CV_8U:  cout << "CV_8U (8位无符号整数)"; break;
        case CV_8S:  cout << "CV_8S (8位有符号整数)"; break;
        case CV_16U: cout << "CV_16U (16位无符号整数)"; break;
        case CV_16S: cout << "CV_16S (16位有符号整数)"; break;
        case CV_32S: cout << "CV_32S (32位有符号整数)"; break;
        case CV_32F: cout << "CV_32F (32位浮点数)"; break;
        case CV_64F: cout << "CV_64F (64位浮点数)"; break;
        default:     cout << "未知类型";
    }
    cout << endl;
    cout << "图像总像素数: " << image.total() << endl;
    cout << "图像元素大小: " << image.elemSize() << " 字节/像素" << endl;

    // ========== 任务3: 显示原图 ==========
    cout << "\n===== 任务3: 显示原图 =====" << endl;
    if (guiAvailable) {
        namedWindow("原图 (Original)", WINDOW_AUTOSIZE);
        imshow("原图 (Original)", image);
        cout << "显示原图窗口，按任意键继续..." << endl;
        waitKey(0);
        destroyWindow("原图 (Original)");
    } else {
        cout << "[跳过] 无GUI环境，原图已保存到 output/01_original.jpg" << endl;
    }

    // ========== 任务4: 转换为灰度图 ==========
    cout << "\n===== 任务4: 转换为灰度图 =====" << endl;
    Mat grayImage;
    cvtColor(image, grayImage, COLOR_BGR2GRAY);
    cout << "成功转换为灰度图" << endl;
    cout << "灰度图尺寸: " << grayImage.cols << "x" << grayImage.rows << endl;
    cout << "灰度图通道数: " << grayImage.channels() << endl;

    // 显示灰度图
    if (guiAvailable) {
        namedWindow("灰度图 (Grayscale)", WINDOW_AUTOSIZE);
        imshow("灰度图 (Grayscale)", grayImage);
        cout << "显示灰度图窗口，按任意键继续..." << endl;
        waitKey(0);
        destroyWindow("灰度图 (Grayscale)");
    } else {
        cout << "[跳过] 无GUI环境，灰度图将保存到文件" << endl;
    }

    // ========== 任务5: 保存处理结果 ==========
    cout << "\n===== 任务5: 保存处理结果 =====" << endl;
    string outputPath = "../output/gray_image.jpg";
    bool saved = imwrite(outputPath, grayImage);
    if (saved) {
        cout << "灰度图已保存到: " << outputPath << endl;
    } else {
        cerr << "保存失败! 尝试其他路径..." << endl;
        // 尝试备选路径
        outputPath = "output/gray_image.jpg";
        saved = imwrite(outputPath, grayImage);
        if (saved) {
            cout << "灰度图已保存到: " << outputPath << endl;
        } else {
            cerr << "所有路径保存失败!" << endl;
        }
    }

    // ========== 任务6: 简单操作 - 访问像素值和裁剪 ==========
    cout << "\n===== 任务6: 简单像素操作 =====" << endl;

    // 6.1 输出特定位置的像素值 (原图)
    int x = image.cols / 2;  // 图像中心
    int y = image.rows / 2;

    cout << "中心点像素值 (" << x << ", " << y << "): ";
    Vec3b pixel = image.at<Vec3b>(y, x);
    cout << "BGR(" << (int)pixel[0] << ", " << (int)pixel[1] << ", " << (int)pixel[2] << ")" << endl;

    // 输出灰度图中心像素值
    cout << "灰度图中心像素值: " << (int)grayImage.at<uchar>(y, x) << endl;

    // 6.2 裁剪左上角区域
    int cropWidth = image.cols / 3;
    int cropHeight = image.rows / 3;

    // 使用Rect裁剪: Rect(x, y, width, height)
    Rect roi(0, 0, cropWidth, cropHeight);
    Mat croppedImage = image(roi).clone();  // clone()创建独立副本

    cout << "\n裁剪左上角区域:" << endl;
    cout << "裁剪区域: (0, 0) 到 (" << cropWidth << ", " << cropHeight << ")" << endl;
    cout << "裁剪后尺寸: " << croppedImage.cols << "x" << croppedImage.rows << endl;

    // 显示裁剪结果
    if (guiAvailable) {
        namedWindow("裁剪结果 (Cropped)", WINDOW_AUTOSIZE);
        imshow("裁剪结果 (Cropped)", croppedImage);
        cout << "显示裁剪结果，按任意键继续..." << endl;
        waitKey(0);
        destroyWindow("裁剪结果 (Cropped)");
    } else {
        cout << "[跳过] 无GUI环境，裁剪图将保存到文件" << endl;
    }

    // 保存裁剪结果
    string cropOutputPath = "../output/cropped_image.jpg";
    bool cropSaved = imwrite(cropOutputPath, croppedImage);
    if (cropSaved) {
        cout << "裁剪图片已保存到: " << cropOutputPath << endl;
    } else {
        imwrite("output/cropped_image.jpg", croppedImage);
        cout << "裁剪图片已保存到: output/cropped_image.jpg" << endl;
    }

    // 额外：使用at<>直接修改像素值示例
    cout << "\n===== 额外: 像素值修改示例 =====" << endl;
    Mat modifiedImage = image.clone();

    // 在图像中心画一个红色方块 (使用直接像素访问)
    int squareSize = 50;
    for (int i = -squareSize/2; i < squareSize/2; i++) {
        for (int j = -squareSize/2; j < squareSize/2; j++) {
            int px = x + j;
            int py = y + i;
            if (px >= 0 && px < modifiedImage.cols && py >= 0 && py < modifiedImage.rows) {
                modifiedImage.at<Vec3b>(py, px) = Vec3b(0, 0, 255); // BGR红色
            }
        }
    }

    cout << "在图像中心画了红色方块" << endl;
    if (guiAvailable) {
        namedWindow("修改后的图像", WINDOW_AUTOSIZE);
        imshow("修改后的图像", modifiedImage);
    }

    string modifiedPath = "../output/modified_image.jpg";
    imwrite(modifiedPath, modifiedImage);
    cout << "修改后的图片已保存到: " << modifiedPath << endl;

    // 保存原图副本（用于对比）
    imwrite("../output/01_original.jpg", image);
    imwrite("../output/02_grayscale.jpg", grayImage);

    if (guiAvailable) {
        cout << "\n按任意键退出程序..." << endl;
        waitKey(0);
        destroyAllWindows();
    }

    cout << "\n===== 所有任务完成! =====" << endl;
    return 0;
}
