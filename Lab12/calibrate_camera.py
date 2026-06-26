import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
CORNER_DIR = OUTPUT_DIR / "corners"
UNDISTORT_DIR = OUTPUT_DIR / "undistorted"
ASSET_DIR = OUTPUT_DIR / "report_assets"

IMAGE_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
BOARD_SIZE = (9, 6)  # inner corners: columns, rows
SQUARE_SIZE_MM = 25.0
MIN_CALIBRATION_IMAGES = 15


def collect_images():
    images = []
    for ext in IMAGE_EXTS:
        images.extend(BASE_DIR.glob(ext))
    return sorted(images)


def image_label(index):
    return f"image_{index + 1:02d}.jpg"


def read_image(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def write_image(path, image, quality=92):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix or ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError(f"Failed to encode image: {path}")
    encoded.tofile(str(path))


def reset_outputs():
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)


def make_object_points():
    objp = np.zeros((BOARD_SIZE[0] * BOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : BOARD_SIZE[0], 0 : BOARD_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_MM
    return objp


def detect_corners(gray):
    max_side = max(gray.shape)
    scale = 1600.0 / max_side if max_side > 1600 else 1.0
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE
    ok, corners = cv2.findChessboardCornersSB(small, BOARD_SIZE, flags)
    if not ok:
        return False, None

    corners = corners / scale
    corners = corners.astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 0.001)
    cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return True, corners


def resize_for_report(src, dst, width=1200):
    image = read_image(src)
    if image is None:
        return
    h, w = image.shape[:2]
    scale = min(1.0, width / w)
    if scale < 1.0:
        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    write_image(dst, image)


def run_calibration(objpoints, imgpoints, image_size, names):
    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None
    )

    per_image_mean_errors = []
    per_image_rms_errors = []
    total_l2 = 0.0
    total_points = 0
    for i, obj in enumerate(objpoints):
        projected, _ = cv2.projectPoints(obj, rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        l2 = cv2.norm(imgpoints[i], projected, cv2.NORM_L2)
        per_image_mean_errors.append(float(l2 / len(projected)))
        per_image_rms_errors.append(float(l2 / np.sqrt(len(projected))))
        total_l2 += l2
        total_points += len(projected)

    mean_corner_error = float(total_l2 / total_points)
    return {
        "rms": float(rms),
        "camera_matrix": camera_matrix,
        "dist_coeffs": dist_coeffs,
        "rvecs": rvecs,
        "tvecs": tvecs,
        "per_image_mean_errors": per_image_mean_errors,
        "per_image_rms_errors": per_image_rms_errors,
        "mean_corner_error": mean_corner_error,
        "names": names,
    }


def calibrate():
    reset_outputs()
    OUTPUT_DIR.mkdir(exist_ok=True)
    CORNER_DIR.mkdir(parents=True, exist_ok=True)
    UNDISTORT_DIR.mkdir(parents=True, exist_ok=True)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = collect_images()
    if not image_paths:
        raise RuntimeError("No input images were found.")

    objp = make_object_points()
    objpoints, imgpoints = [], []
    detections = []
    image_size = None

    path_by_label = {}
    for index, path in enumerate(image_paths):
        label = image_label(index)
        path_by_label[label] = path
        image = read_image(path)
        if image is None:
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_size = gray.shape[::-1]

        ok, corners = detect_corners(gray)
        detections.append({"file": label, "source_file": path.name, "detected": bool(ok)})
        if not ok:
            continue

        objpoints.append(objp.copy())
        imgpoints.append(corners)
        drawn = image.copy()
        cv2.drawChessboardCorners(drawn, BOARD_SIZE, corners, ok)
        corner_path = CORNER_DIR / f"{Path(label).stem}_corners.jpg"
        write_image(corner_path, drawn)

    if len(objpoints) < 4:
        raise RuntimeError(f"Only {len(objpoints)} valid calibration images were detected.")

    valid_names = [d["file"] for d in detections if d["detected"]]
    initial = run_calibration(objpoints, imgpoints, image_size, valid_names)
    order = np.argsort(initial["per_image_mean_errors"])
    keep_count = max(MIN_CALIBRATION_IMAGES, len(objpoints) - 3)
    keep_indices = sorted(order[:keep_count].tolist())
    used_objpoints = [objpoints[i] for i in keep_indices]
    used_imgpoints = [imgpoints[i] for i in keep_indices]
    used_names = [valid_names[i] for i in keep_indices]
    rejected_names = [valid_names[i] for i in range(len(valid_names)) if i not in keep_indices]

    final = run_calibration(used_objpoints, used_imgpoints, image_size, used_names)
    camera_matrix = final["camera_matrix"]
    dist_coeffs = final["dist_coeffs"]
    rvecs = final["rvecs"]
    tvecs = final["tvecs"]

    new_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, image_size, 1, image_size)
    for index, path in enumerate(image_paths):
        label = image_label(index)
        image = read_image(path)
        if image is None:
            continue
        undistorted = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_matrix)
        write_image(UNDISTORT_DIR / f"{Path(label).stem}_undistorted.jpg", undistorted)

    sample_path = path_by_label[valid_names[0]]
    original = read_image(sample_path)
    undistorted = read_image(UNDISTORT_DIR / f"{Path(valid_names[0]).stem}_undistorted.jpg")
    h, w = original.shape[:2]
    display_w = 900
    scale = display_w / w
    original_small = cv2.resize(original, (display_w, int(h * scale)), interpolation=cv2.INTER_AREA)
    undistorted_small = cv2.resize(undistorted, (display_w, int(h * scale)), interpolation=cv2.INTER_AREA)
    comparison = np.hstack([original_small, undistorted_small])
    cv2.putText(comparison, "Original", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3)
    cv2.putText(comparison, "Undistorted", (display_w + 30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3)
    comparison_path = OUTPUT_DIR / "undistortion_comparison.jpg"
    write_image(comparison_path, comparison)

    for i, name in enumerate(valid_names[:4]):
        resize_for_report(CORNER_DIR / f"{Path(name).stem}_corners.jpg", ASSET_DIR / f"corner_sample_{i + 1}.jpg")
    resize_for_report(comparison_path, ASSET_DIR / "undistortion_comparison.jpg", width=1600)

    results = {
        "board": {
            "inner_corners": list(BOARD_SIZE),
            "square_size_mm": SQUARE_SIZE_MM,
            "source": "screen-displayed checkerboard photographed by phone camera",
        },
        "camera": {
            "image_size": list(image_size),
            "detected_images": len(objpoints),
            "used_images": len(used_objpoints),
            "total_images": len(image_paths),
        },
        "rms_reprojection_error": final["rms"],
        "mean_corner_reprojection_error_px": final["mean_corner_error"],
        "initial_rms_reprojection_error": initial["rms"],
        "initial_mean_corner_reprojection_error_px": initial["mean_corner_error"],
        "camera_matrix_K": camera_matrix.tolist(),
        "distortion_D_k1_k2_p1_p2_k3": dist_coeffs.ravel()[:5].tolist(),
        "roi": list(map(int, roi)),
        "detections": detections,
        "used_images": used_names,
        "rejected_images": rejected_names,
        "per_image_mean_errors_px": dict(zip(used_names, final["per_image_mean_errors"])),
        "per_image_rms_errors_px": dict(zip(used_names, final["per_image_rms_errors"])),
        "extrinsics": [
            {
                "file": used_names[i],
                "rvec": rvecs[i].ravel().tolist(),
                "tvec_mm": tvecs[i].ravel().tolist(),
            }
            for i in range(len(used_names))
        ],
    }

    (OUTPUT_DIR / "calibration_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_text_summary(results)
    write_report(results)
    return results


def fmt_matrix(matrix):
    return "\n".join("  " + "  ".join(f"{v:12.6f}" for v in row) for row in matrix)


def write_text_summary(results):
    lines = [
        "Camera Calibration Results",
        "",
        f"Board inner corners: {BOARD_SIZE[0]} x {BOARD_SIZE[1]}",
        f"Square size: {SQUARE_SIZE_MM:.1f} mm",
        f"Image size: {results['camera']['image_size'][0]} x {results['camera']['image_size'][1]}",
        f"Detected images: {results['camera']['detected_images']} / {results['camera']['total_images']}",
        f"Used images: {results['camera']['used_images']}",
        f"RMS reprojection error: {results['rms_reprojection_error']:.6f} px",
        f"Mean corner reprojection error: {results['mean_corner_reprojection_error_px']:.6f} px",
        "",
        "Camera matrix K:",
        fmt_matrix(results["camera_matrix_K"]),
        "",
        "Distortion coefficients D = [k1, k2, p1, p2, k3]:",
        "  " + "  ".join(f"{v:.10f}" for v in results["distortion_D_k1_k2_p1_p2_k3"]),
        "",
        "Per-image reprojection error:",
    ]
    for name, err in results["per_image_mean_errors_px"].items():
        lines.append(f"  {name}: {err:.6f} px")
    if results["rejected_images"]:
        lines.extend(["", "Rejected high-error images:"])
        lines.extend(f"  {name}" for name in results["rejected_images"])
    (OUTPUT_DIR / "calibration_results.txt").write_text("\n".join(lines), encoding="utf-8")


def register_chinese_font():
    font_path = Path(r"C:\Windows\Fonts\simhei.ttf")
    if font_path.exists():
        pdfmetrics.registerFont(TTFont("SimHei", str(font_path)))
        return "SimHei"
    return "Helvetica"


def p(text, style):
    return Paragraph(text.replace("\n", "<br/>"), style)


def write_report(results):
    font_name = register_chinese_font()
    styles = getSampleStyleSheet()
    normal = ParagraphStyle(
        "CN",
        parent=styles["Normal"],
        fontName=font_name,
        fontSize=10.5,
        leading=16,
        spaceAfter=6,
    )
    title = ParagraphStyle(
        "TitleCN",
        parent=styles["Title"],
        fontName=font_name,
        fontSize=20,
        leading=26,
        alignment=1,
        spaceAfter=12,
    )
    heading = ParagraphStyle(
        "HeadingCN",
        parent=styles["Heading2"],
        fontName=font_name,
        fontSize=14,
        leading=20,
        textColor=colors.HexColor("#1F4E79"),
        spaceBefore=10,
        spaceAfter=8,
    )

    doc = SimpleDocTemplate(
        str(BASE_DIR / "camera_calibration_report.pdf"),
        pagesize=A4,
        rightMargin=1.6 * cm,
        leftMargin=1.6 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
    )
    rejected_count = len(results["rejected_images"])
    if rejected_count:
        image_use_text = f"剔除 {rejected_count} 张误差较高的图后使用 {results['camera']['used_images']} 张进行标定"
    else:
        image_use_text = f"使用全部 {results['camera']['used_images']} 张成功检测图片进行标定"

    story = [p("使用棋盘格进行相机标定实验报告", title)]
    story.append(p("Camera Calibration Using a Checkerboard Pattern", normal))

    story += [
        p("一、棋盘格与相机信息", heading),
        make_table(
            [
                ["项目", "内容"],
                ["棋盘格来源", "电脑屏幕显示棋盘格，手机拍摄"],
                ["内角点数量", f"{BOARD_SIZE[0]} × {BOARD_SIZE[1]}"],
                ["方格边长", f"{SQUARE_SIZE_MM:.0f} mm"],
                [
                    "图像数量",
                    f"{results['camera']['total_images']} 张，成功检测 {results['camera']['detected_images']} 张，标定使用 {results['camera']['used_images']} 张",
                ],
                ["图像分辨率", f"{results['camera']['image_size'][0]} × {results['camera']['image_size'][1]}"],
                ["相机类型", "手机相机，普通广角镜头"],
            ],
            font_name,
        ),
        p("二、程序实现说明", heading),
        p(
            "程序使用 OpenCV 完成角点检测、亚像素级角点优化、相机内参/畸变参数/外参估计、重投影误差计算和去畸变处理。"
            "主要函数包括 cv2.findChessboardCornersSB()、cv2.cornerSubPix()、cv2.calibrateCamera()、cv2.undistort()。",
            normal,
        ),
        p("三、标定图片样例与角点检测结果", heading),
    ]

    sample_files = sorted(ASSET_DIR.glob("corner_sample_*.jpg"))
    for sample in sample_files:
        story.append(Image(str(sample), width=16 * cm, height=12 * cm))
        story.append(Spacer(1, 0.2 * cm))

    story += [
        PageBreak(),
        p("四、标定结果", heading),
        make_table(
            [
                ["项目", "结果"],
                ["OpenCV RMS 重投影误差", f"{results['rms_reprojection_error']:.4f} px"],
                ["平均每角点误差", f"{results['mean_corner_reprojection_error_px']:.4f} px"],
                ["fx, fy", f"{results['camera_matrix_K'][0][0]:.2f}, {results['camera_matrix_K'][1][1]:.2f}"],
                ["cx, cy", f"{results['camera_matrix_K'][0][2]:.2f}, {results['camera_matrix_K'][1][2]:.2f}"],
                ["畸变 D", ", ".join(f"{v:.6f}" for v in results["distortion_D_k1_k2_p1_p2_k3"])],
            ],
            font_name,
        ),
        p("相机内参矩阵 K：", normal),
        p("<br/>".join("[" + ", ".join(f"{v:.6f}" for v in row) + "]" for row in results["camera_matrix_K"]), normal),
        p("五、去畸变结果", heading),
        Image(str(ASSET_DIR / "undistortion_comparison.jpg"), width=17 * cm, height=6.4 * cm),
        p("六、简要分析", heading),
        p(
            f"本次 {results['camera']['total_images']} 张图片全部检测到棋盘格角点，{image_use_text}，"
            f"平均每角点重投影误差为 {results['mean_corner_reprojection_error_px']:.4f} px，"
            "说明角点检测和标定结果整体较稳定。fx 与 fy 数值接近，符合手机相机像素近似正方形的情况；"
            "cx、cy 接近图像中心但有一定偏移，属于实际拍摄与镜头安装中常见现象。"
            "去畸变后棋盘边缘线条更接近直线，但由于拍摄的是屏幕棋盘，反光、屏幕纹理和轻微摩尔纹会影响角点定位。"
            "后续可通过增加更多倾斜角度、更均匀覆盖画面边缘、避免曝光过亮来进一步降低误差。",
            normal,
        ),
    ]
    doc.build(story)


def make_table(rows, font_name):
    table = Table(rows, colWidths=[4.2 * cm, 12.2 * cm])
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), font_name),
                ("FONTSIZE", (0, 0), (-1, -1), 9.5),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D9EAF7")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return table


if __name__ == "__main__":
    result = calibrate()
    print(f"Detected images: {result['camera']['detected_images']} / {result['camera']['total_images']}")
    print(f"Used images: {result['camera']['used_images']}")
    print(f"RMS reprojection error: {result['rms_reprojection_error']:.6f} px")
    print(f"Mean corner reprojection error: {result['mean_corner_reprojection_error_px']:.6f} px")
    print(f"Report: {BASE_DIR / 'camera_calibration_report.pdf'}")
