import math
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

# ========== 修复Linux无中文字体报错 ==========
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def create_test_image(width: int = 900, height: int = 700) -> np.ndarray:
    image = np.full((height, width, 3), 255, dtype=np.uint8)

    cv2.rectangle(image, (60, 80), (260, 260), (40, 120, 230), thickness=4)
    cv2.circle(image, (430, 180), 90, (50, 180, 80), thickness=4)
    cv2.line(image, (80, 420), (820, 420), (220, 70, 70), thickness=5)
    cv2.line(image, (300, 80), (300, 620), (180, 60, 200), thickness=5)
    cv2.line(image, (600, 120), (780, 300), (0, 0, 0), thickness=4)
    cv2.line(image, (600, 300), (780, 120), (0, 0, 0), thickness=4)

    cv2.putText(image, "Rectangle", (70, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (40, 120, 230), 2)
    cv2.putText(image, "Circle", (360, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50, 180, 80), 2)
    cv2.putText(image, "Parallel Lines", (70, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (220, 70, 70), 2)
    cv2.putText(image, "Perpendicular Lines", (315, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (180, 60, 200), 2)
    cv2.putText(image, "Perspective Demo", (520, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

    return image


def apply_similarity_transform(image: np.ndarray) -> np.ndarray:
    center = (image.shape[1] / 2, image.shape[0] / 2)
    matrix = cv2.getRotationMatrix2D(center, angle=28, scale=0.82)
    matrix[:, 2] += np.array([40, 30])
    return cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]), borderValue=(255, 255, 255))


def apply_affine_transform(image: np.ndarray) -> np.ndarray:
    src = np.float32([[120, 120], [760, 140], [180, 560]])
    dst = np.float32([[80, 180], [800, 100], [260, 620]])
    matrix = cv2.getAffineTransform(src, dst)
    return cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]), borderValue=(255, 255, 255))


def apply_perspective_transform(image: np.ndarray) -> np.ndarray:
    src = np.float32([[80, 80], [820, 80], [120, 620], [780, 620]])
    dst = np.float32([[160, 140], [740, 70], [60, 650], [840, 560]])
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, matrix, (image.shape[1], image.shape[0]), borderValue=(255, 255, 255))


# ========== 修复1：正确读取你实拍照片 + 删掉后面无效覆盖代码 ==========
def create_document_image(width: int = 900, height: int = 650) -> np.ndarray:
    # 正确完整路径
    img_path = "/home/yiyux/exp1_image_processing/exp1_image_processing_c/image_processing/image_processing/20260423181857_253_100.jpg"
    image = cv2.imread(img_path)
    if image is None:
        raise FileNotFoundError("图片读取失败，请检查路径！")
    return image


# ========== 修复2：换成你真实照片的四个角坐标 ==========
def rectify_document(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # 左上、右上、左下、右下（你自己标的真实坐标）
    src = np.float32([[1140, 24], [90, 194], [1787, 345], [658, 1019]])
    width, height = 595, 842
    dst = np.float32([[0, 0], [width - 1, 0], [0, height - 1], [width - 1, height - 1]])
    matrix = cv2.getPerspectiveTransform(src, dst)
    rectified = cv2.warpPerspective(image, matrix, (width, height), borderValue=(255, 255, 255))
    return rectified, src


def analyze_transforms() -> list[dict[str, str]]:
    return [
        {
            "name": "相似变换",
            "straight_line": "保持为直线",
            "parallel_line": "保持平行",
            "perpendicular_line": "保持垂直",
            "circle": "保持为圆",
        },
        {
            "name": "仿射变换",
            "straight_line": "保持为直线",
            "parallel_line": "保持平行",
            "perpendicular_line": "通常不再垂直",
            "circle": "通常变为椭圆",
        },
        {
            "name": "透视变换",
            "straight_line": "保持为直线",
            "parallel_line": "一般不再平行",
            "perpendicular_line": "通常不再垂直",
            "circle": "通常变为圆锥曲线",
        },
    ]


def save_summary_markdown(path: Path) -> None:
    lines = [
        "# Geometric Transformation Summary",
        "",
        "| Transform | Straight Line | Parallel Line | Perpendicular | Circle |",
        "|---|---|---|---|---|",
    ]
    for row in analyze_transforms():
        lines.append(
            f"| {row['name']} | {row['straight_line']} | {row['parallel_line']} | {row['perpendicular_line']} | {row['circle']} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_results(original: np.ndarray, similarity: np.ndarray, affine: np.ndarray, perspective: np.ndarray, doc_photo: np.ndarray, rectified: np.ndarray) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 11))
    panels = [
        (original, "Original"),
        (similarity, "Similarity"),
        (affine, "Affine"),
        (perspective, "Perspective"),
        (doc_photo, "Document Photo"),
        (rectified, "Rectified"),
    ]

    for ax, (img, title) in zip(axes.flat, panels):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=12)
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "overview.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    # 保留你全部三种变换实验
    original = create_test_image()
    similarity = apply_similarity_transform(original)
    affine = apply_affine_transform(original)
    perspective = apply_perspective_transform(original)

    # 读取你自己的A4实拍图 + 矫正
    doc_photo = create_document_image()
    rectified, corners = rectify_document(doc_photo)

    marked = doc_photo.copy()
    for x, y in corners.astype(int):
        cv2.circle(marked, (x, y), 8, (0, 0, 255), -1)

    # 全部原图输出保留
    cv2.imwrite(str(OUTPUT_DIR / "original_test_image.png"), original)
    cv2.imwrite(str(OUTPUT_DIR / "similarity_transform.png"), similarity)
    cv2.imwrite(str(OUTPUT_DIR / "affine_transform.png"), affine)
    cv2.imwrite(str(OUTPUT_DIR / "perspective_transform.png"), perspective)
    cv2.imwrite(str(OUTPUT_DIR / "document_photo.png"), doc_photo)
    cv2.imwrite(str(OUTPUT_DIR / "document_photo_marked.png"), marked)
    cv2.imwrite(str(OUTPUT_DIR / "document_rectified.png"), rectified)

    plot_results(original, similarity, affine, perspective, doc_photo, rectified)
    save_summary_markdown(OUTPUT_DIR / "analysis_summary.md")

    print("结果已保存到:", OUTPUT_DIR)
    print("1. original_test_image.png")
    print("2. similarity_transform.png")
    print("3. affine_transform.png")
    print("4. perspective_transform.png")
    print("5. document_photo.png")
    print("6. document_photo_marked.png")
    print("7. document_rectified.png")
    print("8. overview.png")
    print("9. analysis_summary.md")


if __name__ == "__main__":
    main()