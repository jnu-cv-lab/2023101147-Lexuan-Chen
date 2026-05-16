import os
from pathlib import Path


OUTPUT_DIR = Path("outputs")
os.environ.setdefault("MPLCONFIGDIR", str((OUTPUT_DIR / ".matplotlib").resolve()))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


RANDOM_STATE = 42


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)


def save_sample_grid(images, labels, filename: str, n: int = 20) -> None:
    rows, cols = 4, 5
    fig, axes = plt.subplots(rows, cols, figsize=(8, 6))
    for ax, image, label in zip(axes.ravel(), images[:n], labels[:n]):
        ax.imshow(image, cmap="gray_r")
        ax.set_title(f"label: {label}")
        ax.axis("off")
    fig.suptitle("Digits sample images")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=160)
    plt.close(fig)


def save_misclassified_grid(images, y_true, y_pred, indices, filename: str, n: int = 12) -> None:
    selected = indices[:n]
    if len(selected) == 0:
        return

    cols = 4
    rows = (len(selected) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(8, 2.2 * rows))
    axes = axes.ravel() if rows > 1 else axes

    for ax in axes:
        ax.axis("off")

    for ax, idx in zip(axes, selected):
        ax.imshow(images[idx], cmap="gray_r")
        ax.set_title(f"true: {y_true[idx]}, pred: {y_pred[idx]}")
        ax.axis("off")

    fig.suptitle("Misclassified test samples")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=160)
    plt.close(fig)


def main() -> None:
    ensure_output_dir()

    digits = load_digits()
    x = digits.data
    y = digits.target
    images = digits.images

    x_train, x_test, y_train, y_test, img_train, img_test = train_test_split(
        x,
        y,
        images,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    models = {
        "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=3)),
        "Naive Bayes": GaussianNB(),
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        ),
        "SVM": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, gamma="scale")),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
    }

    rows = []
    predictions = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        predictions[name] = y_pred
        rows.append(
            {
                "模型": name,
                "测试准确率": accuracy_score(y_test, y_pred),
            }
        )

    results = pd.DataFrame(rows).sort_values("测试准确率", ascending=False)
    results.to_csv(OUTPUT_DIR / "model_accuracy.csv", index=False, encoding="utf-8-sig")

    best_model_name = results.iloc[0]["模型"]
    analysis_model_name = "Naive Bayes"
    analysis_pred = predictions[analysis_model_name]
    cm = confusion_matrix(y_test, analysis_pred, labels=digits.target_names)

    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay(cm, display_labels=digits.target_names).plot(
        ax=ax,
        cmap="Blues",
        colorbar=False,
        values_format="d",
    )
    ax.set_title(f"{analysis_model_name} confusion matrix")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "confusion_matrix_naive_bayes.png", dpi=160)
    plt.close(fig)

    misclassified_indices = (analysis_pred != y_test).nonzero()[0]
    save_sample_grid(images, y, "digits_samples.png")
    save_misclassified_grid(
        img_test,
        y_test,
        analysis_pred,
        misclassified_indices,
        "misclassified_naive_bayes.png",
    )

    with (OUTPUT_DIR / "experiment_summary.txt").open("w", encoding="utf-8") as file:
        file.write("sklearn digits 手写数字分类实验摘要\n")
        file.write("=" * 32 + "\n\n")
        file.write(f"数据集图像数量: {len(images)}\n")
        file.write(f"每张图像大小: {images.shape[1]} x {images.shape[2]}\n")
        file.write(f"展开后特征维度: {x.shape[1]}\n")
        file.write(f"类别标签: {list(map(int, digits.target_names))}\n")
        file.write(f"训练集数量: {len(x_train)}\n")
        file.write(f"测试集数量: {len(x_test)}\n")
        file.write(f"测试集比例: {len(x_test) / len(x):.2%}\n\n")
        file.write("模型测试准确率:\n")
        for _, row in results.iterrows():
            file.write(f"- {row['模型']}: {row['测试准确率']:.4f}\n")
        file.write(f"\n最高准确率模型: {best_model_name}\n")
        file.write(f"最低准确率模型: {results.iloc[-1]['模型']}\n")
        file.write(f"\n错误样本分析模型: {analysis_model_name}\n")
        file.write(f"错误分类样本数: {len(misclassified_indices)}\n")

    print("数据集图像数量:", len(images))
    print("每张图像大小:", f"{images.shape[1]} x {images.shape[2]}")
    print("类别标签:", list(map(int, digits.target_names)))
    print("训练集数量:", len(x_train))
    print("测试集数量:", len(x_test))
    print(results.to_string(index=False, formatters={"测试准确率": "{:.4f}".format}))
    print(f"结果已保存到: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
