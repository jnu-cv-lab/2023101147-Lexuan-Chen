import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


TEMPLATE_IMAGE = Path("box.png")
SCENE_IMAGE = Path("box_in_scene.png")
OUTPUT_DIR = Path("outputs")


@dataclass
class OrbResult:
    nfeatures: int
    template_keypoints: List[cv2.KeyPoint]
    scene_keypoints: List[cv2.KeyPoint]
    template_descriptors: Optional[np.ndarray]
    scene_descriptors: Optional[np.ndarray]
    matches: List[cv2.DMatch]
    homography: Optional[np.ndarray]
    inlier_mask: Optional[np.ndarray]
    projected_corners: Optional[np.ndarray]

    @property
    def total_matches(self) -> int:
        return len(self.matches)

    @property
    def inlier_count(self) -> int:
        if self.inlier_mask is None:
            return 0
        return int(self.inlier_mask.ravel().sum())

    @property
    def inlier_ratio(self) -> float:
        if self.total_matches == 0:
            return 0.0
        return self.inlier_count / self.total_matches

    @property
    def located(self) -> bool:
        return self.homography is not None and self.projected_corners is not None and self.inlier_count >= 4


def load_gray_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return image


def detect_orb(image: np.ndarray, nfeatures: int) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
    orb = cv2.ORB_create(nfeatures=nfeatures)
    return orb.detectAndCompute(image, None)


def match_orb(des1: Optional[np.ndarray], des2: Optional[np.ndarray]) -> List[cv2.DMatch]:
    if des1 is None or des2 is None:
        return []
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    return sorted(matches, key=lambda match: match.distance)


def estimate_homography(
    kp1: List[cv2.KeyPoint],
    kp2: List[cv2.KeyPoint],
    matches: List[cv2.DMatch],
    reprojection_threshold: float = 5.0,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if len(matches) < 4:
        return None, None

    src_points = np.float32([kp1[match.queryIdx].pt for match in matches]).reshape(-1, 1, 2)
    dst_points = np.float32([kp2[match.trainIdx].pt for match in matches]).reshape(-1, 1, 2)
    homography, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, reprojection_threshold)
    return homography, mask


def project_template_corners(template: np.ndarray, homography: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if homography is None:
        return None
    height, width = template.shape[:2]
    corners = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(corners, homography)


def run_orb(template: np.ndarray, scene: np.ndarray, nfeatures: int) -> OrbResult:
    kp1, des1 = detect_orb(template, nfeatures)
    kp2, des2 = detect_orb(scene, nfeatures)
    matches = match_orb(des1, des2)
    homography, mask = estimate_homography(kp1, kp2, matches)
    projected_corners = project_template_corners(template, homography)

    return OrbResult(
        nfeatures=nfeatures,
        template_keypoints=kp1,
        scene_keypoints=kp2,
        template_descriptors=des1,
        scene_descriptors=des2,
        matches=matches,
        homography=homography,
        inlier_mask=mask,
        projected_corners=projected_corners,
    )


def save_keypoints(image: np.ndarray, keypoints: List[cv2.KeyPoint], output_path: Path) -> None:
    vis = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0), flags=cv2.DrawMatchesFlags_DEFAULT)
    cv2.imwrite(str(output_path), vis)


def save_match_image(
    template: np.ndarray,
    kp1: List[cv2.KeyPoint],
    scene: np.ndarray,
    kp2: List[cv2.KeyPoint],
    matches: List[cv2.DMatch],
    output_path: Path,
    max_matches: int = 50,
    mask: Optional[np.ndarray] = None,
) -> None:
    draw_params = {"flags": cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS}
    matches_to_draw = matches[:max_matches]
    if mask is not None:
        draw_params["matchesMask"] = mask.ravel().astype(int).tolist()[: len(matches_to_draw)]

    vis = cv2.drawMatches(template, kp1, scene, kp2, matches_to_draw, None, **draw_params)
    cv2.imwrite(str(output_path), vis)


def save_location_image(scene: np.ndarray, projected_corners: Optional[np.ndarray], output_path: Path) -> None:
    vis = cv2.cvtColor(scene, cv2.COLOR_GRAY2BGR)
    if projected_corners is not None:
        cv2.polylines(vis, [np.int32(projected_corners)], True, (0, 255, 0), 3, cv2.LINE_AA)
    cv2.imwrite(str(output_path), vis)


def descriptor_shape(descriptor: Optional[np.ndarray]) -> str:
    if descriptor is None:
        return "None"
    return str(descriptor.shape)


def format_homography(homography: Optional[np.ndarray]) -> str:
    if homography is None:
        return "Homography estimation failed."
    return np.array2string(homography, precision=6, suppress_small=True)


def write_main_report(result: OrbResult, output_path: Path) -> None:
    lines = [
        "OpenCV ORB Feature Matching Homework Results",
        "",
        "Task 1: ORB keypoints and descriptors",
        f"box.png keypoints: {len(result.template_keypoints)}",
        f"box_in_scene.png keypoints: {len(result.scene_keypoints)}",
        f"box.png descriptor shape: {descriptor_shape(result.template_descriptors)}",
        f"box_in_scene.png descriptor shape: {descriptor_shape(result.scene_descriptors)}",
        "",
        "Task 2: ORB feature matching",
        f"Total matches: {result.total_matches}",
        "Saved top-50 match visualization: outputs/orb_top50_matches.png",
        "",
        "Task 3: RANSAC + Homography",
        f"RANSAC inliers: {result.inlier_count}",
        f"Inlier ratio: {result.inlier_ratio:.4f}",
        "Homography matrix:",
        format_homography(result.homography),
        "",
        "Task 4: Object localization",
        f"Located successfully: {'Yes' if result.located else 'No'}",
        "Saved localization visualization: outputs/orb_object_location.png",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_parameter_csv(results: List[OrbResult], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "nfeatures",
                "template_keypoints",
                "scene_keypoints",
                "matches",
                "ransac_inliers",
                "inlier_ratio",
                "located_successfully",
            ]
        )
        for result in results:
            writer.writerow(
                [
                    result.nfeatures,
                    len(result.template_keypoints),
                    len(result.scene_keypoints),
                    result.total_matches,
                    result.inlier_count,
                    f"{result.inlier_ratio:.4f}",
                    "Yes" if result.located else "No",
                ]
            )


def try_sift(template: np.ndarray, scene: np.ndarray, output_dir: Path) -> str:
    if not hasattr(cv2, "SIFT_create"):
        return "SIFT is not supported by this OpenCV installation."

    start = time.perf_counter()
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(template, None)
    kp2, des2 = sift.detectAndCompute(scene, None)
    if des1 is None or des2 is None:
        return "SIFT descriptors were not found."

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    knn_matches = matcher.knnMatch(des1, des2, k=2)
    good_matches = []
    for pair in knn_matches:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    homography, mask = estimate_homography(kp1, kp2, good_matches)
    corners = project_template_corners(template, homography)
    elapsed = time.perf_counter() - start

    save_match_image(template, kp1, scene, kp2, good_matches, output_dir / "sift_ransac_matches.png", 50, mask)
    save_location_image(scene, corners, output_dir / "sift_object_location.png")

    inliers = int(mask.ravel().sum()) if mask is not None else 0
    ratio = inliers / len(good_matches) if good_matches else 0.0
    located = homography is not None and corners is not None and inliers >= 4
    return (
        "SIFT optional task\n"
        f"SIFT matches after Lowe ratio test: {len(good_matches)}\n"
        f"SIFT RANSAC inliers: {inliers}\n"
        f"SIFT inlier ratio: {ratio:.4f}\n"
        f"SIFT located successfully: {'Yes' if located else 'No'}\n"
        f"SIFT elapsed time: {elapsed:.4f} seconds\n"
    )


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    template = load_gray_image(TEMPLATE_IMAGE)
    scene = load_gray_image(SCENE_IMAGE)

    main_result = run_orb(template, scene, nfeatures=1000)
    save_keypoints(template, main_result.template_keypoints, OUTPUT_DIR / "box_orb_keypoints.png")
    save_keypoints(scene, main_result.scene_keypoints, OUTPUT_DIR / "box_in_scene_orb_keypoints.png")
    save_match_image(
        template,
        main_result.template_keypoints,
        scene,
        main_result.scene_keypoints,
        main_result.matches,
        OUTPUT_DIR / "orb_top50_matches.png",
        50,
    )
    save_match_image(
        template,
        main_result.template_keypoints,
        scene,
        main_result.scene_keypoints,
        main_result.matches,
        OUTPUT_DIR / "orb_ransac_matches.png",
        50,
        main_result.inlier_mask,
    )
    save_location_image(scene, main_result.projected_corners, OUTPUT_DIR / "orb_object_location.png")
    write_main_report(main_result, OUTPUT_DIR / "orb_results.txt")

    parameter_results = [run_orb(template, scene, nfeatures) for nfeatures in (500, 1000, 2000)]
    write_parameter_csv(parameter_results, OUTPUT_DIR / "orb_nfeatures_comparison.csv")

    sift_report = try_sift(template, scene, OUTPUT_DIR)
    (OUTPUT_DIR / "sift_optional_results.txt").write_text(sift_report, encoding="utf-8")

    print((OUTPUT_DIR / "orb_results.txt").read_text(encoding="utf-8"))
    print()
    print((OUTPUT_DIR / "orb_nfeatures_comparison.csv").read_text(encoding="utf-8"))
    print(sift_report)


if __name__ == "__main__":
    main()
