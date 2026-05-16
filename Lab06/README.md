# 第七课 OpenCV 特征匹配作业

本项目完成作业中的编程部分，主要内容包括：

使用 ORB 检测关键点并提取描述子
使用 Hamming distance 进行 ORB 暴力匹配
使用 RANSAC + Homography 剔除错误匹配
在 `box_in_scene.png` 中完成目标定位
对比 ORB 的 `nfeatures=500/1000/2000` 参数效果
如果当前 OpenCV 支持 SIFT，则完成 SIFT 选做对比实验

## 文件说明

`feature_matching_homework.py`：运行 OpenCV 实验，生成关键点图、匹配图、定位图和统计结果。
`generate_experiment_report.py`：读取实验输出结果，自动生成 Word 实验报告。
`requirements.txt`：记录运行程序所需的 Python 依赖。
`outputs/`：保存实验生成的图片、文本结果和参数对比表。

## 运行方法

```powershell
py -3.7 -m pip install -r requirements.txt
py -3.7 feature_matching_homework.py
```

本机已使用 Python 3.7 和 OpenCV 4.6.0 验证通过。

如需重新生成实验报告，可运行：

```powershell
py -3.7 generate_experiment_report.py
```

报告会保存到桌面：

```text
第七课_ORB_SIFT实验报告.docx
```

## 输出结果

所有实验输出保存在 `outputs/` 文件夹中：
 `box_orb_keypoints.png`
 `box_in_scene_orb_keypoints.png`
 `orb_top50_matches.png`
 `orb_ransac_matches.png`
 `orb_object_location.png`- `orb_results.txt`
 `orb_nfeatures_comparison.csv`
 `sift_optional_results.txt`
 如果 SIFT 可用，还会生成 `sift_ransac_matches.png` 和 `sift_object_location.png`
