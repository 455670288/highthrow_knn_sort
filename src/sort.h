#pragma once
#include <vector>
#include <opencv2/opencv.hpp>
#include <queue>
#include <iostream>
#include "kalmanBoxTracker.h"

class Sort {
public:
    /*
    Sort跟踪初始化（获取frame之前初始化）
    1.max_age:跟踪框寿命;
    2.min_hits:最小命中次数;
    3.iou阈值;
    */
    Sort(int max_age = 4, int min_hits = 2, float iou_threshold = 0.1);
    ~Sort();
    

    /*
    每一帧经过背景建模输出的bboxs，传入该函数，进行跟踪框和新检测框的匹配、更新;
     */
    std::vector<cv::Mat> update(const cv::Mat& dets);

private:
    int max_age;
    int min_hits;
    float iou_threshold;
    int frame_count;
    std::vector<KalmanBoxTracker> trackers;



// 计算所有检测框与跟踪框的 IoU
cv::Mat iou_batch(const cv::Mat& bb_test, const cv::Mat& bb_gt);

// 线性匹配算法
std::vector<std::pair<int, int>> hungarian_algorithm(cv::Mat& cost_matrix);

//关联检测和跟踪框
std::tuple<std::vector<std::pair<int, int>>, std::vector<int>, std::vector<int>>
associate_detections_to_trackers(const cv::Mat& detections, const cv::Mat& trackers, float iou_threshold);

};