#include "knnDetector.h"

knnDetector::knnDetector(int history, double dist2Threshold, int minArea) {
    // 初始化背景减法器
    this->detector = cv::createBackgroundSubtractorKNN(history, dist2Threshold, false);
    this->minArea = minArea;
    this->kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
}

knnDetector::~knnDetector(){
}

void knnDetector::detectOneFrame(const cv::Mat& frame, cv::Mat& mask, std::vector<cv::Rect>& bboxs, int max_boxs) {
        if (frame.empty()) {
            return;
        }

        // 应用背景减法器
        detector->apply(frame, mask);

        // 形态学操作改善前景图像
        cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
        cv::morphologyEx(mask, mask, cv::MORPH_DILATE, kernel);

        // 找到轮廓
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        bboxs.clear();

        for (const auto& contour : contours) {
            if (cv::contourArea(contour) < minArea) {
                continue;
            }

            bboxs.push_back(cv::boundingRect(contour)); //计算其最小外接矩形
        }

        //限制bboxs的数量 ， 防止背景建模误检过多导致iou成本矩阵尺寸骤增
        if(bboxs.size() > max_boxs){
            bboxs.resize(max_boxs);
        }
}