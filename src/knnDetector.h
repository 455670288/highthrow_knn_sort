#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>


class knnDetector {
public:
    /*
    背景建模初始化（获取frame之前初始化）
    1.history：要对比的历史帧数，值越大越准确；   
    2.dist2Threshold：当前帧与历史帧的距离阈值；值越小前景框越多;  
    3.minArea：前景区域的最小面积，值越小会输出更多小目标框（无论误检与否）;
    */ 
    knnDetector(int history=500, double dist2Threshold=400, int minArea=10);  
 
    ~knnDetector();
    
    /*
    背景建模（输入frame）
    1.frame: 输入的视频帧;  
    2.mask: 存放背景建模输出的前景掩码; 
    3.bboxs: 存放检测框;  
    4.max_boxs: 设置存储box的最大数量(用于减少耗时，默认值20)，防止背景建模误检过多导致后面iou成本矩阵尺寸骤增;
    */
    void detectOneFrame(const cv::Mat& frame, cv::Mat& mask, std::vector<cv::Rect>& bboxs, int max_boxs = 20); 

private:
    cv::Ptr<cv::BackgroundSubtractorKNN> detector;
    int minArea;
    cv::Mat kernel;
};