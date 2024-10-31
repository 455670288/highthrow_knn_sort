#pragma once
#include <iostream>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/ocl.hpp>
#include <chrono>

class Adjuster {
public:

    /*
    调整器初始化(需要在获取frame之前初始化)
    传入视频第一帧，作为参考图像，用作与后续帧对齐校正
    */
    Adjuster(const cv::Mat& start_image, std::pair<int, int> edge = {60, 20});
    ~Adjuster();

    /*
    去抖动算法(减少由于前后视频帧抖动导致的误检)
    1.image: 传入视频帧；
    2.ratio: 值越大，用于对齐的特征点越多。默认0.7不变；
    3.reprojThresh: 重投影误差的阈值.默认4.0不变；
    */
    cv::Mat debouncing(const cv::Mat& image, double ratio = 0.7, double reprojThresh = 4.0);

private:
    cv::Mat start_image;
    std::pair<int, int> edge;
    cv::Ptr<cv::ORB> descriptor;
    cv::Ptr<cv::BFMatcher> matcher;
    std::vector<cv::KeyPoint> kps;
    cv::Mat features;

    
    cv::Mat lastImage;         //缓存上一张图像
    cv::Mat lastFeaturesImage; // 缓存上一帧的描述子
    std::vector<cv::KeyPoint> lastKeypoints; // 缓存上一帧的关键点
    bool useCached = false; // 标记是否使用缓存    
    
    bool checkFrameDifference(const cv::Mat& currentFrame, const cv::Mat& lastFrame, double threshold = 0.1);

    void detectAndDescribe(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);


    std::pair<std::vector<cv::DMatch>, cv::Mat> matchKeypoints(const std::vector<cv::KeyPoint>& kpsA, const std::vector<cv::KeyPoint>& kpsB, const cv::Mat& featuresA, const cv::Mat& featuresB, double ratio, double reprojThresh);
};