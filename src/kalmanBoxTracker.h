#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/videoio.hpp>  // For VideoCapture and VideoWriter
#include <opencv2/highgui.hpp>  // For highgui functions (like imshow)
#include <opencv2/imgproc.hpp>  // For image processing functions


/*
卡尔曼滤波跟踪，在Sort.cc里调用
 */

class KalmanBoxTracker {
public:
    static int count;
    int id;  
    cv::KalmanFilter kf;
    int time_since_update;
    int hits;
    int hit_streak;
    int age;
    cv::Mat org_box;
    bool is_throw;
    // std::vector<cv::Rect2f> tra_history;  // 用于保存预测历史
    

    /*
    为未匹配的检测创建跟踪器
    */
    KalmanBoxTracker(const cv::Mat& bbox);


    /*
    更新已匹配的跟踪器
    */
    void update(const cv::Mat& bbox);
    
    /*
    预测跟踪框
    */
    cv::Mat predict();

    /*
    对新预测的跟踪框与初始框求欧氏距离，根据欧式距离关系判断是否符合抛物运动轨迹,返回预测框和状态
     */
    std::pair<cv::Mat, bool> get_state();

private:
    cv::Mat convert_bbox_to_z(const cv::Mat& bbox);

    cv::Mat convert_x_to_bbox(const cv::Mat& state);
};

