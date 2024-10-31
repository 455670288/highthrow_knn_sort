#include <iostream>
#include <opencv2/opencv.hpp>
// #include <opencv2/features2d.hpp>
// #include <opencv2/calib3d.hpp>
#include <opencv2/videoio.hpp>  // For VideoCapture and VideoWriter
#include <opencv2/highgui.hpp>  // For highgui functions (like imshow)
#include <opencv2/imgproc.hpp> 
#include <chrono>
#include <iomanip>
#include "knnDetector.h"
#include "sort.h"
#include "adjuster.h"

void start_detect() {
    std::string path = "./video/4.mp4";
    cv::VideoCapture capture(path, cv::CAP_FFMPEG);
    if (!capture.isOpened()) {
        std::cerr << "Error: Unable to open video file." << std::endl;
        return;
    }

    // 设置起始帧
    // capture.set(cv::CAP_PROP_POS_FRAMES, 200);
    int fourcc = cv::VideoWriter::fourcc('X', 'V', 'I', 'D');
    double fps = capture.get(cv::CAP_PROP_FPS);
    // int w = int(capture.get(cv::CAP_PROP_FRAME_WIDTH)- 2 * 120);
    // int h = int(capture.get(cv::CAP_PROP_FRAME_HEIGHT)- 2 * 60);
    int w = int(capture.get(cv::CAP_PROP_FRAME_WIDTH));
    int h = int(capture.get(cv::CAP_PROP_FRAME_HEIGHT));
    cv::VideoWriter out("highThrow-4.mp4", fourcc, 25, cv::Size(w, h));

    knnDetector detector(500, 400, 10);

    // 初始化sort
    Sort sort(4, 2, 0.1);    //  max_age:跟踪框寿命  min_hits:最小命中次数.  

    cv::Mat init_frame;
    capture >> init_frame;
    Adjuster adjuster(init_frame, {120,60});

    cv::Mat frame, mask;
    std::vector<cv::Rect> bboxs;
    std::vector<cv::Mat> trackBoxs;
    cv::Mat frame1;  //存放缩小后的图像

    while (true) {
        capture >> frame; // 逐帧读取视频
        if (frame.empty()) {
            break; // 如果没有读取到帧，退出循环
        }

      
        //消除抖动，耗时
        auto start = std::chrono::high_resolution_clock::now();
        frame1 = adjuster.debouncing(frame);
        std::cout << "Debouncing took " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count() << " ms" << std::endl;
        

        // 检测前景
        auto start1 = std::chrono::high_resolution_clock::now();
        detector.detectOneFrame(frame1, mask, bboxs, 20);
        std::cout << "Dete took " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start1).count() << " ms" << std::endl;
        // std::cout << "bboxs_size: " << bboxs.size() << std::endl;       


        /****************************sort算法**********************************/
        auto start2 = std::chrono::high_resolution_clock::now();

        // 将 bboxs 转换为 [x1, y1, x2, y2]
        if(!bboxs.empty()){
            cv::Mat bboxsMat(bboxs.size(), 4, CV_32F);
            for(size_t i = 0; i < bboxs.size(); i++){
                int x1 = bboxs[i].x;
                int y1 = bboxs[i].y;
                int x2 = x1 + bboxs[i].width;
                int y2 = y1 + bboxs[i].height;
                //转换后坐标放入bboxsMat
                bboxsMat.at<float>(i, 0) = x1;
                bboxsMat.at<float>(i, 1) = y1;
                bboxsMat.at<float>(i, 2) = x2;
                bboxsMat.at<float>(i, 3) = y2;
            }
            
            
            trackBoxs = sort.update(bboxsMat);
        }
        else{
            cv::Mat emptyMat(0, 4, CV_32F);
            trackBoxs = sort.update(emptyMat);
        }
        
        std::cout << "Tracking took " 
          << std::fixed << std::setprecision(2) 
          << std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start2).count() 
          << " ms" << std::endl;


        std::cout << "trackBoxs.size: " << trackBoxs.size() << std::endl;
        std::cout << "追踪结束######################################################" << std::endl;
        

        float scale_x = static_cast<float>(w) / 1280; // 确保浮点数除法
        float scale_y = static_cast<float>(h) / 720;  // 确保浮点数除法
        // // 在原始帧上绘制边界框（可选）
        for (const auto& bbox : trackBoxs) {
            cv::rectangle(frame, cv::Point(bbox.at<float>(0) * scale_x, bbox.at<float>(1)* scale_y) ,cv::Point(bbox.at<float>(2) * scale_x, bbox.at<float>(3)* scale_y), cv::Scalar(0, 255, 0), 3);
        }

        // 写入输出视频文件
        if(trackBoxs.size() > 0){
            cv::imwrite("./1.jpg",frame);
        }
        
        out.write(frame);

        // 可选：显示处理结果
        // cv::imshow("Frame", frame);
        // if (cv::waitKey(30) >= 0) break; // 按下任意键退出
    }

    capture.release();
    out.release();
}

int main() {
    start_detect();
    return 0;
}