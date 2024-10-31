#include "adjuster.h"

Adjuster::Adjuster(const cv::Mat& start_image, std::pair<int, int> edge) {
        // Resize the start image
        this->start_image = cv::Mat();
        // cv::resize(start_image, this->start_image, cv::Size(start_image.cols, start_image.rows));
        cv::resize(start_image, this->start_image, cv::Size(1280,720));
        // cv::resize(start_image, this->start_image, cv::Size(1920, 1080));
        this->edge = edge;
        this->descriptor = cv::ORB::create(); // ORB feature detector
        this->matcher = cv::BFMatcher::create(cv::NORM_HAMMING, false); // 暴力匹配器， check设为false关闭交叉检查，用于后续knn
        detectAndDescribe(this->start_image, this->kps, this->features);
    }

Adjuster::~Adjuster(){
}



cv::Mat Adjuster::debouncing(const cv::Mat& image, double ratio, double reprojThresh) {  //ratio阈值越大，匹配数量越多
        cv::Mat resized_image;


        // 检查 OpenCL 是否可用
        // if (!cv::ocl::haveOpenCL()) {
        //     std::cout <<  "OpenCL is not available." << std::endl;
        // }
        // cv::resize(image, resized_image, cv::Size(image.cols, image.rows));
        // cv::resize(image, resized_image, cv::Size(),0.5, 0.5); // 根据scale_factor缩小图像
        
        cv::resize(image, resized_image, cv::Size(1280, 720));

        

        std::vector<cv::KeyPoint> kps_image;
        cv::Mat features_image;
        auto start = std::chrono::high_resolution_clock::now();
        // 检查当前帧与缓存帧的差异
        if (useCached && checkFrameDifference(resized_image, lastImage)) {
           // 使用缓存的特征点和描述子
           kps_image = lastKeypoints;
           features_image = lastFeaturesImage;
           std::cout << "Using cached features." << std::endl;
        } else {
           // 使用新的检测和描述
           detectAndDescribe(resized_image, kps_image, features_image);
        
           // 更新缓存
           lastImage = resized_image.clone();          //更新缓存帧
           lastFeaturesImage = features_image.clone(); //更新描述子
           lastKeypoints = kps_image;                  //更新特征点
           useCached = true;
        }

        // detectAndDescribe(resized_image, kps_image, features_image);
        std::cout << "detectAndDescribe took " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count() << " ms" << std::endl;
        

        auto start1 = std::chrono::high_resolution_clock::now();
        auto M = matchKeypoints(kps_image, kps, features_image, features, ratio, reprojThresh);
        std::cout << "matchKeypoints took " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start1).count() << " ms" << std::endl;

        if (M.second.empty()) {
            return cv::Mat();
        }

        std::vector<cv::DMatch> matches = M.first;
        cv::Mat H = M.second;


        cv::Mat result;
        cv::UMat resized_image_umat, result_umat;

        auto start2 = std::chrono::high_resolution_clock::now();
        cv::warpPerspective(resized_image, result, H, cv::Size(resized_image.cols, resized_image.rows));
        // resized_image.copyTo(resized_image_umat); // 将 Mat 转换为 UMat
        // cv::warpPerspective(resized_image_umat, result_umat, H, cv::Size(resized_image.cols, resized_image.rows));
        // result_umat.copyTo(result);

        std::cout << "warpPerspective took " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start2).count() << " ms" << std::endl;
        



        // 还原结果到原始尺寸
        // cv::Mat final_result;
        // auto start3 = std::chrono::high_resolution_clock::now();
        // cv::resize(result, final_result, image.size()*2); // 将结果放大到原始图像尺寸      
        // std::cout << "resize took " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start3).count() << " ms" << std::endl;
        // // cv::imwrite("./test1.jpg", final_result);

        // final_result = final_result(cv::Rect(edge.first, edge.second, image.cols - 2 * edge.first, image.rows - 2 * edge.second));
    

        

        //4320 x 7680
        // cv::imwrite("./test.jpg", result);
        // result = result(cv::Rect(edge.first, edge.second, resized_image.cols - 2 * edge.first, resized_image.rows - 2 * edge.second));


        return result;
    }



bool Adjuster::checkFrameDifference(const cv::Mat& currentFrame, const cv::Mat& lastFrame, double threshold) {
       //计算两张图像绝对差值
       cv::Mat diff;
       cv::absdiff(currentFrame, lastFrame, diff);           
       cv::Mat grayDiff;
       cv::cvtColor(diff, grayDiff, cv::COLOR_BGR2GRAY);
       double nonZeroCount = cv::countNonZero(grayDiff > 255 * threshold); //高threshold，会对轻微变化感知迟钝
       return (nonZeroCount < (currentFrame.total() * 0.1)); //判断变化明显的像素数是否超过总像素的10%
    }


void Adjuster::detectAndDescribe(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) {
        // Convert the image to grayscale
        cv::Mat gray;
        if (image.channels() > 1) {
            cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        } else {
            gray = image;
        }
        

        // Detect and extract features
        descriptor->detect(gray, keypoints);
        descriptor->compute(gray, keypoints, descriptors);
}




std::pair<std::vector<cv::DMatch>, cv::Mat> Adjuster::matchKeypoints(const std::vector<cv::KeyPoint>& kpsA, const std::vector<cv::KeyPoint>& kpsB, const cv::Mat& featuresA, const cv::Mat& featuresB, double ratio, double reprojThresh) {
        std::vector<std::vector<cv::DMatch>> rawMatches;
        if (featuresA.empty() || featuresB.empty()) {
             std::cerr << "Features are empty!" << std::endl;
             return {};
        }

        matcher->knnMatch(featuresA, featuresB, rawMatches, 2);
        
        std::vector<cv::DMatch> goodMatches;

        // Apply Lowe's ratio test
        for (const auto& m : rawMatches) {
            if (m.size() == 2 && m[0].distance < m[1].distance * ratio) {
                goodMatches.push_back(m[0]);
            }
        }
        

        // If there are enough matches, compute homography
        if (goodMatches.size() > 4) {
            std::vector<cv::Point2f> ptsA, ptsB;
            for (const auto& match : goodMatches) {
                ptsA.push_back(kpsA[match.queryIdx].pt);
                ptsB.push_back(kpsB[match.trainIdx].pt);
            }
            
            cv::Mat H = cv::findHomography(ptsA, ptsB, cv::RANSAC, reprojThresh);
            
            return {goodMatches, H};
        }
        // Return empty if not enough matches
        return { {}, cv::Mat() };
    }