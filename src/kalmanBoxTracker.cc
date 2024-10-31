#include "kalmanBoxTracker.h"



// 初始化计数器
int KalmanBoxTracker::count = 0;

KalmanBoxTracker::KalmanBoxTracker(const cv::Mat& bbox){
        // 递增静态计数器并为当前对象分配唯一ID
        id = count;
        count++;

        // 定义卡尔曼滤波器的维度
        kf = cv::KalmanFilter(9, 4);  // 状态9维，观测4维
        time_since_update = 0;
        hits = 0;
        hit_streak = 0;
        age = 0;
        is_throw = false;

        // 初始化卡尔曼滤波器参数
        // 状态转移矩阵 F
        kf.transitionMatrix = (cv::Mat_<float>(9, 9) <<     //包含x,y(位置坐标) s(面积) r(宽高比) vx,vy(速度) vs(面积变化速率) ax,ay(加速度)
            1, 0, 0, 0, 1, 0, 0, 0.5, 0,
            0, 1, 0, 0, 0, 1, 0, 0, 0.5,
            0, 0, 1, 0, 0, 0, 1, 0, 0,
            0, 0, 0, 1, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 1, 0, 0, 1, 0,
            0, 0, 0, 0, 0, 1, 0, 0, 1,
            0, 0, 0, 0, 0, 0, 1, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 1, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 1
        );

        // 测量矩阵 H
        kf.measurementMatrix = (cv::Mat_<float>(4, 9) <<   //定义了从状态空间到观测空间的映射关系，这里表明我们只能观测到 x, y, s, r，而无法直接观测到速度和加速度。
            1, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 1, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 1, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 1, 0, 0, 0, 0, 0
        );

        // 设置 过程噪声协方差矩阵Q
        kf.processNoiseCov = cv::Mat::eye(9, 9, CV_32F);

        // 设置 测量噪声协方差矩阵 R
        kf.measurementNoiseCov = cv::Mat::eye(4, 4, CV_32F);

        // 设置初始状态协方差矩阵 P
        kf.errorCovPost = cv::Mat::eye(9, 9, CV_32F);


        //R
        // 1. 对应 self.kf.R[2:, 2:] *= 10.
        for (int i = 2; i < 4; ++i) {
            for (int j = 2; j < 4; ++j) {
               kf.measurementNoiseCov.at<float>(i, j) *= 10.0f;
            }
        }
        
        // P
        // 2. 对应 self.kf.P[4:, 4:] *= 1000.
        for (int i = 4; i < 9; ++i) {
            for (int j = 4; j < 9; ++j) {
               kf.errorCovPost.at<float>(i, j) *= 1000.0f;
            }
        }

        // 3. 对应 self.kf.P *= 10.
        kf.errorCovPost *= 10.0f;  // OpenCV 支持直接矩阵操作
        
        // Q
        // 4. 对应 self.kf.Q[-1, -1] *= 0.01
        kf.processNoiseCov.at<float>(8, 8) *= 0.01f;  // 第 9 行第 9 列（索引从 0 开始）

        // 5. 对应 self.kf.Q[4:, 4:] *= 0.01
        for (int i = 4; i < 9; ++i) {
            for (int j = 4; j < 9; ++j) {
              kf.processNoiseCov.at<float>(i, j) *= 0.01f;
            }
        }

        // 初始状态
        kf.statePost = cv::Mat::zeros(9, 1, CV_32F);
        cv::Mat z = convert_bbox_to_z(bbox); // z 为 (4, 1) 的测量矩阵
        kf.statePost.at<float>(0) = z.at<float>(0); // x
        kf.statePost.at<float>(1) = z.at<float>(1); // y
        kf.statePost.at<float>(2) = z.at<float>(2); // s
        kf.statePost.at<float>(3) = z.at<float>(3); // r

        org_box = bbox;     
}

void KalmanBoxTracker::update(const cv::Mat& bbox){
        time_since_update = 0;
        hits++;
        hit_streak++;
        // tra_history.clear();  // 每次更新后清除历史记录
        // std::cout<< "####################" <<std::endl;
        kf.correct(convert_bbox_to_z(bbox));    
}

cv::Mat KalmanBoxTracker::predict(){
        if ((kf.statePost.at<float>(6) + kf.statePost.at<float>(2)) <= 0) {
            kf.statePost.at<float>(6) = 0;
        }
        /* 检查正常 */
        // std::cout << "kf.statePost.at<float>(0): " << kf.statePost.at<float>(0) << std::endl;
        // std::cout << "kf.statePost.at<float>(1): " << kf.statePost.at<float>(1) << std::endl;
        // std::cout << "kf.statePost.at<float>(2): " << kf.statePost.at<float>(2) << std::endl;
        // std::cout << "kf.statePost.at<float>(3): " << kf.statePost.at<float>(3) << std::endl;
        // std::cout << "kf.statePost.at<float>(4): " << kf.statePost.at<float>(4) << std::endl;
        // std::cout << "kf.statePost.at<float>(5): " << kf.statePost.at<float>(5) << std::endl;
        // std::cout << "kf.statePost.at<float>(6): " << kf.statePost.at<float>(6) << std::endl;
        // std::cout << "kf.statePost.at<float>(7): " << kf.statePost.at<float>(7) << std::endl;
        // std::cout << "kf.statePost.at<float>(8): " << kf.statePost.at<float>(8) << std::endl;


        kf.predict();
        age++;
        if (time_since_update > 0) hit_streak = 0;
        time_since_update++;

        // 将预测的边界框添加到历史记录中
        cv::Mat predicted_bbox = convert_x_to_bbox(kf.statePost);
        // tra_history.push_back(predicted_bbox);

        return predicted_bbox;  
}



std::pair<cv::Mat, bool> KalmanBoxTracker::get_state() {
        cv::Mat bbox = convert_x_to_bbox(kf.statePost);  // 通过Kalman滤波器获得预测框
        float x = (bbox.at<float>(0) + bbox.at<float>(2)) /2 - (org_box.at<float>(0) + org_box.at<float>(2)) /2; // x方向的中心位移
        float y = (bbox.at<float>(1) + bbox.at<float>(3)) /2 - (org_box.at<float>(1) + org_box.at<float>(3)) /2;  // y方向的中心位移
        float distance = std::sqrt(x * x + y * y);  // 计算欧氏距离,   表示预测框中心和原始框中心之间的距离。
        if (distance > 2 * (org_box.at<float>(2) - org_box.at<float>(0) + bbox.at<float>(2) - bbox.at<float>(0)) && 
        distance > (org_box.at<float>(3) - org_box.at<float>(1) + bbox.at<float>(3) - bbox.at<float>(1))) {
            is_throw = true;      // 如果满足条件，标记为"被抛弃"的目标
        }
        return { bbox, is_throw };  // 返回预测框和状态
    }




cv::Mat KalmanBoxTracker::convert_bbox_to_z(const cv::Mat& bbox) {
    /*
      Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
      the aspect ratio
    */
        cv::Mat state = cv::Mat::zeros(4, 1, CV_32F);
        float w = bbox.at<float>(2) - bbox.at<float>(0);
        float h = bbox.at<float>(3) - bbox.at<float>(1);
        float x = bbox.at<float>(0) + w /2;
        float y = bbox.at<float>(1) + h /2; 
        state.at<float>(0) = x;
        state.at<float>(1) = y;
        state.at<float>(2) = w * h ;
        state.at<float>(3) = w / h;
        return state;   // 4*1 
    }



cv::Mat KalmanBoxTracker::convert_x_to_bbox(const cv::Mat& state) {
        // state: 9*1
        float w = std::sqrt(state.at<float>(2) * state.at<float>(3));
        float h = state.at<float>(2) / w;
        

        /*检查正常*/
        // std::cout << "w= " << w << std::endl;
        // std::cout << "h= " << h << std::endl;
        // std::cout << "state.at<float>(0): " << state.at<float>(0) << std::endl;
        // std::cout << "state.at<float>(1): " << state.at<float>(1) << std::endl;
        // std::cout << "state.at<float>(2): " << state.at<float>(2) << std::endl;
        // std::cout << "state.at<float>(3): " << state.at<float>(3) << std::endl;

        cv::Mat bbox(1, 4, CV_32F);
        bbox.at<float>(0) = state.at<float>(0) - w / 2; // x1
        bbox.at<float>(1) = state.at<float>(1) - h / 2; // y1
        bbox.at<float>(2) = state.at<float>(0) + w / 2; // x2
        bbox.at<float>(3) = state.at<float>(1) + h / 2; // y2

        /*检查正常*/
        // std::cout << "bbox.at<float>(0): " << bbox.at<float>(0) << std::endl;
        // std::cout << "bbox.at<float>(1): " << bbox.at<float>(1) << std::endl;
        // std::cout << "bbox.at<float>(2): " << bbox.at<float>(2) << std::endl;
        // std::cout << "bbox.at<float>(3): " << bbox.at<float>(3) << std::endl;

        return bbox;
    }