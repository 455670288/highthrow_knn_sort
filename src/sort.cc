#include "sort.h"
#include "kalmanBoxTracker.h"

Sort::Sort(int max_age, int min_hits, float iou_threshold){
    this->max_age = max_age;
    this->min_hits = min_hits;
    this->iou_threshold = iou_threshold;
    this->frame_count = 0;
}

Sort::~Sort(){
    
}

std::vector<cv::Mat> Sort::update(const cv::Mat& dets){
        frame_count++;

        // 1. 预测：初始化存储跟踪框的矩阵
        // cv::Mat trks = cv::Mat::zeros(trackers.size(), 4, CV_32F);
        std::vector<cv::Mat> valid_tracks;
        std::vector<int> to_del;
        std::vector<cv::Mat> ret;
        
        
        for (size_t t = 0; t < trackers.size(); ++t) {
            cv::Mat pos = trackers[t].predict();

            /*检查正常*/
            // std::cout << "pos.row: " << pos.rows <<std::endl;
            // std::cout << "pos.cols: " << pos.cols <<std::endl;
            // std::cout << "pos.at<float>(0,0) " << pos.at<float>(0,0) <<std::endl;
            // std::cout << "pos.at<float>(0,1) " << pos.at<float>(0,1) <<std::endl;
            // std::cout << "pos.at<float>(0,2) " << pos.at<float>(0,2) <<std::endl;
            // std::cout << "pos.at<float>(0,3) " << pos.at<float>(0,3) <<std::endl;

            // 存在 NaN 值
            if (std::isnan(pos.at<float>(0, 0)) || std::isnan(pos.at<float>(0, 1)) || std::isnan(pos.at<float>(0, 2)) || std::isnan(pos.at<float>(0, 3))) {
                to_del.push_back(t);
            }
            else{ //不存在NaN值
                valid_tracks.push_back(pos.clone()); // 将有效跟踪框添加到容器中
            }
        }

        // 删除包含 NaN 值的跟踪器
        for (int i = to_del.size() - 1; i >= 0; --i) {
            trackers.erase(trackers.begin() + to_del[i]); // 删除 trackers 中无效的跟踪器
        }

        to_del.clear(); // 清空 to_del 列表

        // 将有效的跟踪框合并到 trks 中
        cv::Mat trks = cv::Mat::zeros(valid_tracks.size(), 4, CV_32F);
        if (!valid_tracks.empty()) {
            // trks = cv::Mat(valid_tracks.size(), 4, CV_32F);// 假设只需要 4 列
            for (size_t i = 0; i < valid_tracks.size(); ++i) {
                trks.at<float>(i, 0) = valid_tracks[i].at<float>(0,0);
                trks.at<float>(i, 1) = valid_tracks[i].at<float>(0,1);
                trks.at<float>(i, 2) = valid_tracks[i].at<float>(0,2);
                trks.at<float>(i, 3) = valid_tracks[i].at<float>(0,3);
            }
        }

        // 2. 关联检测和跟踪框
        std::tuple<std::vector<std::pair<int, int>>, std::vector<int>, std::vector<int>> result = associate_detections_to_trackers(dets, trks, iou_threshold);
        
        // 使用 std::get 来提取返回的值
        std::vector<std::pair<int, int>> matched = std::get<0>(result);
        std::vector<int> unmatched_dets = std::get<1>(result);
        std::vector<int> unmatched_trks = std::get<2>(result);
        
        // for(int i = 0 ; i < matched.size(); i++){
        //     std::cout << "index: " << i << " " << "first: "<< matched[i].first << "second: "<<matched[i].second << std::endl;
        // }
        // std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!final_matched.size: " << matched.size() << std::endl;

        // 3. 更新已匹配的跟踪器
        for (const auto& m : matched) {
            cv::Mat det(1,4, CV_32F);
            det.at<float>(0) = dets.at<float>(m.first, 0); // x1
            det.at<float>(1) = dets.at<float>(m.first, 1); // y1
            det.at<float>(2) = dets.at<float>(m.first, 2); // x2
            det.at<float>(3) = dets.at<float>(m.first, 3); // y2
            trackers[m.second].update(det);
        }

        // 4. 为未匹配的检测创建新跟踪器
        for (int i : unmatched_dets) {
            cv::Mat det(1,4, CV_32F);
            det.at<float>(0) = dets.at<float>(i, 0); // x1
            det.at<float>(1) = dets.at<float>(i, 1); // y1
            det.at<float>(2) = dets.at<float>(i, 2); // x2
            det.at<float>(3) = dets.at<float>(i, 3); // y2
            // cv::Mat det = dets.row(i);
            KalmanBoxTracker trk(det);
            trackers.push_back(trk);
        }

        // 5. 输出有效跟踪框
        for (int i = trackers.size() - 1; i >= 0; --i) {
            std::pair<cv::Mat, bool> state_result = trackers[i].get_state();
            cv::Mat bbox =  state_result.first;
            bool is_throw = state_result.second;

            if (is_throw && (trackers[i].time_since_update < 1) && 
                (trackers[i].hit_streak >= min_hits || frame_count <= min_hits)) {
                
                ret.push_back(bbox);
            }

            // 删除超时的跟踪器
            if (trackers[i].time_since_update > max_age) {
                trackers.erase(trackers.begin() + i);
            }
        }

        return ret;    
}



cv::Mat Sort::iou_batch(const cv::Mat& bb_test, const cv::Mat& bb_gt) {
    //bb_test:检测框
    
    int n = bb_test.rows;  
    int m = bb_gt.rows;     

    // 创建 IOU 矩阵
    cv::Mat iou_matrix = cv::Mat::zeros(n, m, CV_32F);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            // 获取跟踪框和检测框的坐标
            float x1 = std::max(bb_test.at<float>(i, 0), bb_gt.at<float>(j, 0)); // max(left)
            float y1 = std::max(bb_test.at<float>(i, 1), bb_gt.at<float>(j, 1)); // max(top)
            float x2 = std::min(bb_test.at<float>(i, 2), bb_gt.at<float>(j, 2)); // min(right)
            float y2 = std::min(bb_test.at<float>(i, 3), bb_gt.at<float>(j, 3)); // min(bottom)

            float w = std::max(0.0f, x2 - x1); // 宽度
            float h = std::max(0.0f, y2 - y1); // 高度
            float wh = w * h; // 交集面积

            float area_bb_test = (bb_test.at<float>(i, 2) - bb_test.at<float>(i, 0)) * (bb_test.at<float>(i, 3) - bb_test.at<float>(i, 1));
            float area_bb_gt = (bb_gt.at<float>(j, 2) - bb_gt.at<float>(j, 0)) * (bb_gt.at<float>(j, 3) - bb_gt.at<float>(j, 1));

            // 计算 IOU
            if (area_bb_test + area_bb_gt - wh > 0) {
                iou_matrix.at<float>(i, j) = wh / (area_bb_test + area_bb_gt - wh);
                // std::cout << iou_matrix.at<float>(i, j) << std::endl;
            }
        }
    }

    return iou_matrix;
}



std::vector<std::pair<int, int>> Sort::hungarian_algorithm(cv::Mat& cost_matrix) {

    int n = cost_matrix.rows; // 任务数量
    int m = cost_matrix.cols; // 工作者数量

    // 标记虚拟任务的行和列
    std::vector<bool> is_virtual_row(n, false);
    std::vector<bool> is_virtual_col(m, false);

    // 如果行数不等于列数，添加虚拟任务
    if (n < m) {
        cv::Mat new_cost_matrix = cv::Mat::zeros(m, m, CV_32F);
        cost_matrix.copyTo(new_cost_matrix(cv::Rect(0, 0, m, n))); // 拷贝原有成本矩阵
        for (int i = n; i < m; ++i) {
            is_virtual_row.push_back(true);    // 标记为虚拟任务行
            for (int j = 0; j < m; ++j) {
                new_cost_matrix.at<float>(i, j) = std::numeric_limits<float>::max(); // 设置虚拟任务成本
            }
        }
        cost_matrix = new_cost_matrix;
        n = m; // 更新任务数量
    } else if (m < n) {
        
        cv::Mat new_cost_matrix = cv::Mat::zeros(n, n, CV_32F);
        cost_matrix.copyTo(new_cost_matrix(cv::Rect(0, 0, m, n))); // 拷贝原有成本矩阵
        for (int j = m; j < n; ++j) {
            is_virtual_col.push_back(true);    // 标记为虚拟任务列
            for (int i = 0; i < n; ++i) {
                new_cost_matrix.at<float>(i, j) = std::numeric_limits<float>::max(); // 设置虚拟任务成本
            }
        }
        cost_matrix = new_cost_matrix;
        m = n; // 更新工人数
    }

  
    std::vector<float> u(n, 0), v(m, 0); // 潜在变量
    std::vector<int> p(m, -1); // 列的匹配
    std::vector<int> way(m, -1); // 路径

    for (int i = 0; i < n; ++i) {
        std::vector<float> minv(m, std::numeric_limits<float>::infinity()); // 当前最小值
        std::vector<bool> used(m, false); // 列使用标记
        int j0 = -1, i0 = i; // j0用于标记列，i0是当前行的索引

        while (true) {
            //****************************************************************************** */
            // 使用优先队列找到最小成本
            using Item = std::tuple<float, int>; // (成本, 列索引)
            std::priority_queue<Item, std::vector<Item>, std::greater<Item>> pq;

            for (int j = 0; j < m; ++j) {
                if (!used[j]) {
                    float cur = cost_matrix.at<float>(i0, j) - u[i0] - v[j];
                    if (cur < minv[j]) {
                        minv[j] = cur;
                        way[j] = j0; // 记录路径
                    }
                    pq.push(std::make_tuple(cur, j));
                }
            }

            // 从优先队列中提取最小值
            if (pq.empty()) break; // 没有可用列，退出循环
            float delta = std::get<0>(pq.top()); // 获取成本
            int j1 = std::get<1>(pq.top()); // 获取列索引
            pq.pop();

            //****************************************************************************** */
            // 在使用 j1 之前检查其有效性
            if (j1 == -1) {
                 break; // 没有可用的列(跟踪框)，退出循环
            }


            for (int j = 0; j < m; ++j) { // 更新潜在变量
                if (used[j]) {
                    u[p[j]] += delta; // 已使用列的潜在变量
                    v[j] -= delta; // 当前列的潜在变量
                } else {
                    minv[j] -= delta; // 更新未使用列的最小值
                }
            }

  
            
            used[j1] = true; // 将当前列标记为已使用
            j0 = j1; // 更新当前列为j1
            i0 = p[j0]; // 更新当前行索引
            // std::cout << "j1: " << j1 << std::endl;
            // std::cout << "i0: " << i0 << std::endl;
            if (i0 == -1) break; // 如果没有更多的行可用，退出                 i0 = 0 死循环
        }

        // 更新匹配结果
        for (; j0 != -1; j0 = way[j0]) {
            p[j0] = i; // 记录匹配
        }
    }
    
    // 收集匹配结果
    std::vector<std::pair<int, int>> matches;
    for (int j = 0; j < m; ++j) {
        if (p[j] != -1 && !is_virtual_row[p[j]] && !is_virtual_col[j]) {
            matches.emplace_back(p[j], j); // 存储匹配的行和列索引
        }
    }

    return matches; // 返回匹配结果
}



std::tuple<std::vector<std::pair<int, int>>, std::vector<int>, std::vector<int>>
Sort::associate_detections_to_trackers(const cv::Mat& detections, const cv::Mat& trackers, float iou_threshold) {
    if (trackers.empty()) {
        std::cout << "第一次追踪框为空,不用执行" << std::endl;
        return { {}, std::vector<int>(detections.rows), {} };
    }
    
    cv::Mat iou_matrix = iou_batch(detections, trackers);

    /*检查无误（大概）*/
    std::cout << "iou_matrix.rows: " << iou_matrix.rows << std::endl;
    std::cout << "iou_matrix.cols: " << iou_matrix.cols << std::endl;

    // double minVal, maxVal;
    // cv::Point minLoc, maxLoc;
    // cv::minMaxLoc(iou_matrix, &minVal, &maxVal, &minLoc, &maxLoc);
    // std::cout << "最大值: " << maxVal << std::endl;



    std::vector<std::pair<int, int>> matched_indices;

    if (iou_matrix.rows > 0 && iou_matrix.cols > 0) {
        // 将大于阈值的元素置为1，小于阈值的置为0
        cv::Mat a = (iou_matrix > iou_threshold) /255;
             
    
        // 统计每行和每列的非零元素数量
        cv::Mat row_sum, col_sum;
        cv::reduce(a, row_sum, 1, cv::REDUCE_SUM, CV_32S);  // 按行求和   (iou_matrix.rows, iou_matrix.cols)
        cv::reduce(a, col_sum, 0, cv::REDUCE_SUM, CV_32S);  // 按列求和    (iou_matrix.cols, iou_matrix.rows)

        
        // 如果每行和每列的和最大值为1，则说明是唯一匹配
        double max_row_sum, max_col_sum;
        cv::minMaxLoc(row_sum, nullptr, &max_row_sum);
        cv::minMaxLoc(col_sum, nullptr, &max_col_sum);

        if (max_row_sum == 1 && max_col_sum == 1) {   //唯一匹配
           matched_indices.clear();
        
           // 找出所有满足条件的 (i, j) 对
            for (int i = 0; i < a.rows; ++i) {
              for (int j = 0; j < a.cols; ++j) {
                if (a.at<bool>(i, j)) {
                    matched_indices.emplace_back(i, j);
                }
              }
            }
            // std::cout << "matched_indices_size: " << matched_indices.size() << std::endl;
        }
        else {
            //使用线性任务指派算法
            // std::cout << "线性任务指派" << std::endl;
            cv::Mat reverse_matrix = -iou_matrix;
            matched_indices = hungarian_algorithm(reverse_matrix);
            // std::cout << "linear_assignment_size: " << matched_indices.size() << std::endl;

            // 验证无误
            // cv::Mat cost_matrix = (cv::Mat_<float>(3, 3) << 4, 1, 3,
            //                                            2, 0, 5,
            //                                            3, 2, 2);    
            // std::vector<std::pair<int, int>> matches = hungarian_algorithm(cost_matrix);
            // for (const std::pair<int, int>& match : matches) {
            //     std::cout << "Row " << match.first << " is matched to Column " << match.second << std::endl;
            // }
            
        }
    }
    

    std::vector<int> unmatched_detections, unmatched_trackers;
    // std::vector<int> matched_det_indices, matched_trk_indices;
    
    //查找检测结果是否在 matched_indices
    for (int d = 0; d < detections.rows; ++d) {
        if (std::find_if(matched_indices.begin(), matched_indices.end(), 
                         [d](const std::pair<int, int>& match) { return match.first == d; }) == matched_indices.end()) {
            unmatched_detections.push_back(d);
        }
    }

    for (int t = 0; t < trackers.rows; ++t) {
        if (std::find_if(matched_indices.begin(), matched_indices.end(), 
                         [t](const std::pair<int, int>& match) { return match.second == t; }) == matched_indices.end()) {
            unmatched_trackers.push_back(t);
        }
    }

    // 过滤掉匹配得分低于 IOU 阈值的匹配
    // 这里之所以还需要筛选一遍，是因为线性任务分配算法是根据全局考虑，有可能某一些分数很差的也被匹配了
    std::vector<std::pair<int, int>> final_matches;
    for (const auto& match : matched_indices) {
        if (iou_matrix.at<float>(match.first, match.second) >= iou_threshold) {
            final_matches.push_back(match);
        } else {
            unmatched_detections.push_back(match.first);
            unmatched_trackers.push_back(match.second);
        }
    }

    return { final_matches, unmatched_detections, unmatched_trackers };
}


