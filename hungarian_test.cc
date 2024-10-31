#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <limits>
#include <queue>
#include <tuple>

/*
测试线性任务匹配算法（匈牙利算法）
 */


std::vector<std::pair<int, int>> 


hungarian_algorithm(cv::Mat& cost_matrix) {

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
            // float delta = std::numeric_limits<float>::infinity(); // 当前最小成本
            // int j1 = -1; // 下一个列索引

            // for (int j = 0; j < m; ++j) {
            //     if (!used[j]) {
            //         float cur = cost_matrix.at<float>(i0, j) - u[i0] - v[j]; // 计算当前成本
            //         if (cur < minv[j]) {
            //             minv[j] = cur;
            //             way[j] = j0; // 记录路径
            //         }
            //         if (minv[j] < delta) {
            //             delta = minv[j];
            //             j1 = j; // 更新下一个列索引
            //         }
                    
            //     }
            // }
            
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
                 break; // 没有可用的列，退出循环
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
            if (i0 == -1) break; // 如果没有更多的行可用，退出
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

int main() {
    // 示例成本矩阵
    cv::Mat cost_matrix = (cv::Mat_<float>(3, 2) << 
        0,9,
        8,0,
        1,2);
    
    auto matches = hungarian_algorithm(cost_matrix);

    // 输出匹配结果
    for (const auto& match : matches) {
        std::cout << "Row " << match.first << " is matched to Column " << match.second << std::endl;
    }

    return 0;
}
