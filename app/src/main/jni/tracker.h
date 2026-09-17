// 多目标跟踪 + 单目测距 + 接近速度估计
// 基于 IoU 贪心匹配(轻量,适合手机端实时),对每个稳定 track 用历史帧做
// 最小二乘拟合,估算目标距离随时间的变化率,即"接近速度"(正=接近,负=远离)。
#ifndef TRACKER_H
#define TRACKER_H

#include <opencv2/core/core.hpp>
#include <vector>
#include "yolo.h"

struct Track {
    int id;
    com::tencent::yoloncnn::Object obj;  // 当前帧的 rect/label/prob
    float distance;                       // 单目估计距离(米),<=0 表示无效
    float speed;                          // 接近速度(米/秒),正=接近,负=远离
    bool has_speed;                       // 速度是否可靠(需足够历史帧且类别可测距)
    bool approaching;                     // 是否正在接近
};

class Tracker {
public:
    Tracker();

    // is_helmet: 决定"类别 -> 真实高度"映射;img_w/img_h 用于估算相机焦距(像素)
    void update(const std::vector<com::tencent::yoloncnn::Object>& dets,
                double time_sec, int img_w, int img_h, bool is_helmet);

    const std::vector<Track>& tracks() const { return m_tracks; }

private:
    struct Hist {
        double t;    // 时间戳(秒)
        float dist;  // 该帧估计距离(米),<=0 表示无效
    };

    struct State {
        int id;
        com::tencent::yoloncnn::Object obj;
        std::vector<Hist> hist;  // 最近若干帧历史
        int hits;                // 连续命中帧数
        int misses;              // 连续丢失帧数
    };

    std::vector<State> m_states;
    std::vector<Track> m_tracks;
    int m_next_id;

    // 返回该类别的假定真实高度(米);返回 0 表示无法可靠测距
    float real_height(int label, bool is_helmet) const;
};

#endif // TRACKER_H
