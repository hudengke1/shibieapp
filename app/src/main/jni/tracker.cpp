#include "tracker.h"
#include <cmath>
#include <algorithm>

using com::tencent::yoloncnn::Object;

// 水平视场角(度),手机主摄典型值,可按实际镜头微调
static const float HFOV_DEG = 65.0f;
// 计算速度所需的最小历史时间跨度(秒)
static const double MIN_SPAN = 0.25;
// 历史窗口最大时间跨度(秒)与最大帧数
static const double MAX_SPAN = 1.0;
static const size_t MAX_HIST = 25;
// IoU 匹配阈值
static const float IOU_MATCH = 0.30f;
// 连续丢失多少帧后删除 track
static const int MAX_MISSES = 8;

static float rect_iou(const cv::Rect_<float>& a, const cv::Rect_<float>& b)
{
    cv::Rect_<float> inter = a & b;
    float ia = inter.area();
    float ua = a.area() + b.area() - ia;
    return (ua > 0.0f) ? (ia / ua) : 0.0f;
}

Tracker::Tracker() : m_next_id(1) {}

float Tracker::real_height(int label, bool is_helmet) const
{
    if (is_helmet) {
        // 0行人 1轿车 2公交车 3卡车 4摩托车 5自行车 6头盔 7头部
        switch (label) {
            case 0: return 1.70f;   // 行人
            case 1: return 1.50f;   // 轿车
            case 2: return 3.00f;   // 公交车
            case 3: return 2.50f;   // 卡车
            case 4: return 1.30f;   // 摩托车
            case 5: return 1.40f;   // 自行车
            default: return 0.0f;   // 头盔/头部太小,不做测距
        }
    } else {
        // COCO: 0 person,1 bicycle,2 car,3 motorcycle,5 bus,7 truck
        switch (label) {
            case 0: return 1.70f;
            case 1: return 1.40f;
            case 2: return 1.50f;
            case 3: return 1.30f;
            case 5: return 3.00f;
            case 7: return 2.50f;
            default: return 0.0f;   // 其它类别尺寸不定,不做真实测距
        }
    }
}

void Tracker::update(const std::vector<Object>& dets, double time_sec,
                     int img_w, int img_h, bool is_helmet)
{
    (void)img_h;  // 目前仅用宽度估算焦距,保留参数以便后续扩展
    // 估算像素焦距:f = (w/2) / tan(HFOV/2)
    float f_pixel = 0.0f;
    if (img_w > 0) {
        float half = HFOV_DEG * 0.5f * (float)CV_PI / 180.0f;
        f_pixel = (img_w * 0.5f) / std::tan(half);
    }

    const int nd = (int)dets.size();
    const int ns = (int)m_states.size();
    std::vector<int> det_used(nd, 0);
    std::vector<int> st_used(ns, 0);

    // 贪心 IoU 匹配:反复挑当前最大 IoU 的 (state,det) 对
    while (true) {
        float best = IOU_MATCH;
        int bi = -1, bj = -1;
        for (int i = 0; i < ns; i++) {
            if (st_used[i]) continue;
            for (int j = 0; j < nd; j++) {
                if (det_used[j]) continue;
                float iou = rect_iou(m_states[i].obj.rect, dets[j].rect);
                if (iou > best) { best = iou; bi = i; bj = j; }
            }
        }
        if (bi < 0) break;
        st_used[bi] = 1;
        det_used[bj] = 1;

        State& s = m_states[bi];
        s.obj = dets[bj];
        s.hits++;
        s.misses = 0;

        float H = real_height(dets[bj].label, is_helmet);
        float bh = dets[bj].rect.height;
        float dist = (f_pixel > 0.0f && H > 0.0f && bh > 1.0f) ? (f_pixel * H / bh) : 0.0f;
        if (dist > 100.0f) dist = 0.0f;
        Hist h; h.t = time_sec; h.dist = dist;
        s.hist.push_back(h);
    }

    // 未匹配的 state:misses++
    for (int i = 0; i < ns; i++) {
        if (!st_used[i]) m_states[i].misses++;
    }

    // 未匹配的 det:新建 track
    for (int j = 0; j < nd; j++) {
        if (det_used[j]) continue;
        State s;
        s.id = m_next_id++;
        s.obj = dets[j];
        s.hits = 1;
        s.misses = 0;
        float H = real_height(dets[j].label, is_helmet);
        float bh = dets[j].rect.height;
        float dist = (f_pixel > 0.0f && H > 0.0f && bh > 1.0f) ? (f_pixel * H / bh) : 0.0f;
        if (dist > 100.0f) dist = 0.0f;
        Hist h; h.t = time_sec; h.dist = dist;
        s.hist.push_back(h);
        m_states.push_back(s);
    }

    // 删除长期丢失的 track
    for (size_t i = 0; i < m_states.size(); ) {
        if (m_states[i].misses > MAX_MISSES) m_states.erase(m_states.begin() + i);
        else i++;
    }

    // 生成输出 + 计算速度
    m_tracks.clear();
    for (size_t i = 0; i < m_states.size(); i++) {
        State& s = m_states[i];

        // 修剪历史窗口(限制帧数与时间跨度)
        while (s.hist.size() > MAX_HIST) s.hist.erase(s.hist.begin());
        while (s.hist.size() >= 2 && (s.hist.back().t - s.hist.front().t) > MAX_SPAN)
            s.hist.erase(s.hist.begin());

        Track tk;
        tk.id = s.id;
        tk.obj = s.obj;
        tk.distance = 0.0f;
        tk.speed = 0.0f;
        tk.has_speed = false;
        tk.approaching = false;

        // 收集有效测距历史
        std::vector<Hist> valid;
        for (size_t k = 0; k < s.hist.size(); k++)
            if (s.hist[k].dist > 0.0f) valid.push_back(s.hist[k]);

        if (valid.size() >= 3) {
            double span = valid.back().t - valid.front().t;
            if (span >= MIN_SPAN) {
                // 最小二乘拟合 dist = slope*t + b,slope = dD/dt
                double n = (double)valid.size();
                double sum_t = 0, sum_d = 0, sum_tt = 0, sum_td = 0;
                for (size_t k = 0; k < valid.size(); k++) {
                    double t = valid[k].t;
                    double d = valid[k].dist;
                    sum_t += t; sum_d += d; sum_tt += t * t; sum_td += t * d;
                }
                double denom = n * sum_tt - sum_t * sum_t;
                if (std::fabs(denom) > 1e-9) {
                    double slope = (n * sum_td - sum_t * sum_d) / denom; // dD/dt
                    tk.distance = valid.back().dist;
                    tk.speed = (float)(-slope);   // 接近速度 = -dD/dt(距离减小为正)
                    tk.has_speed = true;
                    tk.approaching = (tk.speed > 0.3f);
                }
            }
        }

        // 只输出已确认(命中 >=2 帧)的 track,减少闪现抖动
        if (s.hits >= 2) m_tracks.push_back(tk);
    }
}
