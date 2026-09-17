// 中文文本渲染模块:基于 stb_truetype + 安卓系统中文字体
// 解决 opencv-mobile 无 freetype、cv::putText 画不出中文的问题
#ifndef CN_FONT_H
#define CN_FONT_H

#include <opencv2/core/core.hpp>
#include <string>

// 初始化中文字体(加载 /system/fonts 下的系统 CJK 字体)。
// 幂等:多次调用只加载一次。成功返回 true。
bool cn_font_init();

// 字体是否就绪(加载失败时调用方可据此降级处理)
bool cn_font_ready();

// 测量一段 UTF-8 文本按指定字号渲染后的像素尺寸(用于画背景色块)
cv::Size cn_font_measure(const std::string& utf8, float font_size);

// 在 img(CV_8UC3, BGR)上绘制 UTF-8 文本。
// (x, y) 为文本外接框左上角,color 为 BGR 颜色。
void cn_font_draw(cv::Mat& img, const std::string& utf8, int x, int y,
                  float font_size, const cv::Scalar& color);

#endif // CN_FONT_H
