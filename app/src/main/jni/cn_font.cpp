// 中文文本渲染实现:stb_truetype + 安卓系统 CJK 字体 + 字形缓存
#include "cn_font.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <unordered_map>

#include <android/log.h>

#define STB_TRUETYPE_IMPLEMENTATION
#include "stb_truetype.h"

#define CN_FONT_TAG "cn_font"

namespace {

    bool g_ready = false;
    std::vector<unsigned char> g_fontdata;
    stbtt_fontinfo g_info;

    // 单个字形的光栅化结果(按 codepoint + 字号缓存)
    struct Glyph {
        int w = 0;
        int h = 0;
        int xoff = 0;   // 相对 pen 的水平偏移(基线坐标系)
        int yoff = 0;   // 相对基线的垂直偏移(向上为负)
        int advance = 0;
        std::vector<unsigned char> alpha; // w*h,8bit 覆盖率
    };

    // key = (codepoint << 32) | 字号(取整)
    std::unordered_map<uint64_t, Glyph> g_cache;

    // UTF-8 解码:从 s[pos] 解出一个码点,len 返回 consumed 字节数
    unsigned int utf8_decode(const std::string& s, size_t pos, int& len) {
        unsigned char c = (unsigned char)s[pos];
        if (c < 0x80) { len = 1; return c; }
        else if ((c >> 5) == 0x6) {
            len = 2;
            if (pos + 1 >= s.size()) { len = 1; return 0xFFFD; }
            return ((unsigned int)(c & 0x1F) << 6) |
                   (unsigned int)(s[pos + 1] & 0x3F);
        }
        else if ((c >> 4) == 0xE) {
            len = 3;
            if (pos + 2 >= s.size()) { len = 1; return 0xFFFD; }
            return ((unsigned int)(c & 0x0F) << 12) |
                   ((unsigned int)(s[pos + 1] & 0x3F) << 6) |
                   (unsigned int)(s[pos + 2] & 0x3F);
        }
        else if ((c >> 3) == 0x1E) {
            len = 4;
            if (pos + 3 >= s.size()) { len = 1; return 0xFFFD; }
            return ((unsigned int)(c & 0x07) << 18) |
                   ((unsigned int)(s[pos + 1] & 0x3F) << 12) |
                   ((unsigned int)(s[pos + 2] & 0x3F) << 6) |
                   (unsigned int)(s[pos + 3] & 0x3F);
        }
        len = 1;
        return 0xFFFD;
    }

    bool try_load(const char* path) {
        FILE* f = fopen(path, "rb");
        if (!f) return false;
        fseek(f, 0, SEEK_END);
        long n = ftell(f);
        fseek(f, 0, SEEK_SET);
        if (n <= 0) { fclose(f); return false; }
        std::vector<unsigned char> buf((size_t)n);
        size_t rd = fread(buf.data(), 1, (size_t)n, f);
        fclose(f);
        if ((long)rd != n) return false;

        int off = stbtt_GetFontOffsetForIndex(buf.data(), 0);
        if (off < 0) return false;

        stbtt_fontinfo info;
        if (!stbtt_InitFont(&info, buf.data(), off)) return false;

        g_fontdata.swap(buf);
        g_info = info;
        __android_log_print(ANDROID_LOG_INFO, CN_FONT_TAG, "loaded font: %s (%ld bytes)", path, n);
        return true;
    }

    // 取字形(带缓存)。font_size 取整作为缓存维度。
    const Glyph& get_glyph(unsigned int cp, int sz) {
        uint64_t key = ((uint64_t)cp << 32) | (uint32_t)sz;
        auto it = g_cache.find(key);
        if (it != g_cache.end()) return it->second;

        Glyph g;
        float scale = stbtt_ScaleForPixelHeight(&g_info, (float)sz);
        int adv = 0, lsb = 0;
        stbtt_GetCodepointHMetrics(&g_info, (int)cp, &adv, &lsb);
        g.advance = (int)(adv * scale + 0.5f);

        int x0, y0, x1, y1;
        stbtt_GetCodepointBitmapBox(&g_info, (int)cp, scale, scale, &x0, &y0, &x1, &y1);
        g.w = x1 - x0;
        g.h = y1 - y0;
        g.xoff = x0;
        g.yoff = y0;
        if (g.w > 0 && g.h > 0) {
            g.alpha.assign((size_t)g.w * g.h, 0);
            stbtt_MakeCodepointBitmap(&g_info, g.alpha.data(), g.w, g.h, g.w,
                                      scale, scale, (int)cp);
        }
        auto res = g_cache.emplace(key, std::move(g));
        return res.first->second;
    }

    void vertical_metrics(int sz, int& ascent_px, int& height_px) {
        float scale = stbtt_ScaleForPixelHeight(&g_info, (float)sz);
        int a = 0, d = 0, lg = 0;
        stbtt_GetFontVerticalLineMetrics(&g_info, &a, &d, &lg);
        ascent_px = (int)(a * scale + 0.5f);
        height_px = (int)((a - d) * scale + 0.5f);
        if (height_px <= 0) height_px = sz;
    }

} // namespace

bool cn_font_init() {
    if (g_ready) return true;

    // 遍历候选系统字体路径,取第一个可加载的(覆盖主流 ROM)
    static const char* candidates[] = {
            "/system/fonts/NotoSansCJK-Regular.ttc",
            "/system/fonts/NotoSansSC-Regular.otf",
            "/system/fonts/NotoSansHans-Regular.otf",
            "/system/fonts/SourceHanSansCN-Regular.otf",
            "/system/fonts/DroidSansFallback.ttf",
            "/system/fonts/NotoSansCJK-Regular.ttf",
            "/system/fonts/HarmonyOS_Sans_SC.ttf",
            "/system/fonts/MiLan.ttf",
            "/system/fonts/MTLmr3m.ttf",
            "/system/fonts/FZLanTingHeiS-L-GB-Regular.ttf",
            "/system/fonts/NotoSerifCJK-Regular.ttc",
    };
    for (const char* p : candidates) {
        if (try_load(p)) { g_ready = true; return true; }
    }
    __android_log_print(ANDROID_LOG_ERROR, CN_FONT_TAG, "no CJK system font found!");
    g_ready = false;
    return false;
}

bool cn_font_ready() {
    return g_ready;
}

cv::Size cn_font_measure(const std::string& utf8, float font_size) {
    if (!g_ready) return cv::Size(0, 0);
    int sz = (int)(font_size + 0.5f);
    if (sz <= 0) sz = 1;
    int ascent = 0, height = 0;
    vertical_metrics(sz, ascent, height);

    int width = 0;
    size_t i = 0;
    while (i < utf8.size()) {
        int len = 1;
        unsigned int cp = utf8_decode(utf8, i, len);
        i += (size_t)len;
        const Glyph& g = get_glyph(cp, sz);
        width += g.advance;
    }
    return cv::Size(width, height);
}

void cn_font_draw(cv::Mat& img, const std::string& utf8, int x, int y,
                  float font_size, const cv::Scalar& color) {
    if (!g_ready) return;
    if (img.type() != CV_8UC3) return; // 仅处理 3 通道 8bit(BGR)

    int sz = (int)(font_size + 0.5f);
    if (sz <= 0) sz = 1;
    int ascent = 0, height = 0;
    vertical_metrics(sz, ascent, height);

    int baseline = y + ascent;
    int pen_x = x;

    size_t i = 0;
    while (i < utf8.size()) {
        int len = 1;
        unsigned int cp = utf8_decode(utf8, i, len);
        i += (size_t)len;
        if (cp == 0) continue;

        const Glyph& g = get_glyph(cp, sz);
        int bx = pen_x + g.xoff;
        int by = baseline + g.yoff;

        for (int r = 0; r < g.h; r++) {
            int iy = by + r;
            if (iy < 0 || iy >= img.rows) continue;
            const unsigned char* arow = g.alpha.data() + (size_t)r * g.w;
            for (int c = 0; c < g.w; c++) {
                int ix = bx + c;
                if (ix < 0 || ix >= img.cols) continue;
                unsigned char a = arow[c];
                if (a == 0) continue;
                cv::Vec3b& px = img.at<cv::Vec3b>(iy, ix);
                float t = a / 255.f;
                float u = 1.f - t;
                px[0] = (unsigned char)(color[0] * t + px[0] * u);
                px[1] = (unsigned char)(color[1] * t + px[1] * u);
                px[2] = (unsigned char)(color[2] * t + px[2] * u);
            }
        }
        pen_x += g.advance;
    }
}
