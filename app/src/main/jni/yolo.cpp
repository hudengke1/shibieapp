// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "yolo.h"
#include "cn_font.h"
#include "tracker.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "cpu.h"

#include <chrono>

static float fast_exp(float x)
{
    union {
        uint32_t i;
        float f;
    } v{};
    v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
    return v.f;
}

static float sigmoid(float x)
{
    return 1.0f / (1.0f + fast_exp(-x));
}

static float intersection_area(const com::tencent::yoloncnn::Object& a, const com::tencent::yoloncnn::Object& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static float softmax(
        const float* src,
        float* dst,
        int length
)
{
    float alpha = -FLT_MAX;
    for (int c = 0; c < length; c++)
    {
        float score = src[c];
        if (score > alpha)
        {
            alpha = score;
        }
    }

    float denominator = 0;
    float dis_sum = 0;
    for (int i = 0; i < length; ++i)
    {
        dst[i] = expf(src[i] - alpha);
        denominator += dst[i];
    }
    for (int i = 0; i < length; ++i)
    {
        dst[i] /= denominator;
        dis_sum += i * dst[i];
    }
    return dis_sum;
}
static float clamp(
        float val,
        float min = 0.f,
        float max = 1280.f
)
{
    return val > min ? (val < max ? val : max) : min;
}

static void qsort_descent_inplace(std::vector<com::tencent::yoloncnn::Object>& faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

    //     #pragma omp parallel sections
    {
        //         #pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
        //         #pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<com::tencent::yoloncnn::Object>& faceobjects)
{
    if (faceobjects.empty())
        return;

    qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<com::tencent::yoloncnn::Object>& faceobjects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].rect.width * faceobjects[i].rect.height;
    }

    for (int i = 0; i < n; i++)
    {
        const com::tencent::yoloncnn::Object& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const com::tencent::yoloncnn::Object& b = faceobjects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

static void nms_sorted_bboxes_rtmpose(const std::vector<com::tencent::yoloncnn::Object> &objects, std::vector<int> &picked, float nms_threshold, bool agnostic = false)
{
    picked.clear();

    const int n = objects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = objects[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        const com::tencent::yoloncnn::Object &a = objects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const com::tencent::yoloncnn::Object &b = objects[picked[j]];

            if (!agnostic && a.label != b.label)
            {
                continue;
            }

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
            {
                keep = 0;
            }
        }

        if (keep)
        {
            picked.push_back(i);
        }
    }
}

static void generate_grids_and_stride(const int target_w, const int target_h, std::vector<int>& strides, std::vector<GridAndStride>& grid_strides)
{
    for (int i = 0; i < (int)strides.size(); i++)
    {
        int stride = strides[i];
        int num_grid_w = target_w / stride;
        int num_grid_h = target_h / stride;
        for (int g1 = 0; g1 < num_grid_h; g1++)
        {
            for (int g0 = 0; g0 < num_grid_w; g0++)
            {
                GridAndStride gs;
                gs.grid0 = g0;
                gs.grid1 = g1;
                gs.stride = stride;
                grid_strides.push_back(gs);
            }
        }
    }
}
static void generate_proposals(std::vector<GridAndStride> grid_strides,
                               const ncnn::Mat& pred,
                               float prob_threshold,
                               std::vector<com::tencent::yoloncnn::Object>& objects,
                               int num_class)
{
    const int num_points = grid_strides.size();
    // num_class 由调用方(detect)根据模型类型传入:helmet=8,COCO=80
    const int reg_max_1 = 16;

    for (int i = 0; i < num_points; i++)
    {
        const float* scores = pred.row(i) + 4 * reg_max_1;

        // find label with max score
        int label = -1;
        float score = -FLT_MAX;
        for (int k = 0; k < num_class; k++)
        {
            float confidence = scores[k];
            if (confidence > score)
            {
                label = k;
                score = confidence;
            }
        }
        float box_prob = sigmoid(score);
        if (box_prob >= prob_threshold)
        {
            ncnn::Mat bbox_pred(reg_max_1, 4, (void*)pred.row(i));
            {
                ncnn::Layer* softmax = ncnn::create_layer("Softmax");

                ncnn::ParamDict pd;
                pd.set(0, 1); // axis
                pd.set(1, 1);
                softmax->load_param(pd);

                ncnn::Option opt;
                opt.num_threads = 1;
                opt.use_packing_layout = false;

                softmax->create_pipeline(opt);

                softmax->forward_inplace(bbox_pred, opt);

                softmax->destroy_pipeline(opt);

                delete softmax;
            }

            float pred_ltrb[4];
            for (int k = 0; k < 4; k++)
            {
                float dis = 0.f;
                const float* dis_after_sm = bbox_pred.row(k);
                for (int l = 0; l < reg_max_1; l++)
                {
                    dis += l * dis_after_sm[l];
                }

                pred_ltrb[k] = dis * grid_strides[i].stride;
            }

            float pb_cx = (grid_strides[i].grid0 + 0.5f) * grid_strides[i].stride;
            float pb_cy = (grid_strides[i].grid1 + 0.5f) * grid_strides[i].stride;

            float x0 = pb_cx - pred_ltrb[0];
            float y0 = pb_cy - pred_ltrb[1];
            float x1 = pb_cx + pred_ltrb[2];
            float y1 = pb_cy + pred_ltrb[3];

            com::tencent::yoloncnn::Object obj;
            obj.rect.x = x0;
            obj.rect.y = y0;
            obj.rect.width = x1 - x0;
            obj.rect.height = y1 - y0;
            obj.label = label;
            obj.prob = box_prob;

            objects.push_back(obj);
        }
    }
}

static void generate_proposals(int stride,
                               const ncnn::Mat& feat_blob,
                               const float prob_threshold,
                               std::vector<com::tencent::yoloncnn::Object>& objects)
{
    const int reg_max = 16;
    float dst[16];
    const int num_w = feat_blob.w;
    const int num_grid_y = feat_blob.c;
    const int num_grid_x = feat_blob.h;

    const int num_class = num_w - 4 * reg_max;

    for (int i = 0; i < num_grid_y; i++)
    {
        for (int j = 0; j < num_grid_x; j++)
        {

            const float* matat = feat_blob.channel(i).row(j);

            int class_index = 0;
            float class_score = -FLT_MAX;
            for (int c = 0; c < num_class; c++)
            {
                float score = matat[c];
                if (score > class_score)
                {
                    class_index = c;
                    class_score = score;
                }
            }
            if (class_score >= prob_threshold)
            {

                float x0 = j + 0.5f - softmax(matat + num_class, dst, 16);
                float y0 = i + 0.5f - softmax(matat + num_class + 16, dst, 16);
                float x1 = j + 0.5f + softmax(matat + num_class + 2 * 16, dst, 16);
                float y1 = i + 0.5f + softmax(matat + num_class + 3 * 16, dst, 16);

                x0 *= stride;
                y0 *= stride;
                x1 *= stride;
                y1 *= stride;

                com::tencent::yoloncnn::Object obj;
                obj.rect.x = x0;
                obj.rect.y = y0;
                obj.rect.width = x1 - x0;
                obj.rect.height = y1 - y0;
                obj.label = class_index;
                obj.prob = class_score;
                objects.push_back(obj);

            }
        }
    }
}


static void generate_proposals(const ncnn::Mat &anchors,
                               int stride,
                               const ncnn::Mat &in_pad,
                               const ncnn::Mat &feat_blob,
                               float prob_threshold,
                               std::vector<com::tencent::yoloncnn::Object> &objects){
    const int num_grid_x = feat_blob.w;
    const int num_grid_y = feat_blob.h;

    const int num_anchors = anchors.w / 2;

    const int num_class = feat_blob.c / num_anchors - 5;

    const int feat_offset = num_class + 5;

    for (int q = 0; q < num_anchors; q++) {
        const float anchor_w = anchors[q * 2];
        const float anchor_h = anchors[q * 2 + 1];

        for (int i = 0; i < num_grid_y; i++) {
            for (int j = 0; j < num_grid_x; j++) {
                // find class index with max class score
                int class_index = 0;
                float class_score = -FLT_MAX;
                for (int k = 0; k < num_class; k++) {
                    float score = feat_blob.channel(q * feat_offset + 5 + k).row(i)[j];
                    if (score > class_score) {
                        class_index = k;
                        class_score = score;
                    }
                }

                float box_score = feat_blob.channel(q * feat_offset + 4).row(i)[j];

                float confidence = sigmoid(box_score) * sigmoid(class_score);

                if (confidence >= prob_threshold) {
                    // yolov5/models/yolo.py Detect forward
                    // y = x[i].sigmoid()
                    // y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
                    // y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh

                    float dx = sigmoid(feat_blob.channel(q * feat_offset + 0).row(i)[j]);
                    float dy = sigmoid(feat_blob.channel(q * feat_offset + 1).row(i)[j]);
                    float dw = sigmoid(feat_blob.channel(q * feat_offset + 2).row(i)[j]);
                    float dh = sigmoid(feat_blob.channel(q * feat_offset + 3).row(i)[j]);

                    float pb_cx = (dx * 2.f - 0.5f + j) * stride;
                    float pb_cy = (dy * 2.f - 0.5f + i) * stride;

                    float pb_w = pow(dw * 2.f, 2) * anchor_w;
                    float pb_h = pow(dh * 2.f, 2) * anchor_h;

                    float x0 = pb_cx - pb_w * 0.5f;
                    float y0 = pb_cy - pb_h * 0.5f;
                    float x1 = pb_cx + pb_w * 0.5f;
                    float y1 = pb_cy + pb_h * 0.5f;

                    com::tencent::yoloncnn::Object obj;
                    obj.rect.x = x0;
                    obj.rect.y = y0;
                    obj.rect.width = x1 - x0;
                    obj.rect.height = y1 - y0;
                    obj.label = class_index;
                    obj.prob = confidence;

                    objects.push_back(obj);
                }
            }
        }
    }
}


static void generate_proposals_v5(const ncnn::Mat& anchors,
                                  int stride,
                                  const ncnn::Mat& in_pad,
                                  const ncnn::Mat& feat_blob,
                                  float prob_threshold,
                                  std::vector<com::tencent::yoloncnn::Object>& objects)
{
    const int num_grid = feat_blob.h;

    int num_grid_x;
    int num_grid_y;
    if (in_pad.w > in_pad.h)
    {
        num_grid_x = in_pad.w / stride;
        num_grid_y = num_grid / num_grid_x;
    }
    else
    {
        num_grid_y = in_pad.h / stride;
        num_grid_x = num_grid / num_grid_y;
    }

    const int num_class = feat_blob.w - 5;

    const int num_anchors = anchors.w / 2;

    for (int q = 0; q < num_anchors; q++)
    {
        const float anchor_w = anchors[q * 2];
        const float anchor_h = anchors[q * 2 + 1];

        const ncnn::Mat feat = feat_blob.channel(q);

        for (int i = 0; i < num_grid_y; i++)
        {
            for (int j = 0; j < num_grid_x; j++)
            {
                const float* featptr = feat.row(i * num_grid_x + j);

                // find class index with max class score
                int class_index = 0;
                float class_score = -FLT_MAX;
                for (int k = 0; k < num_class; k++)
                {
                    float score = featptr[5 + k];
                    if (score > class_score)
                    {
                        class_index = k;
                        class_score = score;
                    }
                }

                float box_score = featptr[4];

                float confidence = sigmoid(box_score) * sigmoid(class_score);

                if (confidence >= prob_threshold)
                {
                    // yolov5/models/yolo.py Detect forward
                    // y = x[i].sigmoid()
                    // y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
                    // y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh

                    float dx = sigmoid(featptr[0]);
                    float dy = sigmoid(featptr[1]);
                    float dw = sigmoid(featptr[2]);
                    float dh = sigmoid(featptr[3]);

                    float pb_cx = (dx * 2.f - 0.5f + j) * stride;
                    float pb_cy = (dy * 2.f - 0.5f + i) * stride;

                    float pb_w = pow(dw * 2.f, 2) * anchor_w;
                    float pb_h = pow(dh * 2.f, 2) * anchor_h;

                    float x0 = pb_cx - pb_w * 0.5f;
                    float y0 = pb_cy - pb_h * 0.5f;
                    float x1 = pb_cx + pb_w * 0.5f;
                    float y1 = pb_cy + pb_h * 0.5f;

                    com::tencent::yoloncnn::Object obj;
                    obj.rect.x = x0;
                    obj.rect.y = y0;
                    obj.rect.width = x1 - x0;
                    obj.rect.height = y1 - y0;
                    obj.label = class_index;
                    obj.prob = confidence;

                    objects.push_back(obj);
                }
            }
        }
    }
}


static void generate_proposals(const ncnn::Mat &feat_blob, int stride, float prob_threshold, std::vector<com::tencent::yoloncnn::Object> &objects)
{
    const int num_w = feat_blob.w;
    const int num_grid_y = feat_blob.c;
    const int num_grid_x = feat_blob.h;

    // const float prob_threshold_prev = -log((1 - prob_threshold) / (prob_threshold + 1e-5));
    // std::cout << num_grid_y << " " << num_grid_x << " " << num_w << std::endl;
    for (int i = 0; i < num_grid_y; i++)
    {
        for (int j = 0; j < num_grid_x; j++)
        {
            const float *matat = feat_blob.channel(i).row(j);
            float score = matat[0];
            if (score >= prob_threshold)
            {
                score = sigmoid(score);
                // std::cout << j << " " << i << " " << score << " " << matat[1] << " " << matat[2] << " " << matat[3] << " " << matat[4] << std::endl;
                float x0 = j * stride - matat[1];
                float y0 = i * stride - matat[2];
                float x1 = j * stride + matat[3];
                float y1 = i * stride + matat[4];

                com::tencent::yoloncnn::Object obj;
                obj.rect.x = x0;
                obj.rect.y = y0;
                obj.rect.width = x1 - x0;
                obj.rect.height = y1 - y0;
                obj.label = 0;
                obj.prob = score;
                objects.push_back(obj);
            }
        }
    }
}

Yolo::Yolo()
{
    blob_pool_allocator.set_size_compare_ratio(0.f);
    workspace_pool_allocator.set_size_compare_ratio(0.f);
}

int Yolo::load(const char *modeltype, int _target_size, const float* _mean_vals, const float* _norm_vals, bool use_gpu) {
    yolo.clear();
    blob_pool_allocator.clear();
    workspace_pool_allocator.clear();

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    yolo.opt = ncnn::Option();

#if NCNN_VULKAN
    yolo.opt.use_vulkan_compute = use_gpu;
#endif

    yolo.opt.num_threads = ncnn::get_big_cpu_count();
    yolo.opt.blob_allocator = &blob_pool_allocator;
    yolo.opt.workspace_allocator = &workspace_pool_allocator;

    char parampath[256];
    char modelpath[256];
    if (strcmp("rtmdet-nano",modeltype) == 0){
        sprintf(modelpath, "%s.bin", modeltype);
    } else if (strcmp("helmet",modeltype) == 0){
        // 本项目自定义头盔/车辆/行人模型:helmet.param + helmet.bin
        sprintf(parampath, "helmet.param");
        sprintf(modelpath, "helmet.bin");
    } else{
        sprintf(parampath, "yolo%s.param", modeltype);
        sprintf(modelpath, "yolo%s.bin", modeltype);
    }
    yolo.load_param(parampath);
    yolo.load_model(modelpath);

    // 根据模型类型确定类别数:helmet=8,其余(COCO)=80
    num_class = (strcmp("helmet", modeltype) == 0) ? 8 : 80;

    target_size = _target_size;
    norm_vals[0] = _norm_vals[0];
    norm_vals[1] = _norm_vals[1];
    norm_vals[2] = _norm_vals[2];

    return 0;
}

DEFINE_LAYER_CREATOR(YoloV5Focus)
int Yolo::load(AAssetManager* mgr, const char* modeltype, int _target_size, const float* _mean_vals, const float* _norm_vals, bool use_gpu)
{
    yolo.clear();
    blob_pool_allocator.clear();
    workspace_pool_allocator.clear();

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    yolo.opt = ncnn::Option();

#if NCNN_VULKAN
    yolo.opt.use_vulkan_compute = use_gpu;
#endif

    yolo.opt.num_threads = ncnn::get_big_cpu_count();
    yolo.opt.blob_allocator = &blob_pool_allocator;
    yolo.opt.workspace_allocator = &workspace_pool_allocator;

    char parampath[256];
    char modelpath[256];
    if (strcmp("rtmdet-nano",modeltype) == 0){
        sprintf(modelpath, "%s.bin", modeltype);
    } else if (strcmp("helmet",modeltype) == 0){
        // 本项目自定义头盔/车辆/行人模型:helmet.param + helmet.bin
        sprintf(parampath, "helmet.param");
        sprintf(modelpath, "helmet.bin");
    } else{
        sprintf(parampath, "yolo%s.param", modeltype);
        sprintf(modelpath, "yolo%s.bin", modeltype);
    }

    __android_log_print(ANDROID_LOG_ERROR, "yolo", "loadModelName= %s", modelpath);

    if (strcmp("v5s",modeltype))
        yolo.register_custom_layer("YoloV5Focus", YoloV5Focus_layer_creator);
    yolo.load_param(mgr, parampath);
    yolo.load_model(mgr, modelpath);
    // 根据模型类型确定类别数:helmet=8,其余(COCO)=80
    num_class = (strcmp("helmet", modeltype) == 0) ? 8 : 80;
    __android_log_print(ANDROID_LOG_ERROR, "yolo", "_target_size %d", _target_size);
    target_size = _target_size;
    mean_vals[0] = _mean_vals[0];
    mean_vals[1] = _mean_vals[1];
    mean_vals[2] = _mean_vals[2];
    norm_vals[0] = _norm_vals[0];
    norm_vals[1] = _norm_vals[1];
    norm_vals[2] = _norm_vals[2];
    __android_log_print(ANDROID_LOG_ERROR, "yolo", "load end");
    return 0;
}

int Yolo::detect(const cv::Mat& rgb, std::vector<com::tencent::yoloncnn::Object>& objects, float prob_threshold, float nms_threshold)
{
    int width = rgb.cols;
    int height = rgb.rows;

    // pad to multiple of 32
    int w = width;
    int h = height;
    float scale = 1.f;
    if (w > h)
    {
        scale = (float)target_size / w;
        w = target_size;
        h = h * scale;
    }
    else
    {
        scale = (float)target_size / h;
        h = target_size;
        w = w * scale;
    }

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_RGB2BGR, width, height, w, h);

    // pad to target_size rectangle
    int wpad = (w + 31) / 32 * 32 - w;
    int hpad = (h + 31) / 32 * 32 - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 0.f);

    in_pad.substract_mean_normalize(0, norm_vals);

    ncnn::Extractor ex = yolo.create_extractor();

    ex.input("images", in_pad);

    std::vector<com::tencent::yoloncnn::Object> proposals;
    
    ncnn::Mat out;
    ex.extract("output", out);

    std::vector<int> strides = {8, 16, 32}; // might have stride=64
    std::vector<GridAndStride> grid_strides;
    generate_grids_and_stride(in_pad.w, in_pad.h, strides, grid_strides);
    generate_proposals(grid_strides, out, prob_threshold, proposals, num_class);

    // sort all proposals by score from highest to lowest
    qsort_descent_inplace(proposals);

    // apply nms with nms_threshold
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold);

    int count = picked.size();

    objects.resize(count);
    for (int i = 0; i < count; i++)
    {
        objects[i] = proposals[picked[i]];

        // adjust offset to original unpadded
        float x0 = (objects[i].rect.x - (wpad / 2)) / scale;
        float y0 = (objects[i].rect.y - (hpad / 2)) / scale;
        float x1 = (objects[i].rect.x + objects[i].rect.width - (wpad / 2)) / scale;
        float y1 = (objects[i].rect.y + objects[i].rect.height - (hpad / 2)) / scale;

        // clip
        x0 = std::max(std::min(x0, (float)(width - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(height - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(width - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(height - 1)), 0.f);

        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width = x1 - x0;
        objects[i].rect.height = y1 - y0;
    }

    // sort objects by area
    struct
    {
        bool operator()(const com::tencent::yoloncnn::Object& a, const com::tencent::yoloncnn::Object& b) const
        {
            return a.rect.area() > b.rect.area();
        }
    } objects_area_greater;
    std::sort(objects.begin(), objects.end(), objects_area_greater);

    return 0;
}

int Yolo::detect_v11(const cv::Mat& rgb, std::vector<com::tencent::yoloncnn::Object>& objects, float prob_threshold, float nms_threshold)
{
    int width = rgb.cols;
    int height = rgb.rows;

    // pad to multiple of 32
    int w = width;
    int h = height;
    float scale = 1.f;
    if (w > h)
    {
        scale = (float)target_size / w;
        w = target_size;
        h = h * scale;
    }
    else
    {
        scale = (float)target_size / h;
        h = target_size;
        w = w * scale;
    }
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_BGR2RGB, width, height, w, h);
    // pad to target_size rectangle
    // ultralytics/yolo/data/dataloaders/v5augmentations.py letterbox
    // int wpad = (w + MAX_STRIDE - 1) / MAX_STRIDE * MAX_STRIDE - w;
    // int hpad = (h + MAX_STRIDE - 1) / MAX_STRIDE * MAX_STRIDE - h;

    int wpad = target_size - w;
    int hpad = target_size - h;

    int top = hpad / 2;
    int bottom = hpad - hpad / 2;
    int left = wpad / 2;
    int right = wpad - wpad / 2;

    ncnn::Mat in_pad;
    ncnn::copy_make_border(in,
                           in_pad,
                           top,
                           bottom,
                           left,
                           right,
                           ncnn::BORDER_CONSTANT,
                           114.f);

    const float norm_vals[3] = { 1 / 255.f, 1 / 255.f, 1 / 255.f };
    in_pad.substract_mean_normalize(0, norm_vals);

    ncnn::Extractor ex = yolo.create_extractor();

    ex.input("in0", in_pad);

    std::vector<com::tencent::yoloncnn::Object> proposals;


    // stride 8
    {
        ncnn::Mat out;
        ex.extract("out0", out);
        std::vector<int> strides = {8, 16, 32}; // might have stride=64
        std::vector<GridAndStride> grid_strides;
        generate_grids_and_stride(in_pad.w, in_pad.h, strides, grid_strides);
        generate_proposals(grid_strides, out, prob_threshold, proposals, num_class);

//        std::vector<com::tencent::yoloncnn::Object> objects8;
//        generate_proposals(8, out, prob_threshold, objects8);
//
//        proposals.insert(proposals.end(), objects8.begin(), objects8.end());
//        __android_log_print(ANDROID_LOG_ERROR, "v11 fallDetectNcnn", "objects8 size = %d", objects8.size());
    }
    __android_log_print(ANDROID_LOG_ERROR, "v11 fallDetectNcnn", "proposals size = %d", proposals.size());
    // objects = proposals;
    for (auto& pro : proposals)
    {
        float x0 = pro.rect.x;
        float y0 = pro.rect.y;
        float x1 = pro.rect.x + pro.rect.width;
        float y1 = pro.rect.y + pro.rect.height;
        float& score = pro.prob;
        int& label = pro.label;

        x0 = (x0 - (wpad / 2)) / scale;
        y0 = (y0 - (hpad / 2)) / scale;
        x1 = (x1 - (wpad / 2)) / scale;
        y1 = (y1 - (hpad / 2)) / scale;

        x0 = clamp(x0, 0.f, width);
        y0 = clamp(y0, 0.f, height);
        x1 = clamp(x1, 0.f, width);
        y1 = clamp(y1, 0.f, height);

        com::tencent::yoloncnn::Object obj;
        obj.rect.x = x0;
        obj.rect.y = y0;
        obj.rect.width = x1 - x0;
        obj.rect.height = y1 - y0;
        obj.prob = score;
        obj.label = label;
        objects.push_back(obj);
    }
    __android_log_print(ANDROID_LOG_ERROR, "v11 fallDetectNcnn", "detect func end");
    // non_max_suppression(proposals, objects,
    //     img_h, img_w, hpad / 2, wpad / 2,
    //     scale, scale, prob_threshold, nms_threshold);
    return 0;
}


int Yolo::detect_v10(const cv::Mat& rgb, std::vector<com::tencent::yoloncnn::Object>& objects, float prob_threshold, float nms_threshold)
{
    int width = rgb.cols;
    int height = rgb.rows;

    // pad to multiple of 32
    int w = width;
    int h = height;
    float scale = 1.f;
    if (w > h)
    {
        scale = (float)target_size / w;
        w = target_size;
        h = h * scale;
    }
    else
    {
        scale = (float)target_size / h;
        h = target_size;
        w = w * scale;
    }
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_BGR2RGB, width, height, w, h);
    // pad to target_size rectangle
    // ultralytics/yolo/data/dataloaders/v5augmentations.py letterbox
    // int wpad = (w + MAX_STRIDE - 1) / MAX_STRIDE * MAX_STRIDE - w;
    // int hpad = (h + MAX_STRIDE - 1) / MAX_STRIDE * MAX_STRIDE - h;

    int wpad = target_size - w;
    int hpad = target_size - h;

    int top = hpad / 2;
    int bottom = hpad - hpad / 2;
    int left = wpad / 2;
    int right = wpad - wpad / 2;

    ncnn::Mat in_pad;
    ncnn::copy_make_border(in,
                           in_pad,
                           top,
                           bottom,
                           left,
                           right,
                           ncnn::BORDER_CONSTANT,
                           114.f);

    const float norm_vals[3] = { 1 / 255.f, 1 / 255.f, 1 / 255.f };
    in_pad.substract_mean_normalize(0, norm_vals);

    ncnn::Extractor ex = yolo.create_extractor();

    ex.input("in0", in_pad);

    std::vector<com::tencent::yoloncnn::Object> proposals;


    // stride 8
    {
        ncnn::Mat out;
        ex.extract("out0", out);

        std::vector<com::tencent::yoloncnn::Object> objects8;
        generate_proposals(8, out, prob_threshold, objects8);

        proposals.insert(proposals.end(), objects8.begin(), objects8.end());
    }

    // stride 16
    {
        ncnn::Mat out;

        ex.extract("out1", out);

        std::vector<com::tencent::yoloncnn::Object> objects16;
        generate_proposals(16, out, prob_threshold, objects16);

        proposals.insert(proposals.end(), objects16.begin(), objects16.end());
    }

    // stride 32
    {
        ncnn::Mat out;

        ex.extract("out2", out);

        std::vector<com::tencent::yoloncnn::Object> objects32;
        generate_proposals(32, out, prob_threshold, objects32);

        proposals.insert(proposals.end(), objects32.begin(), objects32.end());
    }
    // objects = proposals;
    for (auto& pro : proposals)
    {
        float x0 = pro.rect.x;
        float y0 = pro.rect.y;
        float x1 = pro.rect.x + pro.rect.width;
        float y1 = pro.rect.y + pro.rect.height;
        float& score = pro.prob;
        int& label = pro.label;

        x0 = (x0 - (wpad / 2)) / scale;
        y0 = (y0 - (hpad / 2)) / scale;
        x1 = (x1 - (wpad / 2)) / scale;
        y1 = (y1 - (hpad / 2)) / scale;

        x0 = clamp(x0, 0.f, width);
        y0 = clamp(y0, 0.f, height);
        x1 = clamp(x1, 0.f, width);
        y1 = clamp(y1, 0.f, height);

        com::tencent::yoloncnn::Object obj;
        obj.rect.x = x0;
        obj.rect.y = y0;
        obj.rect.width = x1 - x0;
        obj.rect.height = y1 - y0;
        obj.prob = score;
        obj.label = label;
        objects.push_back(obj);
    }
    // non_max_suppression(proposals, objects,
    //     img_h, img_w, hpad / 2, wpad / 2,
    //     scale, scale, prob_threshold, nms_threshold);
    return 0;
}

int Yolo::detect_v7(const cv::Mat &rgb, std::vector<com::tencent::yoloncnn::Object> &objects, float prob_threshold, float nms_threshold) {
    int img_w = rgb.cols;
    int img_h = rgb.rows;
    // letterbox pad to multiple of 32
    int w = img_w;
    int h = img_h;
    float scale = 1.f;
    if (w > h) {
        scale = (float) target_size / w;
        w = target_size;
        h = h * scale;
    } else {
        scale = (float) target_size / h;
        h = target_size;
        w = w * scale;
    }
    const int max_stride = 64;
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_RGB, img_w, img_h, w, h);

    // pad to target_size rectangle
    int wpad = (w + max_stride - 1) / max_stride * max_stride - w;
    int hpad = (h + max_stride - 1) / max_stride * max_stride - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 114.f);
    in_pad.substract_mean_normalize(0, norm_vals);

    ncnn::Extractor ex = yolo.create_extractor();

    ex.input("in0", in_pad);

    std::vector<com::tencent::yoloncnn::Object> proposals;

    // stride 8
    {
        ncnn::Mat out;
        ex.extract("out0", out);

        ncnn::Mat anchors(6);
        anchors[0] = 12.f;
        anchors[1] = 16.f;
        anchors[2] = 19.f;
        anchors[3] = 36.f;
        anchors[4] = 40.f;
        anchors[5] = 28.f;

        std::vector<com::tencent::yoloncnn::Object> objects8;
        generate_proposals(anchors, 8, in_pad, out, prob_threshold, objects8);

        proposals.insert(proposals.end(), objects8.begin(), objects8.end());
    }

    // stride 16
    {
        ncnn::Mat out;
        ex.extract("out1", out);

        ncnn::Mat anchors(6);
        anchors[0] = 36.f;
        anchors[1] = 75.f;
        anchors[2] = 76.f;
        anchors[3] = 55.f;
        anchors[4] = 72.f;
        anchors[5] = 146.f;

        std::vector<com::tencent::yoloncnn::Object> objects16;
        generate_proposals(anchors, 16, in_pad, out, prob_threshold, objects16);

        proposals.insert(proposals.end(), objects16.begin(), objects16.end());
    }

    // stride 32
    {
        ncnn::Mat out;
        ex.extract("out2", out);

        ncnn::Mat anchors(6);
        anchors[0] = 142.f;
        anchors[1] = 110.f;
        anchors[2] = 192.f;
        anchors[3] = 243.f;
        anchors[4] = 459.f;
        anchors[5] = 401.f;

        std::vector<com::tencent::yoloncnn::Object> objects32;
        generate_proposals(anchors, 32, in_pad, out, prob_threshold, objects32);

        proposals.insert(proposals.end(), objects32.begin(), objects32.end());
    }

    // sort all proposals by score from highest to lowest
    qsort_descent_inplace(proposals);

    // apply nms with nms_threshold
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold);

    int count = picked.size();
    __android_log_print(ANDROID_LOG_ERROR, "yolo", "%d", count);
    objects.resize(count);
    for (int i = 0; i < count; i++) {
        objects[i] = proposals[picked[i]];

        // adjust offset to original unpadded
        float x0 = (objects[i].rect.x - (wpad / 2)) / scale;
        float y0 = (objects[i].rect.y - (hpad / 2)) / scale;
        float x1 = (objects[i].rect.x + objects[i].rect.width - (wpad / 2)) / scale;
        float y1 = (objects[i].rect.y + objects[i].rect.height - (hpad / 2)) / scale;

        // clip
        x0 = std::max(std::min(x0, (float) (img_w - 1)), 0.f);
        y0 = std::max(std::min(y0, (float) (img_h - 1)), 0.f);
        x1 = std::max(std::min(x1, (float) (img_w - 1)), 0.f);
        y1 = std::max(std::min(y1, (float) (img_h - 1)), 0.f);

        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width = x1 - x0;
        objects[i].rect.height = y1 - y0;
    }

    return 0;
}

int Yolo::detect_v5(const cv::Mat &rgb, std::vector<com::tencent::yoloncnn::Object> &objects,
                    float prob_threshold, float nms_threshold) {
    __android_log_print(ANDROID_LOG_ERROR, "yolo", "detect begin");
    target_size = 640;
    int img_w = rgb.cols;
    int img_h = rgb.rows;

    // pad to multiple of 32
    int w = img_w;
    int h = img_h;
    float scale = 1.f;
    if (w > h)
    {
        scale = (float)target_size / w;
        w = target_size;
        h = h * scale;
    }
    else
    {
        scale = (float)target_size / h;
        h = target_size;
        w = w * scale;
    }
//    __android_log_print(ANDROID_LOG_ERROR, "yolo", "detect target_size %d",target_size);
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_RGB2BGR, img_w, img_h, w, h);

    // pad to target_size rectangle
    int wpad = (w + 31) / 32 * 32 - w;
    int hpad = (h + 31) / 32 * 32 - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 114.f);

    in_pad.substract_mean_normalize(0, norm_vals);
//    __android_log_print(ANDROID_LOG_ERROR, "yolo", "detect create_extractor");
    ncnn::Extractor ex = yolo.create_extractor();

    ex.input("images", in_pad);

    std::vector<com::tencent::yoloncnn::Object> proposals;

    // anchor setting from yolov5/models/yolov5s.yaml

    // stride 8
    {
        ncnn::Mat out;
        ex.extract("output", out);

        ncnn::Mat anchors(6);
        anchors[0] = 10.f;
        anchors[1] = 13.f;
        anchors[2] = 16.f;
        anchors[3] = 30.f;
        anchors[4] = 33.f;
        anchors[5] = 23.f;

        std::vector<com::tencent::yoloncnn::Object> objects8;
        generate_proposals_v5(anchors, 8, in_pad, out, prob_threshold, objects8);

        proposals.insert(proposals.end(), objects8.begin(), objects8.end());
    }

    // stride 16
    {
        ncnn::Mat out;
        ex.extract("781", out);

        ncnn::Mat anchors(6);
        anchors[0] = 30.f;
        anchors[1] = 61.f;
        anchors[2] = 62.f;
        anchors[3] = 45.f;
        anchors[4] = 59.f;
        anchors[5] = 119.f;

        std::vector<com::tencent::yoloncnn::Object> objects16;
        generate_proposals_v5(anchors, 16, in_pad, out, prob_threshold, objects16);

        proposals.insert(proposals.end(), objects16.begin(), objects16.end());
    }

    // stride 32
    {
        ncnn::Mat out;
        ex.extract("801", out);

        ncnn::Mat anchors(6);
        anchors[0] = 116.f;
        anchors[1] = 90.f;
        anchors[2] = 156.f;
        anchors[3] = 198.f;
        anchors[4] = 373.f;
        anchors[5] = 326.f;

        std::vector<com::tencent::yoloncnn::Object> objects32;
        generate_proposals_v5(anchors, 32, in_pad, out, prob_threshold, objects32);

        proposals.insert(proposals.end(), objects32.begin(), objects32.end());
    }

    // sort all proposals by score from highest to lowest
    qsort_descent_inplace(proposals);

    // apply nms with nms_threshold
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold);

    int count = picked.size();
//    __android_log_print(ANDROID_LOG_ERROR, "yolo", "%d", count);
    objects.resize(count);
    for (int i = 0; i < count; i++) {
        objects[i] = proposals[picked[i]];

        // adjust offset to original unpadded
        float x0 = (objects[i].rect.x - (wpad / 2)) / scale;
        float y0 = (objects[i].rect.y - (hpad / 2)) / scale;
        float x1 = (objects[i].rect.x + objects[i].rect.width - (wpad / 2)) / scale;
        float y1 = (objects[i].rect.y + objects[i].rect.height - (hpad / 2)) / scale;

        // clip
        x0 = std::max(std::min(x0, (float) (img_w - 1)), 0.f);
        y0 = std::max(std::min(y0, (float) (img_h - 1)), 0.f);
        x1 = std::max(std::min(x1, (float) (img_w - 1)), 0.f);
        y1 = std::max(std::min(y1, (float) (img_h - 1)), 0.f);

        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width = x1 - x0;
        objects[i].rect.height = y1 - y0;
    }

    return 0;
}


int Yolo::detect_rtmpose(const cv::Mat &rgb, std::vector<com::tencent::yoloncnn::Object> &objects,
                         float prob_threshold, float nms_threshold) {
    int img_w = rgb.cols;
    int img_h = rgb.rows;
    // letterbox pad to multiple of 32
    int w = img_w;
    int h = img_h;
    float scale = 1.f;
    if (w > h) {
        scale = (float) target_size / w;
        w = target_size;
        h = h * scale;
    } else {
        scale = (float) target_size / h;
        h = target_size;
        w = w * scale;
    }
    const int max_stride = 64;
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_RGB, img_w, img_h, w, h);

    // pad to target_size rectangle
    int wpad = target_size - w;
    int hpad = target_size - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 114.f);
    in_pad.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = yolo.create_extractor();

    ex.input("images", in);

    std::vector<com::tencent::yoloncnn::Object> proposals;
    __android_log_print(ANDROID_LOG_ERROR, "fallDetectNcnn", "2222");
    // stride 8
    {
        ncnn::Mat out0;

        ex.extract("1211", out0);

        std::vector<com::tencent::yoloncnn::Object> objects8;
        generate_proposals(out0, 8, prob_threshold, objects8);

        proposals.insert(proposals.end(), objects8.begin(), objects8.end());
    }

    // stride 16
    {
        ncnn::Mat out1;
        ex.extract("1213", out1);

        std::vector<com::tencent::yoloncnn::Object> objects16;
        generate_proposals(out1, 16, prob_threshold, objects16);

        proposals.insert(proposals.end(), objects16.begin(), objects16.end());
    }

    // stride 32
    {
        ncnn::Mat out2;
        ex.extract("1215", out2);

        std::vector<com::tencent::yoloncnn::Object> objects32;
        generate_proposals(out2, 32, prob_threshold, objects32);

        proposals.insert(proposals.end(), objects32.begin(), objects32.end());
    }
    __android_log_print(ANDROID_LOG_ERROR, "fallDetectNcnn", "proposals size = %d",proposals.size());
    // sort all proposals by score from highest to lowest
    qsort_descent_inplace(proposals);
    __android_log_print(ANDROID_LOG_ERROR, "fallDetectNcnn", "proposals size11 = %d",proposals.size());
    // apply nms with nms_threshold
    std::vector<int> picked;
    nms_sorted_bboxes_rtmpose(proposals, picked, nms_threshold, true);
    __android_log_print(ANDROID_LOG_ERROR, "fallDetectNcnn", "4444");
    int count = picked.size();
    __android_log_print(ANDROID_LOG_ERROR, "fallDetectNcnn", "count = %d", count);
    objects.resize(count);
    for (int i = 0; i < count; i++) {
        objects[i] = proposals[picked[i]];

        // adjust offset to original unpadded
        float x0 = (objects[i].rect.x - (wpad / 2)) / scale;
        float y0 = (objects[i].rect.y - (hpad / 2)) / scale;
        float x1 = (objects[i].rect.x + objects[i].rect.width - (wpad / 2)) / scale;
        float y1 = (objects[i].rect.y + objects[i].rect.height - (hpad / 2)) / scale;

        // clip
        x0 = std::max(std::min(x0, (float) (img_w - 1)), 0.f);
        y0 = std::max(std::min(y0, (float) (img_h - 1)), 0.f);
        x1 = std::max(std::min(x1, (float) (img_w - 1)), 0.f);
        y1 = std::max(std::min(y1, (float) (img_h - 1)), 0.f);

        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width = x1 - x0;
        objects[i].rect.height = y1 - y0;
    }

    return 0;
}


// 判断一个"人/头部"框是否与某辆两轮车构成"骑车人"关系
// 说明:这是启发式规则(可在优化阶段调整阈值)
static bool is_rider_associated(const com::tencent::yoloncnn::Object& head,
                                const com::tencent::yoloncnn::Object& vehicle)
{
    float head_area = head.rect.width * head.rect.height;
    if (head_area <= 0.f)
        return false;

    // 情况1:头部框与车辆框明显相交(交叠面积 > 头部框的 15%)
    cv::Rect_<float> inter = head.rect & vehicle.rect;
    if (inter.area() > 0.15f * head_area)
        return true;

    // 情况2:头部位于车辆正上方附近(骑车时头通常在车把/车座上方)
    float head_cx = head.rect.x + head.rect.width * 0.5f;
    float head_bottom = head.rect.y + head.rect.height;
    float v_top = vehicle.rect.y;
    float v_h = vehicle.rect.height;
    bool x_aligned = (head_cx >= vehicle.rect.x) && (head_cx <= vehicle.rect.x + vehicle.rect.width);
    bool y_near = (head_bottom >= v_top - 0.3f * v_h) && (head_bottom <= v_top + 0.7f * v_h);
    return x_aligned && y_near;
}

int Yolo::draw(cv::Mat& rgb, const std::vector<com::tencent::yoloncnn::Object>& objects, const char* modeltype)
{
    const bool is_helmet = (modeltype != NULL && strcmp(modeltype, "helmet") == 0);

    // 初始化中文字体渲染(幂等,仅首次加载系统字体)
    cn_font_init();

    // 本项目 8 类(顺序必须与 training/data.yaml 的 names 完全一致)
    static const char* helmet_names[] = {
            "行人", "轿车", "公交车", "卡车", "摩托车", "自行车", "头盔", "头部"
    };
    // COCO 80 类(临时演示模型 yolov8n 使用)
    static const char* coco_names[] = {
            "行人", "自行车", "汽车", "摩托车", "飞机", "公交车", "火车", "卡车", "船", "交通灯",
            "消防栓", "停止标志", "停车计时表", "长椅", "鸟", "猫", "狗", "马", "羊", "牛",
            "大象", "熊", "斑马", "长颈鹿", "背包", "雨伞", "手提包", "领带", "行李箱", "飞盘",
            "滑雪板", "单板滑雪", "运动球", "风筝", "棒球棒", "棒球手套", "滑板", "冲浪板", "网球拍", "瓶子",
            "酒杯", "杯子", "叉子", "刀", "勺子", "碗", "香蕉", "苹果", "三明治", "橙子",
            "西兰花", "胡萝卜", "热狗", "披萨", "甜甜圈", "蛋糕", "椅子", "沙发", "盆栽", "床",
            "餐桌", "马桶", "电视", "笔记本电脑", "鼠标", "遥控器", "键盘", "手机", "微波炉", "烤箱",
            "烤面包机", "水槽", "冰箱", "书", "时钟", "花瓶", "剪刀", "泰迪熊", "吹风机", "牙刷"
    };
    const char** class_names = is_helmet ? helmet_names : coco_names;
    const int num_class = is_helmet ? 8 : 80;
    const int LABEL_PERSON = 0;
    const int LABEL_MOTORCYCLE = 4;
    const int LABEL_BICYCLE = 5;
    const int LABEL_HELMET = 6;   // 戴头盔
    const int LABEL_HEAD = 7;     // 未戴头盔(裸头)

    // 目标跟踪 + 单目测距测速(static 跨帧持久,仅初始化一次)
    static Tracker s_tracker;
    double now_sec = std::chrono::duration<double>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    s_tracker.update(objects, now_sec, rgb.cols, rgb.rows, is_helmet);
    const std::vector<Track>& tracks = s_tracker.tracks();

    // 先收集所有两轮车框,用于判断"骑车人"(仅头盔模型)
    std::vector<const com::tencent::yoloncnn::Object*> two_wheelers;
    if (is_helmet) {
        for (size_t i = 0; i < tracks.size(); i++) {
            if (tracks[i].obj.label == LABEL_MOTORCYCLE || tracks[i].obj.label == LABEL_BICYCLE)
                two_wheelers.push_back(&tracks[i].obj);
        }
    }

    static const unsigned char colors[19][3] = {
            { 54,  67, 244},
            { 99,  30, 233},
            {176,  39, 156},
            {183,  58, 103},
            {181,  81,  63},
            {243, 150,  33},
            {244, 169,   3},
            {212, 188,   0},
            {136, 150,   0},
            { 80, 175,  76},
            { 74, 195, 139},
            { 57, 220, 205},
            { 59, 235, 255},
            {  7, 193, 255},
            {  0, 152, 255},
            { 34,  87, 255},
            { 72,  85, 121},
            {158, 158, 158},
            {139, 125,  96}
    };

    int color_index = 0;
    bool any_alert = false;
    float max_kmh = 0.0f;
    const float ALERT_KMH = 10.0f;   // 接近速度超过此值(km/h)触发快速接近警告

    for (size_t i = 0; i < tracks.size(); i++)
    {
        const Track& tk = tracks[i];
        const com::tencent::yoloncnn::Object& obj = tk.obj;

        // 类别名(越界保护:label 超出类别数时不崩溃)
        const char* cls_name = (obj.label >= 0 && obj.label < num_class) ? class_names[obj.label] : "unknown";

        // 判断是否为"骑车人":人/头部框与两轮车框关联(仅头盔模型)
        bool is_rider = false;
        if (is_helmet && (obj.label == LABEL_PERSON || obj.label == LABEL_HELMET || obj.label == LABEL_HEAD)) {
            for (size_t v = 0; v < two_wheelers.size(); v++) {
                if (is_rider_associated(obj, *two_wheelers[v])) { is_rider = true; break; }
            }
        }

        // 快速接近判定(km/h)
        float kmh = tk.speed * 3.6f;
        bool fast_approach = tk.has_speed && tk.approaching && kmh >= ALERT_KMH;
        if (fast_approach) {
            any_alert = true;
            if (kmh > max_kmh) max_kmh = kmh;
        }

        // 颜色:快速接近=红加粗(最高优先警告),未戴头盔=红,戴头盔=绿,其余按调色板
        cv::Scalar cc;
        int thickness = 2;
        if (fast_approach) {
            cc = cv::Scalar(0, 0, 255);        // BGR: 红
            thickness = 5;
        } else if (is_helmet && obj.label == LABEL_HEAD) {
            cc = cv::Scalar(0, 0, 255);        // BGR: 红
        } else if (is_helmet && obj.label == LABEL_HELMET) {
            cc = cv::Scalar(0, 200, 0);        // BGR: 绿
        } else {
            const unsigned char* color = colors[color_index % 19];
            cc = cv::Scalar(color[0], color[1], color[2]);
        }
        color_index++;

        cv::rectangle(rgb, obj.rect, cc, thickness);

        // 标签文字(UTF-8 中文,由 cn_font 调用系统字体渲染)
        char base[192];
        if (!is_helmet)
            sprintf(base, "%s %.0f%%", cls_name, obj.prob * 100);
        else if (obj.label == LABEL_HEAD && is_rider)
            sprintf(base, "未戴头盔! %.0f%%", obj.prob * 100);
        else if (obj.label == LABEL_HEAD)
            sprintf(base, "头部 %.0f%%", obj.prob * 100);
        else if (obj.label == LABEL_HELMET && is_rider)
            sprintf(base, "头盔(骑车人) %.0f%%", obj.prob * 100);
        else if (obj.label == LABEL_PERSON && is_rider)
            sprintf(base, "骑车人 %.0f%%", obj.prob * 100);
        else
            sprintf(base, "%s %.0f%%", cls_name, obj.prob * 100);

        // 追加接近/远离速度(km/h,单目测距估计,正=接近)
        char text[256];
        if (tk.has_speed) {
            if (kmh >= 1.0f)
                sprintf(text, "%s 接近%.0fkm/h", base, kmh);
            else if (kmh <= -1.0f)
                sprintf(text, "%s 远离%.0fkm/h", base, -kmh);
            else
                sprintf(text, "%s 静止", base);
        } else {
            sprintf(text, "%s", base);
        }

        const float font_size = 26.f;
        const int pad = 2;
        cv::Size label_size = cn_font_measure(text, font_size);

        int x = (int)obj.rect.x;
        int y = (int)obj.rect.y - label_size.height - pad * 2;
        if (y < 0)
            y = 0;
        if (x + label_size.width + pad * 2 > rgb.cols)
            x = rgb.cols - label_size.width - pad * 2;
        if (x < 0)
            x = 0;

        cv::Scalar textcc = (cc[0] + cc[1] + cc[2] >= 381) ? cv::Scalar(0, 0, 0) : cv::Scalar(255, 255, 255);

        // 背景色块 + 中文文字
        if (label_size.width > 0 && label_size.height > 0) {
            cv::rectangle(rgb, cv::Rect(cv::Point(x, y),
                                        cv::Size(label_size.width + pad * 2, label_size.height + pad * 2)), cc, -1);
            cn_font_draw(rgb, text, x + pad, y + pad, font_size, textcc);
        }
    }

    // 顶部快速接近警告横幅(仅当有目标快速接近时)
    if (any_alert) {
        const int banner_h = 64;
        cv::rectangle(rgb, cv::Rect(0, 0, rgb.cols, banner_h), cv::Scalar(0, 0, 220), -1);
        char warn[128];
        sprintf(warn, "注意 前方目标快速接近 %.0fkm/h", max_kmh);
        const float wf = 34.f;
        cv::Size ws = cn_font_measure(warn, wf);
        int wx = (rgb.cols - ws.width) / 2;
        if (wx < 0) wx = 0;
        int wy = (banner_h - ws.height) / 2;
        if (wy < 0) wy = 0;
        cn_font_draw(rgb, warn, wx, wy, wf, cv::Scalar(255, 255, 255));
    }

    return 0;
}

