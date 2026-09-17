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

#ifndef YOLO_H
#define YOLO_H

#include <opencv2/core/core.hpp>
#include <net.h>
#include <jni.h>

namespace com {
    namespace tencent {
        namespace yoloncnn {

            struct Object {
                cv::Rect_<float> rect;
                int label;
                float prob;
            };

        } // namespace yoloncnn
    } // namespace tencent
} // namespace com


struct GridAndStride
{
    int grid0;
    int grid1;
    int stride;
};

class Yolo
{
public:
    Yolo();

    int load(const char* modeltype, int target_size, const float* mean_vals, const float* norm_vals, bool use_gpu = false);

    int load(AAssetManager* mgr, const char* modeltype, int target_size, const float* mean_vals, const float* norm_vals, bool use_gpu = false);

    int detect(const cv::Mat& rgb, std::vector<com::tencent::yoloncnn::Object>& objects, float prob_threshold = 0.4f, float nms_threshold = 0.5f);

    int detect_v11(const cv::Mat& rgb, std::vector<com::tencent::yoloncnn::Object>& objects, float prob_threshold = 0.4f, float nms_threshold = 0.5f);

    int detect_v10(const cv::Mat& rgb, std::vector<com::tencent::yoloncnn::Object>& objects, float prob_threshold = 0.4f, float nms_threshold = 0.5f);

    int detect_v7(const cv::Mat& rgb, std::vector<com::tencent::yoloncnn::Object>& objects, float prob_threshold = 0.4f, float nms_threshold = 0.5f);

    int detect_v5(const cv::Mat& rgb, std::vector<com::tencent::yoloncnn::Object>& objects, float prob_threshold = 0.4f, float nms_threshold = 0.5f);

    int detect_rtmpose(const cv::Mat& rgb, std::vector<com::tencent::yoloncnn::Object>& objects, float prob_threshold = 0.4f, float nms_threshold = 0.5f);

    static int draw(cv::Mat& rgb, const std::vector<com::tencent::yoloncnn::Object>& objects, const char* modeltype);

private:
    ncnn::Net yolo;
    int target_size;
    int num_class = 80;   // 由 modeltype 决定:helmet=8,其余(COCO)=80
    float mean_vals[3];
    float norm_vals[3];
    ncnn::UnlockedPoolAllocator blob_pool_allocator;
    ncnn::PoolAllocator workspace_pool_allocator;
};


class YoloV5Focus : public ncnn::Layer
{
public:
    YoloV5Focus()
    {
        one_blob_only = true;
    }

    virtual int forward(const ncnn::Mat& bottom_blob, ncnn::Mat& top_blob, const ncnn::Option& opt) const
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int channels = bottom_blob.c;

        int outw = w / 2;
        int outh = h / 2;
        int outc = channels * 4;

        top_blob.create(outw, outh, outc, 4u, 1, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

#pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < outc; p++)
        {
            const float* ptr = bottom_blob.channel(p % channels).row((p / channels) % 2) + ((p / channels) / 2);
            float* outptr = top_blob.channel(p);

            for (int i = 0; i < outh; i++)
            {
                for (int j = 0; j < outw; j++)
                {
                    *outptr = *ptr;

                    outptr += 1;
                    ptr += 2;
                }

                ptr += w;
            }
        }

        return 0;
    }
};

#endif // YOLO_H
