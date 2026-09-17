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

#include <android/asset_manager_jni.h>
#include <android/native_window_jni.h>
#include <android/native_window.h>

#include <android/log.h>
#include <jni.h>
#include <string>
#include <vector>
#include <platform.h>
#include <benchmark.h>
#include "yolo.h"
#include "ndkcamera.h"
#include "hepler.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#if __ARM_NEON

#include <arm_neon.h>

#endif // __ARM_NEON

static Yolo *g_yolo = 0;
struct Helper *helper = 0;
static ncnn::Mutex lock;
// 是否实时显示检测框
static bool isDetected = false;
const char *modeltype = nullptr;

static jclass objCls = nullptr;
static jmethodID constructortorId;
static jfieldID xId;
static jfieldID yId;
static jfieldID wId;
static jfieldID hId;
static jfieldID labelId;
static jfieldID probId;

static int draw_unsupported(cv::Mat &rgb) {
    const char text[] = "unsupported";

    int baseLine = 0;
    cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 1.0, 1, &baseLine);

    int y = (rgb.rows - label_size.height) / 2;
    int x = (rgb.cols - label_size.width) / 2;

    cv::rectangle(rgb, cv::Rect(cv::Point(x, y),
                                cv::Size(label_size.width, label_size.height + baseLine)),
                  cv::Scalar(255, 255, 255), -1);

    cv::putText(rgb, text, cv::Point(x, y + label_size.height),
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 0));

    return 0;
}

static int draw_fps(cv::Mat &rgb) {
    // resolve moving average
    float avg_fps = 0.f;
    {
        static double t0 = 0.f;
        static float fps_history[10] = {0.f};

        double t1 = ncnn::get_current_time();
        if (t0 == 0.f) {
            t0 = t1;
            return 0;
        }

        float fps = 1000.f / (t1 - t0);
        t0 = t1;

        for (int i = 9; i >= 1; i--) {
            fps_history[i] = fps_history[i - 1];
        }
        fps_history[0] = fps;

        if (fps_history[9] == 0.f) {
            return 0;
        }

        for (int i = 0; i < 10; i++) {
            avg_fps += fps_history[i];
        }
        avg_fps /= 10.f;
    }

    char text[32];
    sprintf(text, "FPS=%.2f", avg_fps);

    int baseLine = 0;
    cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

    int y = 0;
    int x = rgb.cols - label_size.width;

    cv::rectangle(rgb, cv::Rect(cv::Point(x, y),
                                cv::Size(label_size.width, label_size.height + baseLine)),
                  cv::Scalar(255, 255, 255), -1);

    cv::putText(rgb, text, cv::Point(x, y + label_size.height),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));

    return 0;
}


class MyNdkCamera : public NdkCameraWindow {
public:
    virtual void on_image_render(cv::Mat &rgb) const;
};

void MyNdkCamera::on_image_render(cv::Mat &rgb) const {
    // nanodet
    {
        ncnn::MutexLockGuard g(lock);

        if (g_yolo && isDetected) {
            std::vector<com::tencent::yoloncnn::Object> objects;
            __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "%s", modeltype);
            if (strcmp("helmet",modeltype) == 0 || strcmp("v8n",modeltype) == 0 || strcmp("v8s",modeltype) == 0){
                g_yolo->detect(rgb, objects);
            }
            else if ((strcmp("v10n",modeltype) == 0 || strcmp("v10s",modeltype) == 0)){
                g_yolo->detect_v10(rgb, objects);
            }
            else if (strcmp("v7-tiny",modeltype) == 0){
                g_yolo->detect_v7(rgb, objects);
            }
            else if (strcmp("v5",modeltype) == 0){
                g_yolo->detect_v5(rgb, objects);
            }
            else if (strcmp("rtmdet-nano",modeltype) == 0){
                g_yolo->detect_rtmpose(rgb, objects);
            }
            else if (strcmp("v11n-falldown-ncnn",modeltype) == 0){
                g_yolo->detect_v11(rgb, objects);
            }

            __android_log_print(ANDROID_LOG_ERROR, "ncnn", "detect result num = %d", objects.size());
            g_yolo->draw(rgb, objects, modeltype);
            // 在此处将数据返回给java层
            helper->update2Android(objects);

        } else {
            draw_unsupported(rgb);
        }
    }

    draw_fps(rgb);
}

static MyNdkCamera *g_camera = 0;

extern "C" {

JNIEXPORT jint JNI_OnLoad(JavaVM *vm, void *reserved) {
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "JNI_OnLoad");

    g_camera = new MyNdkCamera;

    if (helper) {
        delete helper;
        helper = 0;
    }

    // 帮助类初始化
    helper = new Helper();
    helper->init(vm);

    return JNI_VERSION_1_4;
}

JNIEXPORT void JNI_OnUnload(JavaVM *vm, void *reserved) {
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "JNI_OnUnload");

    {
        ncnn::MutexLockGuard g(lock);

        delete g_yolo;
        g_yolo = 0;

        delete helper;
        helper = 0;
    }

    delete g_camera;
    g_camera = 0;
}

// public native boolean loadModel(AssetManager mgr, int modelid, int cpugpu);
JNIEXPORT jboolean JNICALL
Java_com_tencent_yoloncnn_YoloNcnn_loadModel(JNIEnv *env, jobject thiz, jobject assetManager,
                                                 jint modelid, jint cpugpu) {
    if (modelid < 0 || modelid > 10 || cpugpu < 0 || cpugpu > 1) {
        return JNI_FALSE;
    }

    AAssetManager *mgr = AAssetManager_fromJava(env, assetManager);

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "loadModel %p", mgr);

    const char *modeltypes[] =
            {
                    // 【临时演示】当前用 COCO 通用模型 yolov8n(80类)。
                    // 训练好头盔模型后:把下行改回 "helmet",并将 helmet.param/bin 放入 assets。
                    "v8n",
                    "v8s",
                    "v10n",
                    "v10s",
                    "v7-tiny",
                    "v5s",
                    "rtmdet-nano",
                    "v11n-falldown-ncnn"
            };

    const int target_sizes[] =
            {
                    640,
                    320,
                    640,
                    640,
                    640,
                    640,
                    320,
                    640
            };

    const float mean_vals[][3] =
            {
                    {103.53f, 116.28f, 123.675f},
                    {103.53f, 116.28f, 123.675f},
                    {103.53f, 116.28f, 123.675f},
                    {103.53f, 116.28f, 123.675f},
                    {103.53f, 116.28f, 123.675f},
                    {127.0f, 127.0f, 127.0f},
                    {103.53f, 116.28f, 123.675f}, //rtmdet-nano
                    {127.0f, 127.0f, 127.0f},
            };

    const float norm_vals[][3] =
            {
                    {1 / 255.f, 1 / 255.f, 1 / 255.f},
                    {1 / 255.f, 1 / 255.f, 1 / 255.f},
                    {1 / 255.f, 1 / 255.f, 1 / 255.f},
                    {1 / 255.f, 1 / 255.f, 1 / 255.f},
                    {1 / 255.f, 1 / 255.f, 1 / 255.f},
                    {1 / 255.f, 1 / 255.f, 1 / 255.f},
                    {1 / 57.375f, 1 / 57.12f, 1 / 58.395f}, //rtmdet-nano
                    {1 / 255.f, 1 / 255.f, 1 / 255.f}
            };

    modeltype = modeltypes[(int) modelid];
    int target_size = target_sizes[(int) modelid];
    bool use_gpu = (int) cpugpu == 1;

    // reload
    {
        ncnn::MutexLockGuard g(lock);

        if (use_gpu && ncnn::get_gpu_count() == 0) {
            // no gpu
            delete g_yolo;
            g_yolo = 0;
        } else {
            if (!g_yolo)
                g_yolo = new Yolo;
            g_yolo->load(mgr, modeltype, target_size, mean_vals[(int) modelid],
                         norm_vals[(int) modelid], use_gpu);

        }
    }

    return JNI_TRUE;
}

JNIEXPORT jboolean JNICALL
Java_com_tencent_yoloncnn_YoloNcnn_openCamera(JNIEnv *env, jobject thiz, jint facing) {
    if (facing < 0 || facing > 1)
        return JNI_FALSE;

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "openCamera %d", facing);

    g_camera->open((int) facing);


    return JNI_TRUE;
}

// public native boolean closeCamera();
extern "C"
JNIEXPORT jboolean JNICALL
Java_com_tencent_yoloncnn_YoloNcnn_closeCamera(JNIEnv *env, jobject thiz) {
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "closeCamera");

    g_camera->close();

    return JNI_TRUE;
}

// public native boolean setOutputWindow(Surface surface);
JNIEXPORT jboolean JNICALL
Java_com_tencent_yoloncnn_YoloNcnn_setOutputWindow(JNIEnv *env, jobject thiz, jobject surface) {
    ANativeWindow *win = ANativeWindow_fromSurface(env, surface);

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "setOutputWindow %p", win);

    g_camera->set_window(win);

    return JNI_TRUE;
}

}



extern "C"
JNIEXPORT void JNICALL
Java_com_tencent_yoloncnn_YoloNcnn_callNativeMethod(JNIEnv *env, jobject thiz) {
    // 测试代码
    jclass javaClass = env->GetObjectClass(thiz);
    jmethodID addMethod = env->GetMethodID(javaClass, "addNub", "(II)V");
    if (addMethod == NULL) {
        __android_log_print(ANDROID_LOG_ERROR, "MyNativeClass", "Failed to find addNub method");
        return;
    }

    env->CallVoidMethod(thiz, addMethod, 3, 5);


}

extern "C"
JNIEXPORT void JNICALL
Java_com_tencent_yoloncnn_YoloNcnn_detect(JNIEnv *env, jobject thiz, jboolean start) {
    //
    isDetected = start;
}

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_tencent_yoloncnn_YoloNcnn_getDetect(JNIEnv *env, jobject thiz) {
    return isDetected;
}



extern "C"
JNIEXPORT void JNICALL
Java_com_tencent_yoloncnn_YoloNcnn_initGlobalObj(JNIEnv *env, jobject thiz) {
    jclass clazz = env->FindClass("com/tencent/yoloncnn/Object");
    helper->initGlobal((jclass) env->NewGlobalRef(clazz), env->NewGlobalRef(thiz));

}

#define ASSERT(status, ret)     if (!(status)) { return ret; }
#define ASSERT_FALSE(status)    ASSERT(status, false)
bool BitmapToMatrix(JNIEnv * env, jobject obj_bitmap, cv::Mat & matrix) {
    void * bitmapPixels;                                            // Save picture pixel data
    AndroidBitmapInfo bitmapInfo;                                   // Save picture parameters

    ASSERT_FALSE( AndroidBitmap_getInfo(env, obj_bitmap, &bitmapInfo) >= 0);        // Get picture parameters
    ASSERT_FALSE( bitmapInfo.format == ANDROID_BITMAP_FORMAT_RGBA_8888
                  || bitmapInfo.format == ANDROID_BITMAP_FORMAT_RGB_565 );          // Only ARGB? 8888 and RGB? 565 are supported
    ASSERT_FALSE( AndroidBitmap_lockPixels(env, obj_bitmap, &bitmapPixels) >= 0 );  // Get picture pixels (lock memory block)
    ASSERT_FALSE( bitmapPixels );

    if (bitmapInfo.format == ANDROID_BITMAP_FORMAT_RGBA_8888) {
        cv::Mat tmp(bitmapInfo.height, bitmapInfo.width, CV_8UC4, bitmapPixels);    // Establish temporary mat
        tmp.copyTo(matrix);                                                         // Copy to target matrix
    } else {
        cv::Mat tmp(bitmapInfo.height, bitmapInfo.width, CV_8UC2, bitmapPixels);
        cv::cvtColor(tmp, matrix, cv::COLOR_BGR5652RGB);
    }

    //convert RGB to BGR
    cv::cvtColor(matrix,matrix,cv::COLOR_RGB2BGR);

    AndroidBitmap_unlockPixels(env, obj_bitmap);            // Unlock
    return true;
}



extern "C"
JNIEXPORT jobjectArray JNICALL
Java_com_tencent_yoloncnn_YoloNcnn_detectPicure(JNIEnv *env, jobject thiz, jobject bitmap) {

    cv::Mat rgb;
    BitmapToMatrix(env,bitmap,rgb);

    double start_time = ncnn::get_current_time();
    std::vector<com::tencent::yoloncnn::Object> objects;
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "%s", modeltype);
    if (strcmp("helmet",modeltype) == 0 || strcmp("v8n",modeltype) == 0 || strcmp("v8s",modeltype) == 0){
        g_yolo->detect(rgb, objects);
    }
    else if ((strcmp("v10n",modeltype) == 0 || strcmp("v10s",modeltype) == 0)){
        g_yolo->detect_v10(rgb, objects);
    }
    else if (strcmp("v7-tiny",modeltype) == 0){
        g_yolo->detect_v7(rgb, objects);
    }
    else if (strcmp("v5",modeltype) == 0){
        g_yolo->detect_v5(rgb, objects);
    }
    else if (strcmp("rtmdet-nano",modeltype) == 0){
        g_yolo->detect_rtmpose(rgb, objects);
    }
    else if (strcmp("v11n-falldown-ncnn",modeltype) == 0){
        g_yolo->detect_v11(rgb, objects);
    }
    __android_log_print(ANDROID_LOG_ERROR, "ncnn", "%d", objects.size());

    // objects to Obj[]
    const bool is_helmet = (strcmp("helmet", modeltype) == 0);
    // 本项目 8 类(顺序必须与 training/data.yaml 的 names 一致)
    static const char* helmet_names[] = {
            "person", "car", "bus", "truck", "motorcycle", "bicycle", "helmet", "head"
    };
    // COCO 80 类(临时演示模型 yolov8n)
    static const char* coco_names[] = {
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
            "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
            "hair drier", "toothbrush"
    };
    const char** class_names = is_helmet ? helmet_names : coco_names;
    const int num_class = is_helmet ? 8 : 80;

    // init jni glue
    jclass localObjCls = env->FindClass("com/tencent/yoloncnn/YoloNcnn$Obj");
    objCls = reinterpret_cast<jclass>(env->NewGlobalRef(localObjCls));

    constructortorId = env->GetMethodID(objCls, "<init>", "()V");

    xId = env->GetFieldID(objCls, "x", "F");
    yId = env->GetFieldID(objCls, "y", "F");
    wId = env->GetFieldID(objCls, "w", "F");
    hId = env->GetFieldID(objCls, "h", "F");
    labelId = env->GetFieldID(objCls, "label", "Ljava/lang/String;");
    probId = env->GetFieldID(objCls, "prob", "F");


    jobjectArray jObjArray = env->NewObjectArray(objects.size(), objCls, nullptr);
    for (size_t i=0; i<objects.size(); i++)
    {
        jobject jObj = env->NewObject(objCls, constructortorId, thiz);
        env->SetFloatField(jObj, xId, objects[i].rect.x);
        env->SetFloatField(jObj, yId, objects[i].rect.y);
        env->SetFloatField(jObj, wId, objects[i].rect.width);
        env->SetFloatField(jObj, hId, objects[i].rect.height);
        env->SetObjectField(jObj, labelId, env->NewStringUTF(
                (objects[i].label >= 0 && objects[i].label < num_class) ? class_names[objects[i].label] : "unknown"));
        env->SetFloatField(jObj, probId, objects[i].prob);

        env->SetObjectArrayElement(jObjArray, i, jObj);
    }
    double elasped = ncnn::get_current_time() - start_time;
    __android_log_print(ANDROID_LOG_ERROR, "YoloNcnn", "inference time: %.2fms", elasped);


    return jObjArray;
}