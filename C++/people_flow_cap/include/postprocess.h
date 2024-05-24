#ifndef _RKNN_ZERO_COPY_DEMO_POSTPROCESS_H_
#define _RKNN_ZERO_COPY_DEMO_POSTPROCESS_H_

#include <stdint.h>
#include <vector>
#include "opencv2/core/core.hpp"

#define OBJ_NAME_MAX_SIZE 16
#define OBJ_NUMB_MAX_SIZE 64
#define OBJ_CLASS_NUM     1
#define NMS_THRESH        0.5
#define BOX_THRESH        0.3
#define PROP_BOX_SIZE     (5+OBJ_CLASS_NUM)
#define TRACK_LENGTH      20
#define TRACK_IOU_THRESH  0.3
#define max_lose_times    10

typedef struct _BOX_RECT
{
    int left;
    int right;
    int top;
    int bottom;
} BOX_RECT;

typedef struct __detect_result_t
{
    char name[OBJ_NAME_MAX_SIZE];
    BOX_RECT box;
    float prop;
} detect_result_t;

typedef struct _detect_result_group_t
{
    int id;
    int count;
    detect_result_t results[OBJ_NUMB_MAX_SIZE];
} detect_result_group_t;

typedef struct _PERSON
{
    int id;
    int lose_times = 0;
    std::vector<cv::Mat> person_status;
    cv::Mat P = cv::Mat::eye(cv::Size(6, 6), CV_32FC1);
} person;

int post_process(int8_t *input0, int8_t *input1, int8_t *input2, int model_in_h, int model_in_w,
                 float conf_threshold, float nms_threshold, float scale_w, float scale_h,
                 std::vector<int32_t> &qnt_zps, std::vector<float> &qnt_scales,
                 detect_result_group_t *group);

void kalman_filtering(std::vector<person>* people, detect_result_group_t* group);

bool cross_line(int x1, int y1, int x2, int y2, int x3, int y3, int x4, int y4);

void deinitPostProcess();
#endif //_RKNN_ZERO_COPY_DEMO_POSTPROCESS_H_
