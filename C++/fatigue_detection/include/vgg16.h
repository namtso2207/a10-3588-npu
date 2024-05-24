#ifndef __FACENET_H__
#define __FACENET_H__

#include <stdint.h>
#include <vector>
#include "rknn_api.h"

int create_vgg16(char *model_name, rknn_context *ctx, int &width, int &height, int &channel, rknn_input_output_num &io_num, unsigned char *model_data);

int vgg16_inference(rknn_context *ctx, cv::Mat img, rknn_input_output_num io_num, rknn_input *inputs, rknn_output *outputs, float **result);

int vgg16_output_release(rknn_context *ctx, rknn_input_output_num io_num, rknn_output *outputs);

void release_vgg16(rknn_context *ctx, unsigned char *model_data);
#endif //__FACENET_H__
