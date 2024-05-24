// Copyright (c) 2021 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <dirent.h>
#include <iostream>
#include <fstream>

#define _BASETSD_H

#include "RgaUtils.h"
#include "im2d.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "postprocess.h"
#include "retinaface.h"
#include "vgg16.h"
#include "rga.h"
#include "rknn_api.h"

#define PERF_WITH_POST 1

int compare_boundary(int boundary_xmin, int boundary_ymin, int boundary_xmax, int boundary_ymax, int* xmin, int* ymin, int* xmax, int* ymax, int x_ratio, int y_ratio)
{
	if (*xmin < boundary_xmin)
	{
		if ((boundary_xmin - *xmin) < (*xmax - *xmin) * x_ratio)
		{
			*xmin = boundary_xmin;
		}
		else
		{
			return 0;
		}
	}
	
	if (*ymin < boundary_ymin)
	{
		if ((boundary_ymin - *ymin) < (*ymax - *ymin) * y_ratio)
		{
			*ymin = boundary_ymin;
		}
		else
		{
			return 0;
		}
	}
	
	if (*xmax > boundary_xmax)
	{
		if ((*xmax - boundary_xmax) < (*xmax - *xmin) * x_ratio)
		{
			*xmax = boundary_xmax;
		}
		else
		{
			return 0;
		}
	}
	
	if (*ymax > boundary_ymax)
	{
		if ((*ymax - boundary_ymax) < (*ymax - *ymin) * y_ratio)
		{
			*ymax = boundary_ymax;
		}
		else
		{
			return 0;
		}
	}
	
	return 1;
}

/*-------------------------------------------
                  Main Functions
-------------------------------------------*/
int main(int argc, char** argv)
{
  	char*          retinaface_model_name = NULL;
  	rknn_context   retinaface_ctx;
  	int            retinaface_width      = 0;
  	int            retinaface_height     = 0;
  	int            retinaface_channel    = 0;
  	std::vector<float> retinaface_out_scales;
  	std::vector<int32_t> retinaface_out_zps;
  	rknn_input_output_num retinaface_io_num;
  	static unsigned char *retinaface_model_data;
  	
  	float dst_landmark[5][2] = {{54.7065, 73.8519},
				    {105.0454, 73.5734},
				    {80.036, 102.4808},
				    {59.3561, 131.9507},
				    {89.6141, 131.7201}};
	cv::Mat dst(5, 2, CV_32FC1, dst_landmark);
	memcpy(dst.data, dst_landmark, 2 * 5 * sizeof(float));
  	
  	char*          vgg16_model_name = NULL;
  	rknn_context   vgg16_ctx;
  	int            vgg16_width      = 0;
  	int            vgg16_height     = 0;
  	int            vgg16_channel    = 0;
  	rknn_input_output_num vgg16_io_num;
  	static unsigned char *vgg16_model_data;
  	float *vgg16_result;

  	const float    nms_threshold      = NMS_THRESH;
  	const float    box_conf_threshold = BOX_THRESH;
  	const float    vgg16_threshold  = VGG16_THRESH;
  	struct timeval start_time, stop_time;
  	int            ret;

  	if (argc != 4) {
		printf("Usage: %s <retinaface model> <vgg16 model> <jpg/1> \n", argv[0]);
		return -1;
  	}

  	printf("post process config: box_conf_threshold = %.2f, nms_threshold = %.2f\n", box_conf_threshold, nms_threshold);

  	retinaface_model_name = (char*)argv[1];
  	vgg16_model_name = (char*)argv[2];
  	char* image_name = argv[3];
  	
	cv::namedWindow("Image Window");
	
  	create_retinaface(retinaface_model_name, &retinaface_ctx, retinaface_width, retinaface_height, retinaface_channel, retinaface_out_scales, retinaface_out_zps, retinaface_io_num, retinaface_model_data);
  	create_vgg16(vgg16_model_name, &vgg16_ctx, vgg16_width, vgg16_height, vgg16_channel, vgg16_io_num, vgg16_model_data);
  	
  	rknn_input retinaface_inputs[1];
  	memset(retinaface_inputs, 0, sizeof(retinaface_inputs));
  	retinaface_inputs[0].index        = 0;
  	retinaface_inputs[0].type         = RKNN_TENSOR_UINT8;
  	retinaface_inputs[0].size         = retinaface_width * retinaface_height * retinaface_channel;
  	retinaface_inputs[0].fmt          = RKNN_TENSOR_NHWC;
  	retinaface_inputs[0].pass_through = 0;
  	
  	rknn_output retinaface_outputs[retinaface_io_num.n_output];
  	memset(retinaface_outputs, 0, sizeof(retinaface_outputs));
  	for (int i = 0; i < retinaface_io_num.n_output; i++) {
  		if (i != 1)
  		{
			retinaface_outputs[i].want_float = 0;
		}
		else
		{
			retinaface_outputs[i].want_float = 1;
		}
  	}
  	
  	rknn_input vgg16_inputs[1];
  	memset(vgg16_inputs, 0, sizeof(vgg16_inputs));
  	vgg16_inputs[0].index        = 0;
  	vgg16_inputs[0].type         = RKNN_TENSOR_UINT8;
  	vgg16_inputs[0].size         = vgg16_width * vgg16_height * vgg16_channel;
  	vgg16_inputs[0].fmt          = RKNN_TENSOR_NHWC;
  	vgg16_inputs[0].pass_through = 0;
  	
  	rknn_output vgg16_outputs[vgg16_io_num.n_output];
  	memset(vgg16_outputs, 0, sizeof(vgg16_outputs));
  	for (int i = 0; i < vgg16_io_num.n_output; i++) {
		vgg16_outputs[i].want_float = 1;
  	}
	
	cv::Mat orig_img = cv::imread(image_name);
	cv::Mat img;
	int orig_img_width  = orig_img.cols;
	int orig_img_height = orig_img.rows;
	int img_width;
	int img_height;
	int right_res;
	int left_res;
	
	if (orig_img_width >= orig_img_height)
	{
		img_width = orig_img_width;
		img_height = orig_img_width;
	}
	else if (orig_img_width < orig_img_height)
	{
		img_height = orig_img_height;
		img_width = orig_img_height;
	}
	
	int x_padding = img_width - orig_img_width;
	int y_padding = img_height - orig_img_height;
	cv::copyMakeBorder(orig_img, img, 0, y_padding, 0, x_padding, cv::BorderTypes::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
	
	cv::resize(img, img, cv::Size(retinaface_width, retinaface_height), (0, 0), (0, 0), cv::INTER_LINEAR);
	
	detect_result_group_t retinaface_detect_result_group;
	
	retinaface_inference(&retinaface_ctx, img, retinaface_width, retinaface_height, retinaface_channel, box_conf_threshold, nms_threshold, img_width, img_height, retinaface_io_num, retinaface_inputs, retinaface_outputs, retinaface_out_scales, retinaface_out_zps, &retinaface_detect_result_group);
	
	for (int i = 0; i < retinaface_detect_result_group.count; i++) {
		right_res = 1;
		left_res = 1;
		
		int eye_w = (int)(retinaface_detect_result_group.results[i].point.point_2_x - retinaface_detect_result_group.results[i].point.point_1_x);
		int eye_h = (int)(retinaface_detect_result_group.results[i].point.point_3_y - (retinaface_detect_result_group.results[i].point.point_2_y + retinaface_detect_result_group.results[i].point.point_1_y) / 2) + 1;
		
		int right_eye_xmin = (int)(retinaface_detect_result_group.results[i].point.point_1_x - eye_w / 2) - 1;
		int right_eye_xmax = (int)(retinaface_detect_result_group.results[i].point.point_1_x + eye_w / 2) - 1;
		int right_eye_ymin = (int)(retinaface_detect_result_group.results[i].point.point_1_y - eye_h / 2) - 1;
		int right_eye_ymax = (int)(retinaface_detect_result_group.results[i].point.point_1_y + eye_h / 2) - 1;
		
		int left_eye_xmin = (int)(retinaface_detect_result_group.results[i].point.point_2_x - eye_w / 2) - 1;
		int left_eye_xmax = (int)(retinaface_detect_result_group.results[i].point.point_2_x + eye_w / 2) - 1;
		int left_eye_ymin = (int)(retinaface_detect_result_group.results[i].point.point_2_y - eye_h / 2) - 1;
		int left_eye_ymax = (int)(retinaface_detect_result_group.results[i].point.point_2_y + eye_h / 2) - 1;
		
		right_res = compare_boundary(0, 0, orig_img_width, orig_img_height, &right_eye_xmin, &right_eye_ymin, &right_eye_xmax, &right_eye_ymax, 0.33, 0.33);
		left_res  = compare_boundary(0, 0, orig_img_width, orig_img_height, &left_eye_xmin,  &left_eye_ymin,  &left_eye_xmax,  &left_eye_ymax,  0.33, 0.33);
		
		if (right_res)
		{
			cv::Rect right_eye_roi(right_eye_xmin, right_eye_ymin, right_eye_xmax - right_eye_xmin, right_eye_ymax - right_eye_ymin);
			cv::Mat right_eye = orig_img(right_eye_roi);
			cv::resize(right_eye, right_eye, cv::Size(32, 32));
			cv::cvtColor(right_eye, right_eye, cv::COLOR_BGR2GRAY);
			
			vgg16_inference(&vgg16_ctx, right_eye, vgg16_io_num, vgg16_inputs, vgg16_outputs, &vgg16_result);
			
			if (vgg16_result[1] > 0.95)
			{
				putText(orig_img, "open", cv::Point(right_eye_xmin, right_eye_ymin - 6), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0));
				rectangle(orig_img, cv::Point(right_eye_xmin, right_eye_ymin), cv::Point(right_eye_xmax, right_eye_ymax), cv::Scalar(0, 255, 0), 1);
			}
			else
			{
				putText(orig_img, "close", cv::Point(right_eye_xmin, right_eye_ymin - 6), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255));
				rectangle(orig_img, cv::Point(right_eye_xmin, right_eye_ymin), cv::Point(right_eye_xmax, right_eye_ymax), cv::Scalar(0, 0, 255), 1);
			}
		}
		
		if (left_res)
		{
			cv::Rect left_eye_roi(left_eye_xmin, left_eye_ymin, left_eye_xmax - left_eye_xmin, left_eye_ymax - left_eye_ymin);
			cv::Mat left_eye = orig_img(left_eye_roi);
			cv::resize(left_eye, left_eye, cv::Size(32, 32));
			cv::cvtColor(left_eye, left_eye, cv::COLOR_BGR2GRAY);
			
			vgg16_inference(&vgg16_ctx, left_eye, vgg16_io_num, vgg16_inputs, vgg16_outputs, &vgg16_result);
			
			if (vgg16_result[1] > 0.95)
			{
				putText(orig_img, "open", cv::Point(left_eye_xmin, left_eye_ymin - 6), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0));
				rectangle(orig_img, cv::Point(left_eye_xmin, left_eye_ymin), cv::Point(left_eye_xmax, left_eye_ymax), cv::Scalar(0, 255, 0), 1);
			}
			else
			{
				putText(orig_img, "close", cv::Point(left_eye_xmin, left_eye_ymin - 6), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255));
				rectangle(orig_img, cv::Point(left_eye_xmin, left_eye_ymin), cv::Point(left_eye_xmax, left_eye_ymax), cv::Scalar(0, 0, 255), 1);
			}
		}
		
		if (right_res || left_res)
		{
			vgg16_output_release(&vgg16_ctx, vgg16_io_num, vgg16_outputs);
		}
	}
	cv::imshow("Image Window", orig_img);
	cv::waitKey(-1);
	cv::imwrite("./result.jpg", orig_img);

	release_retinaface(&retinaface_ctx, retinaface_model_data);
	release_vgg16(&vgg16_ctx, vgg16_model_data);
  	return 0;
}
