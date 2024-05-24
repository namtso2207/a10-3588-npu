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
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/freetype.hpp>
#include <opencv2/imgproc/imgproc_c.h>

#define _BASETSD_H

#include "RgaUtils.h"
#include "im2d.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "postprocess.h"
#include "lpd.h"
#include "lpr.h"
#include "lpc.h"
#include "rga.h"
#include "rknn_api.h"

#define PERF_WITH_POST 1

/*-------------------------------------------
                  Functions
-------------------------------------------*/

double __get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

void putTextZH(cv::Mat &img, char* text_lpr[], int len, cv::Point org, int fontSize, cv::Scalar color, const char* fontpath)
{
	std::string result;
	for (int i = 0; i < len; ++i)
	{
		result = result + std::string(text_lpr[i]);
	}
	
	cv::Ptr<cv::freetype::FreeType2> ft2;
	ft2 = cv::freetype::createFreeType2();
	ft2->loadFontData(fontpath, 0);
	ft2->putText(img, result, org, fontSize, color, CV_FILLED, cv::LINE_AA, true);
}

/*-------------------------------------------
                  Main Functions
-------------------------------------------*/
int main(int argc, char** argv)
{
  	char*          lpd_model_name = NULL;
  	rknn_context   lpd_ctx;
  	int            lpd_width          = 0;
  	int            lpd_height         = 0;
  	int            lpd_channel        = 0;
  	std::vector<float> lpd_out_scales;
  	std::vector<int32_t> lpd_out_zps;
  	rknn_input_output_num lpd_io_num;
  	static unsigned char *lpd_model_data;
  	
  	cv::Point2f dst_lpr[4] = {cv::Point2f(30, 1), cv::Point2f(250, 1), cv::Point2f(250, 31), cv::Point2f(30, 31)};
  	cv::Point2f dst_lpc[4] = {cv::Point2f(0, 0), cv::Point2f(32, 0), cv::Point2f(32, 32), cv::Point2f(0, 32)};
  	cv::Mat M;
  	cv::Mat imagedst;
  	
  	char*          lpr_model_name = NULL;
  	rknn_context   lpr_ctx;
  	int            lpr_width          = 0;
  	int            lpr_height         = 0;
  	int            lpr_channel        = 0;
  	std::vector<float> lpr_out_scales;
  	std::vector<int32_t> lpr_out_zps;
  	rknn_input_output_num lpr_io_num;
  	static unsigned char *lpr_model_data;
  	
  	char*          lpc_model_name = NULL;
  	rknn_context   lpc_ctx;
  	int            lpc_width          = 0;
  	int            lpc_height         = 0;
  	int            lpc_channel        = 0;
  	std::vector<float> lpc_out_scales;
  	std::vector<int32_t> lpc_out_zps;
  	rknn_input_output_num lpc_io_num;
  	static unsigned char *lpc_model_data;
  	
  	const float    lpd_nms_threshold      = NMS_THRESH;
  	const float    lpd_box_conf_threshold = BOX_THRESH;
  	const float    lpr_conf_threshold     = STR_THRESH;
  	const float    lpc_conf_threshold     = COLOR_THRESH;
  	
  	struct timeval start_time, stop_time;
  	int            ret;
	
  	if (argc != 5) {
		printf("Usage: %s <lpd model> <lpr model> <lpc model> <device id> \n", argv[0]);
		return -1;
  	}

  	lpd_model_name = (char*)argv[1];
  	lpr_model_name = (char*)argv[2];
  	lpc_model_name = (char*)argv[3];
  	std::string device_number = argv[4];
  	
  	create_lpd(lpd_model_name, &lpd_ctx, lpd_width, lpd_height, lpd_channel, lpd_out_scales, lpd_out_zps, lpd_io_num, lpd_model_data);
  	create_lpr(lpr_model_name, &lpr_ctx, lpr_width, lpr_height, lpr_channel, lpr_out_scales, lpr_out_zps, lpr_io_num, lpr_model_data);
  	create_lpc(lpc_model_name, &lpc_ctx, lpc_width, lpc_height, lpc_channel, lpc_out_scales, lpc_out_zps, lpc_io_num, lpc_model_data);

  	rknn_input lpd_inputs[1];
  	memset(lpd_inputs, 0, sizeof(lpd_inputs));
  	lpd_inputs[0].index        = 0;
  	lpd_inputs[0].type         = RKNN_TENSOR_UINT8;
  	lpd_inputs[0].size         = lpd_width * lpd_height * lpd_channel;
  	lpd_inputs[0].fmt          = RKNN_TENSOR_NHWC;
  	lpd_inputs[0].pass_through = 0;

  	rknn_output lpd_outputs[lpd_io_num.n_output];
  	memset(lpd_outputs, 0, sizeof(lpd_outputs));
  	for (int i = 0; i < lpd_io_num.n_output; i++) {
		lpd_outputs[i].want_float = 1;
  	}
  	
  	rknn_input lpr_inputs[1];
  	memset(lpr_inputs, 0, sizeof(lpr_inputs));
  	lpr_inputs[0].index        = 0;
  	lpr_inputs[0].type         = RKNN_TENSOR_UINT8;
  	lpr_inputs[0].size         = lpr_width * lpr_height * lpr_channel;
  	lpr_inputs[0].fmt          = RKNN_TENSOR_NHWC;
  	lpr_inputs[0].pass_through = 0;

  	rknn_output lpr_outputs[lpr_io_num.n_output];
  	memset(lpr_outputs, 0, sizeof(lpr_outputs));
  	for (int i = 0; i < lpr_io_num.n_output; i++) {
		lpr_outputs[i].want_float = 1;
  	}
  	
  	rknn_input lpc_inputs[1];
  	memset(lpc_inputs, 0, sizeof(lpc_inputs));
  	lpc_inputs[0].index        = 0;
  	lpc_inputs[0].type         = RKNN_TENSOR_UINT8;
  	lpc_inputs[0].size         = lpc_width * lpc_height * lpc_channel;
  	lpc_inputs[0].fmt          = RKNN_TENSOR_NHWC;
  	lpc_inputs[0].pass_through = 0;

  	rknn_output lpc_outputs[lpr_io_num.n_output];
  	memset(lpc_outputs, 0, sizeof(lpc_outputs));
  	for (int i = 0; i < lpc_io_num.n_output; i++) {
		lpc_outputs[i].want_float = 1;
  	}
  	
  	cv::namedWindow("Image Window");
  	cv::Mat orig_img;
	cv::Mat img;
	cv::VideoCapture cap(std::stoi(device_number));
	cap.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
	cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);
	
	if (!cap.isOpened()) {
		printf("capture device failed to open!");
		cap.release();
		exit(-1);
	}

	if (!cap.read(orig_img)) {
		printf("Capture read error");
	}
  	
  	cv::copyMakeBorder(orig_img, img, 0, 840, 0, 0, cv::BorderTypes::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
  	int img_width  = img.cols;
  	int img_height = img.rows;
  	
  	float scale_w = (float)lpd_width / img_width;
  	float scale_h = (float)lpd_height / img_height;

  	while(1){

		if (!cap.read(orig_img)) {
			printf("Capture read error");
			break;
		}
		cv::copyMakeBorder(orig_img, img, 0, 840, 0, 0, cv::BorderTypes::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
		cv::resize(img, img, cv::Size(lpd_width, lpd_height));
		
	  	detect_result_group_t lpd_detect_result_group;
	  	lpd_inference(&lpd_ctx, img, lpd_width, lpd_height, lpd_channel, lpd_box_conf_threshold, lpd_nms_threshold, img_width, img_height, lpd_io_num, lpd_inputs, lpd_outputs, &lpd_detect_result_group);
	  	
	  	char text[256];
	  	for (int i = 0; i < lpd_detect_result_group.count; i++) {
			detect_result_t* det_result = &(lpd_detect_result_group.results[i]);
			sprintf(text, "%s %.1f%%", det_result->name, det_result->prop * 100);
			printf("%s @ (%d %d %d %d) %f\n", det_result->name, det_result->box.left, det_result->box.top,
		 			det_result->box.right, det_result->box.bottom, det_result->prop);
			int x1 = det_result->box.left;
			int y1 = det_result->box.top;
			int x2 = det_result->box.right;
			int y2 = det_result->box.bottom;
			
			cv::Point2f src[4];
			for (int j = 0; j < KEY_POINT_NUM; ++j)
			{
				if (det_result->point[j].conf < POINT_THRESH)
				{
					continue;
				}
				int ponit_x = det_result->point[j].x;
				int ponit_y = det_result->point[j].y;
				src[j] = cv::Point2f(ponit_x, ponit_y);
			}
			
			//lpr
			M = cv::getPerspectiveTransform(src, dst_lpr);
			cv::warpPerspective(orig_img, imagedst, M, cv::Size(280, 32));
			cv::cvtColor(imagedst, imagedst, cv::COLOR_BGR2GRAY);
			
			char* lpr_result[35];
	  		int len = 0;
	  		lpr_inference(&lpr_ctx, imagedst, lpr_io_num, lpr_inputs, lpr_outputs, lpr_conf_threshold, lpr_result, &len);
	  		
	  		//lpc
	  		M = cv::getPerspectiveTransform(src, dst_lpc);
	  		cv::warpPerspective(orig_img, imagedst, M, cv::Size(32, 32));
	  		
	  		char* lpc_result[1];
	  		lpc_inference(&lpc_ctx, imagedst, lpc_io_num, lpc_inputs, lpc_outputs, lpc_conf_threshold, lpc_result);
	  		
	  		rectangle(orig_img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0, 255), 3);
	  		putTextZH(orig_img, lpr_result, len, cv::Point(x1, y1 - 12), 96, cv::Scalar(0, 255, 0), "./data/simfang.ttf");
	  		
	  		putText(orig_img, lpc_result[0], cv::Point(x1, y1 - 108), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
	  	}
	  	cv::imshow("Image Window", orig_img);
		cv::waitKey(1);
	}

  	release_lpd(&lpd_ctx, lpd_model_data);
  	release_lpr(&lpr_ctx, lpr_model_data);
  	release_lpc(&lpc_ctx, lpc_model_data);
  	deinitPostProcess();

  	return 0;
}
