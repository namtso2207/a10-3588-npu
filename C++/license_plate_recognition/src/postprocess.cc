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

#include "postprocess.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <fstream>
#include <locale>
#include <iostream>

#include <set>
#include <vector>
#define LABEL_LPD_TXT_PATH "./data/lpd_class.txt"
#define LABEL_LPR_TXT_PATH "./data/lpr_class.txt"
#define LABEL_LPC_TXT_PATH "./data/lpc_class.txt"

static char* lpd_labels[OBJ_CLASS_NUM];
static char* lpr_labels[STR_CLASS_NUM];
static char* lpc_labels[COLOR_CLASS_NUM];

inline static int clamp(float val, int min, int max) { return val > min ? (val < max ? val : max) : min; }

char* readLine(FILE* fp, char* buffer, int* len)
{
  int    ch;
  int    i        = 0;
  size_t buff_len = 0;

  buffer = (char*)malloc(buff_len + 1);
  if (!buffer)
    return NULL; // Out of memory

  while ((ch = fgetc(fp)) != '\n' && ch != EOF) {
    buff_len++;
    void* tmp = realloc(buffer, buff_len + 1);
    if (tmp == NULL) {
      free(buffer);
      return NULL; // Out of memory
    }
    buffer = (char*)tmp;

    buffer[i] = (char)ch;
    i++;
  }
  buffer[i] = '\0';

  *len = buff_len;

  // Detect end
  if (ch == EOF && (i == 0 || ferror(fp))) {
    free(buffer);
    return NULL;
  }
  return buffer;
}

int readLines(const char* fileName, char* lines[], int max_line)
{
  FILE* file = fopen(fileName, "r");
  char* s;
  int   i = 0;
  int   n = 0;

  if (file == NULL) {
    printf("Open %s fail!\n", fileName);
    return -1;
  }

  while ((s = readLine(file, s, &n)) != NULL) {
    lines[i++] = s;
    if (i >= max_line)
      break;
  }
  //fclose(file);
  return i;
}

int loadLabelName(const char* locationFilename, char* label[], int len)
{
  printf("loadLabelName %s\n", locationFilename);
  readLines(locationFilename, label, len);
  return 0;
}

static float CalculateOverlap(float xmin0, float ymin0, float xmax0, float ymax0, float xmin1, float ymin1, float xmax1,
                              float ymax1)
{
  float w = fmax(0.f, fmin(xmax0, xmax1) - fmax(xmin0, xmin1) + 1.0);
  float h = fmax(0.f, fmin(ymax0, ymax1) - fmax(ymin0, ymin1) + 1.0);
  float i = w * h;
  float u = (xmax0 - xmin0 + 1.0) * (ymax0 - ymin0 + 1.0) + (xmax1 - xmin1 + 1.0) * (ymax1 - ymin1 + 1.0) - i;
  return u <= 0.f ? 0.f : (i / u);
}

static int nms(int validCount, std::vector<float>& outputLocations, std::vector<int> classIds, std::vector<int>& order,
               int filterId, float threshold)
{
  for (int i = 0; i < validCount; ++i) {
    if (order[i] == -1 || classIds[i] != filterId) {
      continue;
    }
    int n = order[i];
    for (int j = i + 1; j < validCount; ++j) {
      int m = order[j];
      if (m == -1 || classIds[i] != filterId) {
        continue;
      }
      float xmin0 = outputLocations[n * 4 + 0];
      float ymin0 = outputLocations[n * 4 + 1];
      float xmax0 = outputLocations[n * 4 + 0] + outputLocations[n * 4 + 2];
      float ymax0 = outputLocations[n * 4 + 1] + outputLocations[n * 4 + 3];

      float xmin1 = outputLocations[m * 4 + 0];
      float ymin1 = outputLocations[m * 4 + 1];
      float xmax1 = outputLocations[m * 4 + 0] + outputLocations[m * 4 + 2];
      float ymax1 = outputLocations[m * 4 + 1] + outputLocations[m * 4 + 3];

      float iou = CalculateOverlap(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1);

      if (iou > threshold) {
        order[j] = -1;
      }
    }
  }
  return 0;
}

static int quick_sort_indice_inverse(std::vector<float>& input, int left, int right, std::vector<int>& indices)
{
  float key;
  int   key_index;
  int   low  = left;
  int   high = right;
  if (left < right) {
    key_index = indices[left];
    key       = input[left];
    while (low < high) {
      while (low < high && input[high] <= key) {
        high--;
      }
      input[low]   = input[high];
      indices[low] = indices[high];
      while (low < high && input[low] >= key) {
        low++;
      }
      input[high]   = input[low];
      indices[high] = indices[low];
    }
    input[low]   = key;
    indices[low] = key_index;
    quick_sort_indice_inverse(input, left, low - 1, indices);
    quick_sort_indice_inverse(input, low + 1, right, indices);
  }
  return low;
}

static float sigmoid(float x) { return 1.0 / (1.0 + expf(-x)); }

static float unsigmoid(float y) { return -1.0 * logf((1.0 / y) - 1.0); }

inline static int32_t __clip(float val, float min, float max)
{
  float f = val <= min ? min : (val >= max ? max : val);
  return f;
}

static float decode_box(float* x, int len) {
  float sum = 0;
  float y[16];
  
  // softmax
  for (int i = 0; i < len; i++) {
    y[i] = expf(x[i]);
    sum += y[i];
  }
  for (int i = 0; i < len; i++) {
    y[i] = y[i] / sum;
  }
  
  float output = 0;
  for (int i = 0; i < len; i++) {
    output += i * y[i];
  }
  return output;
}

static int process(float* input, int grid_h, int grid_w, int height, int width, int stride,
                   std::vector<float>& boxes, std::vector<float>& points, std::vector<float>& objProbs, std::vector<int>& classId, float threshold)
{
  int    validCount = 0;
  int    grid_len   = grid_h * grid_w;
  float  thres      = unsigmoid(threshold);
  for (int i = 0; i < grid_h; i++) {
    for (int j = 0; j < grid_w; j++) {
      // printf("%d %d %d %d\n", grid_h, grid_w, i, j);
      int8_t prob = input[(i * grid_h + j) * PROP_BOX_SIZE + 64];
      if (prob >= thres) {
        int     offset = (i * grid_h + j) * PROP_BOX_SIZE;
        float   box_x1 = (j + 0.5 - decode_box(&input[offset + 16 * 0], 16)) * (float)stride;
        float   box_y1 = (i + 0.5 - decode_box(&input[offset + 16 * 1], 16)) * (float)stride;
        float   box_x2 = (j + 0.5 + decode_box(&input[offset + 16 * 2], 16)) * (float)stride;
        float   box_y2 = (i + 0.5 + decode_box(&input[offset + 16 * 3], 16)) * (float)stride;
        float   box_w = box_x2 - box_x1;
        float   box_h = box_y2 - box_y1;
        // printf("%f %f %f %f %f\n", box_x1, box_y1, box_x2, box_y2, sigmoid(prob));
        boxes.push_back(box_x1);
        boxes.push_back(box_y1);
        boxes.push_back(box_w);
        boxes.push_back(box_h);
        
        for (int k = 0; k < KEY_POINT_NUM; ++k)
        {
        	float point_x = (input[offset + 65 + 3 * k + 0] * 2 + j) * (float)stride;
        	float point_y = (input[offset + 65 + 3 * k + 1] * 2 + i) * (float)stride;
        	float point_conf = sigmoid(input[offset + 65 + 3 * k + 2]);
        	points.push_back(point_x);
        	points.push_back(point_y);
        	points.push_back(point_conf);
        }
        
        objProbs.push_back(sigmoid(prob));
        classId.push_back(int(0));
        validCount++;
      }
    }
  }
  return validCount;
}

int load_lpd_label()
{
  static int init = -1;
  if (init == -1) {
    int ret = 0;
    ret     = loadLabelName(LABEL_LPD_TXT_PATH, lpd_labels, OBJ_CLASS_NUM);
    if (ret < 0) {
      return -1;
    }
  }
  return 0;
}

int lpd_post_process(float* input0, float* input1, float* input2, int model_in_h, int model_in_w, float conf_threshold,
                 float nms_threshold, float scale_w, float scale_h, detect_result_group_t* group)
{
  memset(group, 0, sizeof(detect_result_group_t));

  std::vector<float> filterBoxes;
  std::vector<float> objProbs;
  std::vector<float> points;
  std::vector<int>   classId;

  // stride 8
  int stride0     = 8;
  int grid_h0     = model_in_h / stride0;
  int grid_w0     = model_in_w / stride0;
  int validCount0 = 0;
  validCount0 = process(input0, grid_h0, grid_w0, model_in_h, model_in_w, stride0, filterBoxes, points, objProbs,
                        classId, conf_threshold);

  // stride 16
  int stride1     = 16;
  int grid_h1     = model_in_h / stride1;
  int grid_w1     = model_in_w / stride1;
  int validCount1 = 0;
  validCount1 = process(input1, grid_h1, grid_w1, model_in_h, model_in_w, stride1, filterBoxes, points, objProbs,
                        classId, conf_threshold);

  // stride 32
  int stride2     = 32;
  int grid_h2     = model_in_h / stride2;
  int grid_w2     = model_in_w / stride2;
  int validCount2 = 0;
  validCount2 = process(input2, grid_h2, grid_w2, model_in_h, model_in_w, stride2, filterBoxes, points, objProbs,
                        classId, conf_threshold);

  int validCount = validCount0 + validCount1 + validCount2;
  // no object detect
  if (validCount <= 0) {
    return 0;
  }

  std::vector<int> indexArray;
  for (int i = 0; i < validCount; ++i) {
    indexArray.push_back(i);
  }

  quick_sort_indice_inverse(objProbs, 0, validCount - 1, indexArray);

  std::set<int> class_set(std::begin(classId), std::end(classId));

  for (auto c : class_set) {
    nms(validCount, filterBoxes, classId, indexArray, c, nms_threshold);
  }

  int last_count = 0;
  group->count   = 0;
  /* box valid detect target */
  for (int i = 0; i < validCount; ++i) {
    if (indexArray[i] == -1 || last_count >= OBJ_NUMB_MAX_SIZE) {
      continue;
    }
    int n = indexArray[i];

    float x1       = filterBoxes[n * 4 + 0];
    float y1       = filterBoxes[n * 4 + 1];
    float x2       = x1 + filterBoxes[n * 4 + 2];
    float y2       = y1 + filterBoxes[n * 4 + 3];
    int   id       = classId[n];
    float obj_conf = objProbs[i];

    group->results[last_count].box.left   = (int)(clamp(x1, 0, model_in_w) / scale_w);
    group->results[last_count].box.top    = (int)(clamp(y1, 0, model_in_h) / scale_h);
    group->results[last_count].box.right  = (int)(clamp(x2, 0, model_in_w) / scale_w);
    group->results[last_count].box.bottom = (int)(clamp(y2, 0, model_in_h) / scale_h);
    
    for (int j = 0; j < KEY_POINT_NUM; ++j)
    {
    	float point_x = points[n * 3 * KEY_POINT_NUM + j * 3 + 0];
    	float point_y = points[n * 3 * KEY_POINT_NUM + j * 3 + 1];
    	float point_conf = points[n * 3 * KEY_POINT_NUM + j * 3 + 2];
    	group->results[last_count].point[j].x = (int)(clamp(point_x, 0, model_in_w) / scale_w);
    	group->results[last_count].point[j].y = (int)(clamp(point_y, 0, model_in_h) / scale_h);
    	group->results[last_count].point[j].conf = point_conf;
    }
    
    group->results[last_count].prop       = obj_conf;
    char* label                           = lpd_labels[id];
    group->results[last_count].name = label;
    last_count++;
  }
  group->count = last_count;

  return 0;
}

int load_lpr_label()
{
  static int init = -1;
  if (init == -1) {
    int ret = 0;
    ret     = loadLabelName(LABEL_LPR_TXT_PATH, lpr_labels, STR_CLASS_NUM);
    if (ret < 0) {
      return -1;
    }
  }
  return 0;
}

int lpr_post_process(float* input0, float conf_threshold, char* result[], int* len)
{
  int class_num = STR_CLASS_NUM;
  int box = 35;
  
  int last_index = class_num;
  int last_count = 0;
  for (int i = 0; i < box; ++i)
  {
  	float max_conf = 0;
  	int index = class_num;
  	for (int j = 0; j < class_num; ++j)
  	{
  		float conf = input0[i * (class_num + 1) + j];
  		if (conf > conf_threshold && conf > max_conf)
  		{
  			max_conf = conf;
  			index = j;
  		}
  	}
  	if (index != class_num && index != last_index)
  	{
  		char* label = lpr_labels[index];
  		result[last_count] = label;
  		//std::wcout << result[last_count] << std::endl;
    		last_count++;
  	}
  	last_index = index;
  }
  *len = last_count;

  return 0;
}

int load_lpc_label()
{
  static int init = -1;
  if (init == -1) {
    int ret = 0;
    ret     = loadLabelName(LABEL_LPC_TXT_PATH, lpc_labels, COLOR_CLASS_NUM);
    if (ret < 0) {
      return -1;
    }
  }
  return 0;
}

int lpc_post_process(float* input0, float conf_threshold, char* result[])
{
  int class_num = COLOR_CLASS_NUM;
  
  float max_conf = 0;
  int index = -1;
  for (int i = 0; i < class_num; ++i)
  {
  	if (input0[i] > conf_threshold && input0[i] > max_conf)
  	{
  		index = i;
  		max_conf = input0[i];
  	}
  }
  result[0] = lpc_labels[index];

  return 0;
}

void deinitPostProcess()
{
  for (int i = 0; i < OBJ_CLASS_NUM; i++) {
    if (lpd_labels[i] != nullptr) {
      free(lpd_labels[i]);
      lpd_labels[i] = nullptr;
    }
  }
  
  for (int i = 0; i < STR_CLASS_NUM; i++) {
    if (lpr_labels[i] != nullptr) {
      free(lpr_labels[i]);
      lpr_labels[i] = nullptr;
    }
  }
  
  for (int i = 0; i < COLOR_CLASS_NUM; i++) {
    if (lpc_labels[i] != nullptr) {
      free(lpc_labels[i]);
      lpc_labels[i] = nullptr;
    }
  }
}
