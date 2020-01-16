#pragma once

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

typedef struct
{
  /** ID of the class to which the object belongs. */
  unsigned int classId;

  /** Horizontal offset of the bounding box shape for the object. */
  unsigned int left;
  /** Vertical offset of the bounding box shape for the object. */
  unsigned int top;
  /** Width of the bounding box shape for the object. */
  unsigned int width;
  /** Height of the bounding box shape for the object. */
  unsigned int height;

  /** Object detection confidence. Should be a float value in the range [0,1] */
  float detectionConfidence;
} NvDsInferParseObjectInfo;

inline unsigned clamp(const int val, const int minVal, const int maxVal)
{
    assert(minVal <= maxVal);
    return std::min(maxVal, std::max(minVal, val));
}

std::vector<NvDsInferParseObjectInfo>
nonMaximumSuppression(const float nmsThresh, std::vector<NvDsInferParseObjectInfo> binfo);

std::vector<NvDsInferParseObjectInfo>
nmsAllClasses(const float nmsThresh,
        std::vector<NvDsInferParseObjectInfo>& binfo,
        const uint numClasses);

void draw_bbox(cv::Mat& m, std::vector<NvDsInferParseObjectInfo>& objectList);

//std::vector supports move semantics
std::vector<NvDsInferParseObjectInfo> post_process(float* loc_data, float* conf_data);

//hwc->chw, bgr->rgb, mean value subtraction, normalization
void ImgPreprocess(cv::Mat &input, int re_width, int re_height, float *data_unifrom, const float3 mean, float normalize, const float3 stdDev, const float scale);