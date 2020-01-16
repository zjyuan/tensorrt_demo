#include "utils.h"

using namespace cv;

void ImgPreprocess(cv::Mat &input, int re_width, int re_height, float *data_unifrom, const float3 mean, float normalize, const float3 stdDev, const float scale)
{
    int line_offset;
    int offset_g;
    int offset_b;
    
    unsigned char *line = NULL;
    float *unifrom_data = data_unifrom;

    cv::Mat dst;
    cv::resize(input, dst, cv::Size(re_width, re_height), (0.0), (0.0), cv::INTER_LINEAR);
    offset_g = re_width * re_height;
    offset_b = re_width * re_height * 2;
    for (int i = 0; i < re_height; ++i)
    {
        line = dst.ptr<uchar>(i);
        line_offset = i * re_width;
        for(int j = 0; j < re_width; ++j)
        {
            // r
            unifrom_data[line_offset + j] = float(line[j * 3 + 2]/normalize - mean.x)/stdDev.x;
            // g
            unifrom_data[offset_g + line_offset + j] = float(line[j * 3 + 1]/normalize - mean.y)/stdDev.y;
            // b
            unifrom_data[offset_b + line_offset + j] = float(line[j * 3]/normalize - mean.z)/stdDev.z;
        }
    }  
}

std::vector<NvDsInferParseObjectInfo>
nonMaximumSuppression(const float nmsThresh, std::vector<NvDsInferParseObjectInfo> binfo)
{
    auto overlap1D = [](float x1min, float x1max, float x2min, float x2max) -> float {
        if (x1min > x2min)
        {
            std::swap(x1min, x2min);
            std::swap(x1max, x2max);
        }
        return x1max < x2min ? 0 : std::min(x1max, x2max) - x2min;
    };
    auto computeIoU
        = [&overlap1D](NvDsInferParseObjectInfo& bbox1, NvDsInferParseObjectInfo& bbox2) -> float {
        float overlapX
            = overlap1D(bbox1.left, bbox1.left + bbox1.width, bbox2.left, bbox2.left + bbox2.width);
        float overlapY
            = overlap1D(bbox1.top, bbox1.top + bbox1.height, bbox2.top, bbox2.top + bbox2.height);
        float area1 = (bbox1.width) * (bbox1.height);
        float area2 = (bbox2.width) * (bbox2.height);
        float overlap2D = overlapX * overlapY;
        float u = area1 + area2 - overlap2D;
        return u == 0 ? 0 : overlap2D / u;
    };

    std::stable_sort(binfo.begin(), binfo.end(),
                     [](const NvDsInferParseObjectInfo& b1, const NvDsInferParseObjectInfo& b2) {
                         return b1.detectionConfidence > b2.detectionConfidence;
                     });
    std::vector<NvDsInferParseObjectInfo> out;
    for (auto i : binfo)
    {
        bool keep = true;
        for (auto j : out)
        {
            if (keep)
            {
                float overlap = computeIoU(i, j);
                keep = overlap <= nmsThresh;
            }
            else
                break;
        }
        if (keep) out.push_back(i);
    }
    return out;
}

std::vector<NvDsInferParseObjectInfo>
nmsAllClasses(const float nmsThresh,
        std::vector<NvDsInferParseObjectInfo>& binfo,
        const uint numClasses)
{
    std::vector<NvDsInferParseObjectInfo> result;
    std::vector<std::vector<NvDsInferParseObjectInfo>> splitBoxes(numClasses);
    for (auto& box : binfo)
    {
        splitBoxes.at(box.classId).push_back(box);
    }

    for (auto& boxes : splitBoxes)
    {
        boxes = nonMaximumSuppression(nmsThresh, boxes);
        result.insert(result.end(), boxes.begin(), boxes.end());
    }
    return result;
}

void draw_bbox(cv::Mat& m, std::vector<NvDsInferParseObjectInfo>& objectList)
{
    for(auto &b:objectList)
    {
        if(b.classId == 0)
            cv::rectangle(m, Point(b.left, b.top), Point(b.left+b.width, b.top+b.height), cv::Scalar(0, 0, 255), 2, 8, 0);
        else
            cv::rectangle(m, Point(b.left, b.top), Point(b.left+b.width, b.top+b.height), cv::Scalar(0, 255, 0), 2, 8, 0);
    }
}

extern std::vector<std::vector<float>> prior_data;
std::vector<NvDsInferParseObjectInfo> post_process(float* loc_data, float* conf_data)
{
    std::vector<NvDsInferParseObjectInfo> objects;
    int box_num = 43485;
    for(int i=0; i<box_num; i++)
    {
        float *box_conf_data = conf_data + 3*i;
        float *box_loc_data = loc_data + 4*i;

        float total_conf = (exp(*(box_conf_data + 0)) + exp(*(box_conf_data + 1)) + exp(*(box_conf_data + 2)));
        float body_conf = exp(*(box_conf_data + 2))/total_conf;
        float face_conf = exp(*(box_conf_data + 1))/total_conf;

        if(body_conf > 0.81)
        {
            box_loc_data[0] = box_loc_data[0]*0.1*prior_data[i][2]+prior_data[i][0];
            box_loc_data[1] = box_loc_data[1]*0.1*prior_data[i][3]+prior_data[i][1];
            box_loc_data[2] = exp(box_loc_data[2]*0.2)*prior_data[i][2];
            box_loc_data[3] = exp(box_loc_data[3]*0.2)*prior_data[i][3];

            box_loc_data[0] -= box_loc_data[2]/2;
            box_loc_data[1] -= box_loc_data[3]/2;
            box_loc_data[2] += box_loc_data[0];
            box_loc_data[3] += box_loc_data[1];

            box_loc_data[0] *= 1920;
            box_loc_data[1] *= 1080;
            box_loc_data[2] *= 1920;
            box_loc_data[3] *= 1080;

            NvDsInferParseObjectInfo b;

            b.classId = 0;
            b.left = box_loc_data[0];
            b.top = box_loc_data[1];                   //opengl坐标系原点在左上
            b.width = box_loc_data[2]-box_loc_data[0];
            b.height = box_loc_data[3]-box_loc_data[1];
            b.detectionConfidence = body_conf;
            // std::cout << b.left << " " << b.top << " " << b.width << " " << b.height << " " << b.detectionConfidence << std::endl;

            b.left = clamp(b.left, 0, 1920);
            b.width = clamp(b.width, 0, 1920);
            b.top = clamp(b.top, 0, 1080);
            b.height = clamp(b.height, 0, 1080);

            objects.push_back(b);
        }

        if(face_conf > 0.75)
        {
            box_loc_data[0] = box_loc_data[0]*0.1*prior_data[i][2]+prior_data[i][0];
            box_loc_data[1] = box_loc_data[1]*0.1*prior_data[i][3]+prior_data[i][1];
            box_loc_data[2] = exp(box_loc_data[2]*0.2)*prior_data[i][2];
            box_loc_data[3] = exp(box_loc_data[3]*0.2)*prior_data[i][3];

            box_loc_data[0] -= box_loc_data[2]/2;
            box_loc_data[1] -= box_loc_data[3]/2;
            box_loc_data[2] += box_loc_data[0];
            box_loc_data[3] += box_loc_data[1];

            box_loc_data[0] *= 1920;
            box_loc_data[1] *= 1080;
            box_loc_data[2] *= 1920;
            box_loc_data[3] *= 1080;

            NvDsInferParseObjectInfo b;

            b.classId = 1;
            b.left = box_loc_data[0];
            b.top = box_loc_data[1];                   //opengl坐标系原点在左上
            b.width = box_loc_data[2]-box_loc_data[0];
            b.height = box_loc_data[3]-box_loc_data[1];
            b.detectionConfidence = face_conf;
            // std::cout << b.left << " " << b.top << " " << b.width << " " << b.height << " " << b.detectionConfidence << std::endl;

            b.left = clamp(b.left, 0, 1920);
            b.width = clamp(b.width, 0, 1920);
            b.top = clamp(b.top, 0, 1080);
            b.height = clamp(b.height, 0, 1080);
            // std::cout << b.left << " " << b.top << " " << b.width << " " << b.height << " " << b.detectionConfidence << std::endl;

            objects.push_back(b);
        }
    }

    return nmsAllClasses(0.4, objects, 2); 
}