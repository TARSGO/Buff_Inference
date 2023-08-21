//
// Created by shaobing2 on 8/20/23.
//

#ifndef WIND_INFERENCE_INFERENCE_H
#define WIND_INFERENCE_INFERENCE_H

#include <iterator>
#include <memory>
#include <string>
#include <vector>
#include <iostream>

#include <inference_engine.hpp>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>


using namespace std;
using namespace cv;
using namespace InferenceEngine;

struct alignas(4) bbox_t {
    cv::Point2f pts[5]; // [pt0, pt1, pt2, pt3]
    float confidence;
    int color_id; // 0: blue, 1: red, 2: gray
    int tag_id;   // 0: guard, 1-5: number, 6: base

    bool operator==(const bbox_t&) const = default;
    bool operator!=(const bbox_t&) const = default;
};

struct BuffObject
{
    Point2f apex[5];
    cv::Rect_<float> rect;
    int cls;
    int color;
    float prob;
    std::vector<cv::Point2f> pts;
};

class BuffDetector
{

private:
    Core ie;
    CNNNetwork network;                // 网络
    ExecutableNetwork executable_network;       // 可执行网络
    InferRequest infer_request;      // 推理请求
    MemoryBlob::CPtr moutput;
    string input_name;
    string output_name;

    Eigen::Matrix<float,3,3> transfrom_matrix;

public:
    BuffDetector();
    ~BuffDetector();

    bool detect(Mat &src,vector<BuffObject>& objects);
    bool initModel(string path);

};
struct GridAndStride
{
    int grid0;
    int grid1;
    int stride;
};

#endif //WIND_INFERENCE_INFERENCE_H
