//
// Created by shaobing2 on 8/24/23.
//

#ifndef WIND_INFERENCE_INFERENCE_H
#define WIND_INFERENCE_INFERENCE_H

#include <opencv2/dnn.hpp>
#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>
using namespace std;

struct Detection
{
    int cls;
    int color;
    float confidence;
    cv::Point2f ppts[5];
    cv::Rect rect;
};

struct Resize
{
    cv::Mat resized_image;
    int dw;
    int dh;
};

class BuffDetector
{
public:

    bool detect(cv::Mat &src,vector<Detection>& output);
    bool initModel(string path);

private:
    ov::Core core;
    std::shared_ptr<ov::Model> model;
    ov::CompiledModel compiled_model;
    ov::InferRequest infer_request;
};


#endif //WIND_INFERENCE_INFERENCE_H
