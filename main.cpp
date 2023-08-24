
#include "detect/inference.h"

int main(){
    BuffDetector buff;
    buff.initModel("../model/buff2.xml");
    // Step 3. Read input image
    cv::VideoCapture cap;
    cap.open( "../test.mp4");
    cv::Mat img;

    while(1)
    {
        cap>>img;
        std::vector<Detection> output;
        buff.detect(img,output);
        cv::waitKey(1);
    }

}







//#include "detect/inference.h"
//int main() {
//    cv::VideoCapture cap;
//    cap.open( "../test.mp4");
//    cv::Mat frame;
//    while(1)
//    {
//        cap >> frame;
//        vector<BuffObject> objects;
//        vector<Mat> fans;
//        auto input = frame;
//        string name[6] = {"BR","BU","BA","RR","RU","RA"};
//
//        BuffDetector b;
//        b.initModel("../model/buff2.xml");
//        b.detect(input,objects);
//        for(auto object: objects)
//        {
//            auto cl = object.color*3 + object.cls;
//            string n = name[cl];
//            line(input, object.apex[0], object.apex[1], Scalar(0, 255, 0), 2);
//            line(input, object.apex[1], object.apex[2], Scalar(0, 255, 0), 2);
//            line(input, object.apex[2], object.apex[3], Scalar(0, 255, 0), 2);
//            line(input, object.apex[3], object.apex[4], Scalar(0, 255, 0), 2);
//            line(input, object.apex[4], object.apex[0], Scalar(0, 255, 0), 2);
//            putText(input, n, object.apex[0], 1, 1, Scalar(0, 255, 0));
//        }
//        namedWindow("network_input",0);
//        imshow("network_input",input);
//        waitKey(1);
//    }
//
//    return 0;
//}
