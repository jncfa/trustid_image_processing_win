#include "trustid_image_processing/utils.h"
#include <opencv2/opencv.hpp>
#include <dlib/geometry.h>

trustid::image::utils::ImageQualityResultEnum trustid::image::utils::checkImageQuality(const cv::Mat image){
    //convert image to grayscale
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    // compute histogram of grayscale image
    cv::Mat hist;
    int histSize = 256;
    float range[] = { 0, 256 };
    const float* histRange = { range };
    bool uniform = true, accumulate = false;
    cv::calcHist(&gray, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);

    //normalize histogram
    cv::normalize(hist, hist, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());

    //std::partial_sum(hist.begin(), hist.end(), Sum.begin(), std::plus<double>());
    
    
    return ImageQualityResultEnum::UNKNOWN;
}

trustid::image::utils::ImageQualityResultEnum trustid::image::utils::checkImageQuality(const trustid::image::FaceDetectionResultEntry detectionResultEntry){
    return trustid::image::utils::checkImageQuality(detectionResultEntry.getCroppedImage());
}

cv::Rect trustid::image::utils::dlibRectangleToOpenCV(const dlib::rectangle r)
{
  return cv::Rect(cv::Point2i(r.left(), r.top()), cv::Point2i(r.right() + 1, r.bottom() + 1));
}

dlib::rectangle trustid::image::utils::openCVRectToDlib(const cv::Rect r)
{
  return dlib::rectangle((long)r.tl().x, (long)r.tl().y, (long)r.br().x - 1, (long)r.br().y - 1);
}
