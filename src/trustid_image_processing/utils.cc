#include "trustid_image_processing/utils.h"
#include <opencv2/opencv.hpp>
#include <dlib/geometry.h>

trustid::image::utils::ImageQualityResultEnum trustid::image::utils::checkImageQuality(const cv::Mat image){
    return ImageQualityResultEnum::UNKNOWN;
}

trustid::image::utils::ImageQualityResultEnum trustid::image::utils::checkImageQuality(const trustid::image::FaceDetectionResultEntry detectionResultEntry){
    return ImageQualityResultEnum::UNKNOWN;
}

cv::Rect trustid::image::utils::dlibRectangleToOpenCV(const dlib::rectangle r)
{
  return cv::Rect(cv::Point2i(r.left(), r.top()), cv::Point2i(r.right() + 1, r.bottom() + 1));
}

dlib::rectangle trustid::image::utils::openCVRectToDlib(const cv::Rect r)
{
  return dlib::rectangle((long)r.tl().x, (long)r.tl().y, (long)r.br().x - 1, (long)r.br().y - 1);
}
