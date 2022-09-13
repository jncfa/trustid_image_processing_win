#include "trustid_image_processing/utils.h"

trustid::image::utils::ImageQualityResultEnum trustid::image::utils::checkImageQuality(const cv::Mat image){
    return ImageQualityResultEnum::UNKNOWN;
}

trustid::image::utils::ImageQualityResultEnum trustid::image::utils::checkImageQuality(const trustid::image::FaceDetectionResultEntry detectionResultEntry){
    return ImageQualityResultEnum::UNKNOWN;
}