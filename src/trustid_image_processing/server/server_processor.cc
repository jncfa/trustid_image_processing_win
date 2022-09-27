#include "trustid_image_processing/server/server_processor.h"

#include "trustid_image_processing/dlib_impl/face_verificator.h"
#include "trustid_image_processing/face_detector.h"
#include "trustid_image_processing/face_normalization.h"
#include "trustid_image_processing/face_verificator.h"
#include "trustid_image_processing/utils.h"

trustid::image::ServerImageProcessor::ServerImageProcessor() {}

std::unique_ptr<trustid::image::impl::DlibFaceVerificator>
trustid::image::ServerImageProcessor::createVerificationModel(
    std::vector<FaceDetectionResultEntry> groundTruthResults) {
  return std::make_unique<impl::DlibFaceVerificator>(groundTruthResults);
}

trustid::image::utils::ImageQualityResultEnum
trustid::image::ServerImageProcessor::checkImageQuality(const cv::Mat image) {
  return utils::checkImageQuality(image);
}

trustid::image::utils::ImageQualityResultEnum
trustid::image::ServerImageProcessor::checkImageQuality(
    const FaceDetectionResultEntry detectionResultEntry) {
  return utils::checkImageQuality(detectionResultEntry);
}