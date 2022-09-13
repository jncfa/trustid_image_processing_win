#include "trustid_image_processing/face_detector.h"
#include "trustid_image_processing/face_verificator.h"
#include "trustid_image_processing/dlib_impl/face_detector.h"
#include "trustid_image_processing/dlib_impl/face_verificator.h"
#include "trustid_image_processing/client/client_processor.h"

trustid::image::ClientImageProcessor::ClientImageProcessor() : faceDetector(std::make_unique<impl::DlibFaceDetector>()) {}

trustid::image::ClientImageProcessor::ClientImageProcessor(const impl::DlibFaceVerificatorConfig config) : faceDetector(std::make_unique<impl::DlibFaceDetector>()),
                                                                                                                           faceVerificator(std::make_unique<impl::DlibFaceVerificator>(config)) {}

void trustid::image::ClientImageProcessor::loadFaceVerificationData(const impl::DlibFaceVerificatorConfig config)
{
    faceVerificator = std::make_unique<impl::DlibFaceVerificator>(config);
}

trustid::image::FaceDetectionResult trustid::image::ClientImageProcessor::detectFaces(const cv::Mat image)
{
    return faceDetector->detectFaces(image);
}

trustid::image::FaceVerificationResult trustid::image::ClientImageProcessor::verifyUser(const FaceDetectionResultEntry detectionResultEntry)
{
    return faceVerificator->verifyUser(detectionResultEntry);
}

trustid::image::HeadPoseEstimationResult trustid::image::ClientImageProcessor::estimateHeadPose(const FaceDetectionResultEntry detectionResultEntry)
{
    if (faceNormalizer != nullptr)
    {
        return faceNormalizer->estimateHeadPose(detectionResultEntry);
    }
    else
    {
        throw std::runtime_error("Face normalizer not initialized");
    }
}

trustid::image::utils::ImageQualityResultEnum trustid::image::ClientImageProcessor::checkImageQuality(const cv::Mat image)
{
    return utils::checkImageQuality(image);
}

trustid::image::utils::ImageQualityResultEnum trustid::image::ClientImageProcessor::checkImageQuality(const FaceDetectionResultEntry detectionResultEntry)
{
    return utils::checkImageQuality(detectionResultEntry);
}
