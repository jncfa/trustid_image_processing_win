#ifndef TRUSTID_CLIENT_IMAGE_PROCESSING_PIPELINE_H_
#define TRUSTID_CLIENT_IMAGE_PROCESSING_PIPELINE_H_

#include <memory>

#include "trustid_image_processing/dlib_impl/face_detector.h"
#include "trustid_image_processing/dlib_impl/face_verificator.h"
#include "trustid_image_processing/face_detector.h"
#include "trustid_image_processing/face_normalization.h"
#include "trustid_image_processing/face_verificator.h"
#include "trustid_image_processing/utils.h"

namespace trustid {
namespace image {
class ClientImageProcessor {
 public:
  /**
   * Default constructor. Requires the face verification data to properly work.
   */
  ClientImageProcessor(const impl::DlibFaceVerificatorConfig config);

  /**
   * Empty constructor, to allow instances where the data is loaded after
   * instancing the class.
   */
  ClientImageProcessor();

  /**
   * Loads the face verificator according to the supplied configuration.
   */
  void loadFaceVerificationData(const impl::DlibFaceVerificatorConfig config);

  /**
   * Check if the face verificator is loaded.
   */
  bool canVerifyFaces();

  /**
   * Detect any faces in the given image.
   */
  FaceDetectionResult detectFaces(const cv::Mat image);

  /**
   * Verify if the detected face matches the currently loaded identity.
   */
  FaceVerificationResult verifyUser(
      const FaceDetectionResultEntry detectionResultEntry);

  /**
   * Estimate the head pose of detected face.
   */
  HeadPoseEstimationResult estimateHeadPose(
      const FaceDetectionResultEntry detectionResultEntry);

  /**
   * Check the constrast of a given image.
   */
  static utils::ImageQualityResultEnum checkImageQuality(const cv::Mat image);

  /**
   * Check the constrast of a given face image.
   */
  static utils::ImageQualityResultEnum checkImageQuality(
      const FaceDetectionResultEntry detectionResultEntry);

 private:
  std::unique_ptr<impl::DlibFaceDetector> faceDetector;
  std::unique_ptr<impl::DlibFaceVerificator> faceVerificator;
  std::unique_ptr<impl::FaceNormalizer> faceNormalizer;
};
}  // namespace image
}  // namespace trustid
#endif  // TRUSTID_CLIENT_IMAGE_PROCESSING_PIPELINE_H_