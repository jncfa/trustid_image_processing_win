
#include "trustid_image_processing/face_verificator.h"

#include <opencv2/opencv.hpp>
#include <utility>
#include <vector>

trustid::image::FaceVerificationResult::FaceVerificationResult(
    FaceDetectionResultEntry detectionResultEntry, double matchConfidence,
    FaceVerificationResultEnum resultValue)
    : detectionResultEntry(detectionResultEntry),
      matchConfidence(matchConfidence),
      resultValue(resultValue) {}

/**
 * Returns the confidence score of the match of the given face image to a
 * certain user.
 */
double trustid::image::FaceVerificationResult::getMatchConfidence() const {
  return matchConfidence;
}

/**
 * Returns the result of the face verification operation.
 */
trustid::image::FaceVerificationResultEnum
trustid::image::FaceVerificationResult::getResult() const {
  return resultValue;
}

/**
 * Returns the face detection result for the given face verification operation.
 */
trustid::image::FaceDetectionResultEntry
trustid::image::FaceVerificationResult::getDetectionResult() const {
  return detectionResultEntry;
}

trustid::image::IFaceVerifyImageProcessor::IFaceVerifyImageProcessor() {}

trustid::image::ResizeImageProcessor::ResizeImageProcessor(int width,
                                                           int height)
    : width(width), height(height) {}

trustid::image::FaceDetectionResultEntry
trustid::image::ResizeImageProcessor::operator()(
    const FaceDetectionResultEntry detectionResultEntry) {
  // Resize the image to the specified size
  // Here we do a little trick, and instead of resizing the image to the
  // specified size, we set the bounding box to the size, and then resize the
  // image to keep the image ratio to the bounding box
  cv::Mat originalImage = detectionResultEntry.getImage();
  auto originalBoundingBox = detectionResultEntry.getFaceDetBoundingBox();

  // we need to update the x,y coordinates to the correct position in the
  // resized image
  auto newDetectionBoundingBox = FaceDetectionConfidenceBoundingBox{
      cv::Rect(0, 0, width, height), originalBoundingBox.confidenceScore};
  newDetectionBoundingBox.boundingBox.x =
      (int)(originalBoundingBox.boundingBox.x * width /
            (double)originalBoundingBox.boundingBox.width);
  newDetectionBoundingBox.boundingBox.y =
      (int)(originalBoundingBox.boundingBox.y * height /
            (double)originalBoundingBox.boundingBox.height);

  std::cout << "original image size:" << (int)(originalImage.cols) << "|"
            << (int)(originalImage.rows) << std::endl;
  std::cout << "reshape factor:"
            << width / (double)originalBoundingBox.boundingBox.width << "|"
            << height / (double)originalBoundingBox.boundingBox.height
            << std::endl;
  std::cout << "resized image size:"
            << (int)(originalImage.cols * width /
                     (double)originalBoundingBox.boundingBox.width)
            << "|"
            << (int)(originalImage.rows * height /
                     (double)originalBoundingBox.boundingBox.height)
            << std::endl;

  cv::Mat resizedImage;
  cv::resize(originalImage, resizedImage, cv::Size(),
             width / (double)originalBoundingBox.boundingBox.width,
             height / (double)originalBoundingBox.boundingBox.height,
             cv::INTER_LINEAR);

  // reshape bounding box
  return FaceDetectionResultEntry(resizedImage, newDetectionBoundingBox);
}

trustid::image::CropImageProcessor::CropImageProcessor(cv::Rect cropInfo)
    : cropInfo(cropInfo) {}

trustid::image::FaceDetectionResultEntry
trustid::image::CropImageProcessor::operator()(
    const FaceDetectionResultEntry detectionResultEntry) {
  // Crop the image to the specified size
  cv::Mat image = detectionResultEntry.getImage();
  auto cropBoundingBox = detectionResultEntry.getFaceDetBoundingBox();

  // TODO: Add a check when the crop rectangle is outside the image
  cropBoundingBox.boundingBox = cv::Rect(
      cropBoundingBox.boundingBox.x - cropInfo.x,
      cropBoundingBox.boundingBox.y - cropInfo.y,
      cropBoundingBox.boundingBox.width, cropBoundingBox.boundingBox.height);

  return FaceDetectionResultEntry(image(cropInfo), cropBoundingBox);
}

trustid::image::IFaceVerificator::IFaceVerificator() : preprocessors() {}

trustid::image::IFaceVerificator::IFaceVerificator(
    std::vector<std::unique_ptr<IFaceVerifyImageProcessor>> preprocessors)
    : preprocessors(std::move(preprocessors)) {}

void trustid::image::IFaceVerificator::addPreprocessor(
    std::unique_ptr<IFaceVerifyImageProcessor> preprocessor) {
  preprocessors.push_back(std::move(preprocessor));
}

void trustid::image::IFaceVerificator::removePreprocessors() {
  preprocessors.clear();
}

trustid::image::FaceVerificationResult
trustid::image::IFaceVerificator::verifyUser(
    const FaceDetectionResultEntry detectionResultEntry) {
  return _verifyUser(applyProcessors(detectionResultEntry));
}

trustid::image::FaceDetectionResultEntry
trustid::image::IFaceVerificator::applyProcessors(
    const FaceDetectionResultEntry detectionResultEntry) {
  FaceDetectionResultEntry detectionResultEntryCopy =
      detectionResultEntry.copy();
  for (auto &processor : preprocessors) {
    detectionResultEntryCopy = processor->operator()(detectionResultEntryCopy);
  }
  return detectionResultEntryCopy;
}
