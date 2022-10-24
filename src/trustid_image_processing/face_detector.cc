#include "trustid_image_processing/face_detector.h"

#include <stdio.h>

#include <algorithm>
#include <iostream>
#include <memory>
#include <opencv2/core.hpp>
#include <stdexcept>
#include <vector>

void trustid::image::serialize(const FaceDetectionResultValueEnum& item,
                               std::ostream& out) {
  dlib::serialize(static_cast<int>(item), out);
}

void trustid::image::deserialize(FaceDetectionResultValueEnum& item,
                                 std::istream& in) {
  int value;
  dlib::deserialize(value, in);
  item = static_cast<FaceDetectionResultValueEnum>(value);
}
trustid::image::FaceDetectionResultEntry::FaceDetectionResultEntry(){}

trustid::image::FaceDetectionResultEntry::FaceDetectionResultEntry(
    cv::Mat image,
    trustid::image::FaceDetectionConfidenceBoundingBox detectionBoundingbox)
    : detectionBoundingbox(detectionBoundingbox), image(image.clone()) {}

cv::Mat trustid::image::FaceDetectionResultEntry::getImage() const {
  return image;
}

cv::Rect trustid::image::FaceDetectionResultEntry::getBoundingBox() const {
  return detectionBoundingbox.boundingBox;
}

trustid::image::FaceDetectionConfidenceBoundingBox
trustid::image::FaceDetectionResultEntry::getFaceDetBoundingBox() const {
  return detectionBoundingbox;
}

cv::Mat trustid::image::FaceDetectionResultEntry::getCroppedImage() const {
  if (image.empty()) {
    throw std::runtime_error("Image is empty");
  } else {
    return image(detectionBoundingbox.boundingBox);
  }
}

trustid::image::FaceDetectionResultEntry
trustid::image::FaceDetectionResultEntry::copy() const {
  return FaceDetectionResultEntry(image, detectionBoundingbox);
}
trustid::image::FaceDetectionResult::FaceDetectionResult() {}
trustid::image::FaceDetectionResult::FaceDetectionResult(
    cv::Mat image,
    std::vector<FaceDetectionConfidenceBoundingBox> boundingBoxes)
    : image(image), boundingBoxes(boundingBoxes) {
  if (boundingBoxes.size() == 0) {
    resultValue = NO_RESULTS;
  } else if (boundingBoxes.size() == 1) {
    resultValue = ONE_RESULT;
  } else {
    resultValue = MULTIPLE_RESULTS;
  }
}
trustid::image::FaceDetectionResult::FaceDetectionResult(
    cv::Mat image,
    std::vector<FaceDetectionConfidenceBoundingBox> boundingBoxes,
    FaceDetectionResultValueEnum resultValue)
    : image(image), boundingBoxes(boundingBoxes), resultValue(resultValue) {}

/**
 * Returns the result value of the face detection operation.
 */
trustid::image::FaceDetectionResultValueEnum
trustid::image::FaceDetectionResult::getResult() const {
  return resultValue;
}

/**
 * Get the image that was used for the face detection operation.
 */
cv::Mat trustid::image::FaceDetectionResult::getImage() const { return image; }

/**
 * Returns the bounding box of the face detection operation according to the
 * specified heuristic.
 */
cv::Rect trustid::image::FaceDetectionResult::getBoundingBox(
    int detectionIdx, BoundingBoxHeuristicEnum heuristic) const {
  if (boundingBoxes.size() == 0) {
    throw std::runtime_error("No bounding boxes found");
  } else if (boundingBoxes.size() == 1) {
    return boundingBoxes[detectionIdx].boundingBox;
  } else {
    // sends the first bounding box according to euristic
    return getBoundingBoxes(heuristic)[detectionIdx].boundingBox;
  }
}

/**
 * Returns the cropped face image of the face selected using the specified
 * heuristic.
 */
cv::Mat trustid::image::FaceDetectionResult::getCroppedImage(
    int detectionIdx, BoundingBoxHeuristicEnum heuristic) const {
  if (image.empty()) {
    throw std::runtime_error("Image is empty");
  } else {
    return image(getBoundingBox(detectionIdx, heuristic));
  }
}

/**
 * Returns the bounding boxes of the face detection operation, sorted according
 * to the specified heuristic.
 */
std::vector<trustid::image::FaceDetectionConfidenceBoundingBox>
trustid::image::FaceDetectionResult::getBoundingBoxes(
    BoundingBoxHeuristicEnum heuristic) const {
  // sort data according to heuristic
  std::vector<FaceDetectionConfidenceBoundingBox> sortedBoundingBoxes =
      boundingBoxes;
  std::sort(sortedBoundingBoxes.begin(), sortedBoundingBoxes.end(),
            [heuristic](FaceDetectionConfidenceBoundingBox a,
                        FaceDetectionConfidenceBoundingBox b) {
              switch (heuristic) {
                case LARGEST_AREA:
                  return a.boundingBox.area() > b.boundingBox.area();
                default:
                  throw std::runtime_error("Unknown heuristic");
              }
            });

  return sortedBoundingBoxes;
}

std::vector<trustid::image::FaceDetectionResultEntry>
trustid::image::FaceDetectionResult::getBoundingBoxEntries(
    BoundingBoxHeuristicEnum heuristic) const {
  // sort data according to heuristic
  std::vector<FaceDetectionConfidenceBoundingBox> sortedBoundingBoxes =
      getBoundingBoxes(heuristic);

  auto entries = std::vector<FaceDetectionResultEntry>();
  for (auto& boundingBox : sortedBoundingBoxes) {
    entries.push_back(FaceDetectionResultEntry(image, boundingBox));
  }
  return entries;
}

trustid::image::FaceDetectionResultEntry
trustid::image::FaceDetectionResult::getEntry(
    int detectionIdx, BoundingBoxHeuristicEnum heuristic) const {
  if (detectionIdx < boundingBoxes.size()) {
    // TODO: Should this clone the image?
    return FaceDetectionResultEntry(image,
                                    getBoundingBoxes(heuristic)[detectionIdx]);
  } else {
    throw std::runtime_error("Invalid detection index");
  }
}

trustid::image::FaceDetectionResult trustid::image::FaceDetectionResult::copy()
    const {
  return FaceDetectionResult(image, boundingBoxes, resultValue);
}

trustid::image::IFaceDetectImageProcessor::IFaceDetectImageProcessor() {}

trustid::image::IFaceDetector::IFaceDetector() : imagePreprocessors() {}

trustid::image::IFaceDetector::~IFaceDetector() {}

trustid::image::FaceDetectionResult trustid::image::IFaceDetector::detectFaces(
    const cv::Mat image) {
  cv::Mat imageCopy = image.clone();
  for (auto& processor : imagePreprocessors) {
    if (processor != nullptr) {
      imageCopy = processor->operator()(imageCopy);
    }
  }
  return _detectFaces(imageCopy);
}

void trustid::image::IFaceDetector::addPreprocessor(
    std::unique_ptr<IFaceDetectImageProcessor> preprocessor) {
  imagePreprocessors.push_back(std::move(preprocessor));
}