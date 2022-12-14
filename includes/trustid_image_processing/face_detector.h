#ifndef TRUSTID_FACE_DETECTOR_H_
#define TRUSTID_FACE_DETECTOR_H_

#include <dlib/serialize.h>
#include <stdio.h>

#include <algorithm>
#include <iostream>
#include <memory>
#include <opencv2/core.hpp>
#include <stdexcept>
#include <vector>

#include "serialize.h"

namespace trustid {
namespace image {
/**
 * Result of a given face detection operation.
 */
enum FaceDetectionResultValueEnum {
  NO_RESULTS,
  ONE_RESULT,
  MULTIPLE_RESULTS,
};

void serialize(const FaceDetectionResultValueEnum& item, std::ostream& out);
void deserialize(FaceDetectionResultValueEnum& item, std::istream& in);

/**
 * List of possible heuristics to select a bounding box through the
 * FaceDetectionResult::getBoundingBox procedure.
 */
enum BoundingBoxHeuristicEnum { LARGEST_AREA };

/**
 * Stores information regarding a single detection for a face detection
 * operation.
 */
struct FaceDetectionConfidenceBoundingBox {
  cv::Rect boundingBox;
  double confidenceScore;
  DLIB_DEFINE_DEFAULT_SERIALIZATION(FaceDetectionConfidenceBoundingBox,
                                    boundingBox, confidenceScore);
};

class FaceDetectionResultEntry {
 public:
  FaceDetectionResultEntry();
  FaceDetectionResultEntry(
      cv::Mat image, FaceDetectionConfidenceBoundingBox detectionBoundingbox);

  /**
   * Get the image that was used for the face detection operation.
   */
  cv::Mat getImage() const;

  /**
   * Returns the bounding box of the face detection operation according to the
   * specified heuristic.
   */
  cv::Rect getBoundingBox() const;

  /**
   * Returns the bounding box (with the confidence value) of the face detection
   * operation according to the specified heuristic.
   */
  FaceDetectionConfidenceBoundingBox getFaceDetBoundingBox() const;

  /**
   * Returns the cropped face image of the face selected using the specified
   * heuristic.
   */
  cv::Mat getCroppedImage() const;

  /**
   * Retuns a copy of the current FaceDetectionResultEntry with the original
   * image cropped to save memory.
   */
  FaceDetectionResultEntry applyCrop() const;

  /**
   * Returns a copy of the current FaceDetectionResultEntry.
   */
  FaceDetectionResultEntry copy() const;

 private:
  cv::Mat image;
  FaceDetectionConfidenceBoundingBox
      detectionBoundingbox;  // bounding box of the detected face
  DLIB_DEFINE_DEFAULT_SERIALIZATION(FaceDetectionResultEntry, image,
                                    detectionBoundingbox);
};

/**
 * Stores information regarding a face detection operation.
 */
class FaceDetectionResult {
 public:
  FaceDetectionResult();

  FaceDetectionResult(
      cv::Mat image,
      std::vector<FaceDetectionConfidenceBoundingBox> boundingBoxes);

  FaceDetectionResult(
      cv::Mat image,
      std::vector<FaceDetectionConfidenceBoundingBox> boundingBoxes,
      FaceDetectionResultValueEnum resultValue);

  /**
   * Returns the result value of the face detection operation.
   */
  FaceDetectionResultValueEnum getResult() const;

  /**
   * Get the image that was used for the face detection operation.
   */
  cv::Mat getImage() const;

  /**
   * Returns the bounding box of the face detection operation according to the
   * specified heuristic.
   */
  cv::Rect getBoundingBox(
      int detectionIdx = 0,
      BoundingBoxHeuristicEnum heuristic = LARGEST_AREA) const;

  /**
   * Returns the cropped face image of the face selected using the specified
   * heuristic.
   */
  cv::Mat getCroppedImage(
      int detectionIdx = 0,
      BoundingBoxHeuristicEnum heuristic = LARGEST_AREA) const;

  /**
   * Returns the bounding boxes of the face detection operation, sorted
   * according to the specified heuristic.
   */
  std::vector<FaceDetectionConfidenceBoundingBox> getBoundingBoxes(
      BoundingBoxHeuristicEnum heuristic = LARGEST_AREA) const;

  std::vector<FaceDetectionResultEntry> getBoundingBoxEntries(
      BoundingBoxHeuristicEnum heuristic = LARGEST_AREA) const;

  FaceDetectionResultEntry getEntry(
      int detectionIdx = 0,
      BoundingBoxHeuristicEnum heuristic = LARGEST_AREA) const;

  FaceDetectionResult copy() const;

 private:
  FaceDetectionResultValueEnum resultValue;
  cv::Mat image;
  std::vector<FaceDetectionConfidenceBoundingBox> boundingBoxes;
  
  DLIB_DEFINE_DEFAULT_SERIALIZATION(FaceDetectionResult, resultValue, image,
                                    boundingBoxes);
};

/**
 * Class to implement a processing pipeline for images.
 */
class IFaceDetectImageProcessor {
 public:
  IFaceDetectImageProcessor();
  virtual cv::Mat operator()(const cv::Mat image) = 0;
};

/**
 * Detects faces in an image.
 */
class IFaceDetector {
 public:
  IFaceDetector();

  virtual ~IFaceDetector();

  /**
   * Detects faces in an image.
   */
  FaceDetectionResult detectFaces(const cv::Mat image);

  void addPreprocessor(std::unique_ptr<IFaceDetectImageProcessor> preprocessor);

 protected:
  /**
   * Internal procedure for detecting faces in an image.
   * This is split up from the public detectFaces procedure to allow the
   * implementation of pre and/or post-processing steps common to every face
   * detection algorithm.
   */
  virtual FaceDetectionResult _detectFaces(const cv::Mat image) = 0;
  std::vector<std::unique_ptr<IFaceDetectImageProcessor>> imagePreprocessors;
};
}  // namespace image
}  // namespace trustid
#endif  // TRUSTID_FACE_DETECTOR_H_
