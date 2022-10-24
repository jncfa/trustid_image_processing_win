#include "trustid_image_processing/face_verificator.h"

#include <dlib/dnn.h>
#include <dlib/opencv.h>

#include <iostream>
#include <istream>
#include <memory>
#include <vector>

#include "trustid_image_processing/dlib_impl/face_verificator.h"
#include "trustid_image_processing/utils.h"

trustid::image::impl::DlibFaceChipExtractor::DlibFaceChipExtractor(
    dlib::shape_predictor sp)
    : sp(sp) {}

trustid::image::FaceDetectionResultEntry
trustid::image::impl::DlibFaceChipExtractor::operator()(
    const FaceDetectionResultEntry detectionResultEntry) {
  // get the bounding box of the face
  auto detection = detectionResultEntry.getBoundingBox();
  auto image = detectionResultEntry.getImage();

  // convert it to dlib objects
  dlib::cv_image<dlib::bgr_pixel> dlibImage(image);
  auto shape = sp(dlibImage, utils::openCVRectToDlib(detection));

  // extract the face chip
  dlib::array2d<dlib::bgr_pixel> face_chip;
  dlib::extract_image_chip(
      dlibImage, dlib::get_face_chip_details(shape, 150, 0.25), face_chip);

  // convert it back to OpenCV format
  cv::Mat face_chip_mat = dlib::toMat(face_chip);
  dlib::matrix<dlib::rgb_pixel> matrix;

  // return the new result
  return FaceDetectionResultEntry(
      face_chip_mat,
      FaceDetectionConfidenceBoundingBox{
          cv::Rect(0, 0, face_chip_mat.cols, face_chip_mat.rows),
          detectionResultEntry.getFaceDetBoundingBox().confidenceScore});
}

trustid::image::impl::DlibFaceVerificator::DlibFaceVerificator(
    const std::vector<FaceDetectionResultEntry> groundTruthChips,
    const float distanceThreshold, const float votingThreshold) {
  // initialize config
  config = DlibFaceVerificatorConfig();
  config.distanceThreshold = distanceThreshold;
  config.votingThreshold = votingThreshold;

  // load the base network
  dlib::deserialize("resources/dlib_face_recognition_resnet_model_v1.dat") >>
      config.net;

  // add the preprocessor to extract the face chips
  dlib::deserialize("resources/ERT68.dat") >> config.sp;

  this->addPreprocessor(std::make_unique<DlibFaceChipExtractor>(config.sp));

  for (auto &chip : groundTruthChips) {
    // Convert OpenCV image to dlib format
    dlib::matrix<dlib::rgb_pixel> matrix;
    dlib::assign_image(matrix, dlib::cv_image<dlib::bgr_pixel>(
                                   applyProcessors(chip).getCroppedImage()));

    // Extract the face descriptor
    auto groundTruthVec = config.net(matrix);

    // calculate embedding add it to the ground truth vector list
    config.groundTruthVecs.push_back(groundTruthVec);
  }
  std::cout << "Ground truth vector size: " << config.groundTruthVecs.size()
            << std::endl;
}
trustid::image::impl::DlibFaceVerificator::DlibFaceVerificator(
    const DlibFaceVerificatorConfig config)
    : config(config) {
  // add the preprocessor to extract the face chips
  this->addPreprocessor(std::make_unique<DlibFaceChipExtractor>(config.sp));
}

trustid::image::impl::DlibFaceVerificatorConfig
trustid::image::impl::DlibFaceVerificator::getConfig() {
  return config;
}

trustid::image::FaceVerificationResult
trustid::image::impl::DlibFaceVerificator::_verifyUser(
    const FaceDetectionResultEntry detectionResultEntry) {
  // Get the face embedding of the image to test
  dlib::matrix<dlib::rgb_pixel> matrix;
  dlib::assign_image(matrix, dlib::cv_image<dlib::bgr_pixel>(
                                 detectionResultEntry.getCroppedImage()));

  auto testVec = config.net(matrix);
  // Calculate the distance to all ground truth vectors
  int count = 0;
  for (auto groundTruthVec : config.groundTruthVecs) {
    // Calculate the distance ofS
    auto distance = dlib::length(testVec - groundTruthVec);
    // std::cout << "distance: " << distance << std::endl;
    if (distance < config.distanceThreshold) {
      count++;
    }
  }
  // Calculate the voting percentage and determine if it's the real user based
  // on voting threshold
  return FaceVerificationResult(
      detectionResultEntry,
      static_cast<float>(count) / config.groundTruthVecs.size(),
      static_cast<float>(count) / config.groundTruthVecs.size() >
              config.votingThreshold
          ? SAME_USER
          : DIFFERENT_USER);
}