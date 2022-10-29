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
    std::shared_ptr<dlib::shape_predictor> sp)
    : sp(sp) {}

trustid::image::FaceDetectionResultEntry
trustid::image::impl::DlibFaceChipExtractor::operator()(
    const FaceDetectionResultEntry detectionResultEntry) {
  // get the bounding box of the face
  auto detection = detectionResultEntry.getBoundingBox();
  auto image = detectionResultEntry.getImage();

  // convert it to dlib objects
  dlib::cv_image<dlib::bgr_pixel> dlibImage(image);
  auto shape = sp->operator()(dlibImage, utils::openCVRectToDlib(detection));

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
    const std::shared_ptr<ResNet34> net,
    const std::shared_ptr<dlib::shape_predictor> sp,
    const std::vector<FaceDetectionResultEntry> groundTruthChips,
    const float distanceThreshold, const float votingThreshold) {
  // initialize config
  this->userParams = DlibFaceVerificatorModelParams();
  this->userParams.distanceThreshold = distanceThreshold;
  this->userParams.votingThreshold = votingThreshold;

  // load the base network
  this->net = net;

  // add the preprocessor to extract the face chips
  this->addPreprocessor(std::make_unique<DlibFaceChipExtractor>(sp));

  for (auto &chip : groundTruthChips) {
    // Convert OpenCV image to dlib format
    dlib::matrix<dlib::rgb_pixel> matrix;
    dlib::assign_image(matrix, dlib::cv_image<dlib::bgr_pixel>(
                                   applyProcessors(chip).getCroppedImage()));

    // Extract the face descriptor
    auto groundTruthVec = this->net->operator()(matrix);

    #ifndef NDEBUG
      std::cout << "vec: " << groundTruthVec << std::endl;
    #endif // DEBUG


    // calculate embedding add it to the ground truth vector list
    this->userParams.groundTruthVecs.push_back(groundTruthVec);
  }

#ifndef NDEBUG
  std::cout << "Ground truth vector size: "
            << this->userParams.groundTruthVecs.size() << std::endl;
#endif // DEBUG

}
trustid::image::impl::DlibFaceVerificator::DlibFaceVerificator(
    const std::shared_ptr<ResNet34> net,
    const std::shared_ptr<dlib::shape_predictor> sp,
    const DlibFaceVerificatorModelParams userParams)
    : userParams(userParams), net(net) {
  // add the preprocessor to extract the face chips
  this->addPreprocessor(std::make_unique<DlibFaceChipExtractor>(sp));
}

trustid::image::impl::DlibFaceVerificatorModelParams
trustid::image::impl::DlibFaceVerificator::getUserParams() {
  return this->userParams;
}

trustid::image::FaceVerificationResult
trustid::image::impl::DlibFaceVerificator::_verifyUser(
    const FaceDetectionResultEntry detectionResultEntry) {
  // Get the face embedding of the image to test
  dlib::matrix<dlib::rgb_pixel> matrix;
  dlib::assign_image(matrix, dlib::cv_image<dlib::bgr_pixel>(
                                 detectionResultEntry.getCroppedImage()));

  auto testVec = this->net->operator()(matrix);
  // Calculate the distance to all ground truth vectors
  int count = 0;
  for (auto groundTruthVec : this->userParams.groundTruthVecs) {
    // Calculate the distance ofS
    auto distance = dlib::length(testVec - groundTruthVec);
    //#ifndef NDEBUG
    //std::cout << "distance: " << distance << std::endl;
    //#endif // DEBUG
    if (distance < this->userParams.distanceThreshold) {
      count++;
    }
  }

  // Calculate the voting percentage and determine if it's the real user based
  // on voting threshold
  auto votingConfidence =
      static_cast<float>(count) / this->userParams.groundTruthVecs.size();
  return FaceVerificationResult(detectionResultEntry, votingConfidence,
                                votingConfidence > userParams.votingThreshold
                                    ? SAME_USER
                                    : DIFFERENT_USER);
}

std::shared_ptr<dlib::shape_predictor>
trustid::image::impl::loadShapePredictorFromDisk(std::string pathToFile) {
  auto sp = std::make_shared<dlib::shape_predictor>();
  dlib::deserialize(pathToFile) >> (*sp);

  return sp;
}
std::shared_ptr<ResNet34> trustid::image::impl::loadResNet34FromDisk(
    std::string pathToFile) {

  auto net = std::make_shared<ResNet34>();
  dlib::deserialize(pathToFile) >> (*net);

  return net;
}