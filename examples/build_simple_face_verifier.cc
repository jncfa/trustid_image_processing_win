/**
 * @file build_simple_face_verifier.cc
 * @author jncfa (jose.faria@isr.uc.pt)
 * @brief Example of how to build a simple face verifier
 * @version 0.2
 * @date 2022-10-26
 *
 * Example of how to build a simple face verifier, demonstrating the difference in how the library exposes the models compared to the previous version.
 */

#include <iostream>
#include <memory>

#include "trustid_image_processing/dlib_impl/face_detector.h"
#include "trustid_image_processing/dlib_impl/face_verificator.h"

int main(int argc, char **argv) {
  std::cout << "Building face model using test images.." << std::endl;
  //std::make_unique<impl::DlibFaceVerificator>

  // load needed pre-trained models
  auto net = trustid::image::impl::loadResNet34FromDisk("resources/dlib_face_recognition_resnet_model_v1.dat");
  auto sp = trustid::image::impl::loadShapePredictorFromDisk("resources/ERT68.dat");

  std::unique_ptr<trustid::image::IFaceDetector> faceDetector = std::make_unique<trustid::image::impl::DlibFaceDetector>();
  
  // The face verificator can't be built yet because we don't have the user model data
  std::unique_ptr<trustid::image::impl::DlibFaceVerificator> faceVerificator(nullptr); 
  
  // The server will create a face verification model based on the given ground
  // truth results These are the images of the user that will be used to create
  // the model
  std::vector<trustid::image::FaceDetectionResultEntry> facesDetected = {};

  // Open directory with images
  dlib::directory dir("test_resources/person1");
  for (auto &f : dir.get_files()) {
    std::cout << f.full_name() << std::endl;

    // Load image via OpenCV (this is just an example, you can use any image
    // loading method you want, ideally one that supports loading from memory
    // since you'll want to retrieve images from the webcam)
    cv::Mat img = cv::imread(f.full_name(), cv::IMREAD_COLOR);
    auto detectedFaces = faceDetector->detectFaces(img);

    // check if there's exactly one face on the image
    if (detectedFaces.getResult() == trustid::image::ONE_RESULT) {
      // This currently returns the full image, but we can change this to crop
      // it if needed
      facesDetected.push_back(detectedFaces.getEntry());
    } else {
      throw std::runtime_error("There should be exactly one face on the image");
    }
  }

  std::cout << "creating model" << std::endl;

  // Create face verification model and get configuration to be sent to the
  // client
  faceVerificator = std::make_unique<trustid::image::impl::DlibFaceVerificator>(net, sp, facesDetected);
  std::cout << "face built" << std::endl;

  // We would serialize and send this back to the client, but here we're just
  // passing the configuration to the client processor directly
  auto faceVerificatorUserParams = faceVerificator->getUserParams();

  // Now we can use the face verificator to verify faces
  // Let's use the same images we used to create the model
  
  auto newFaceVerificator = std::make_unique<trustid::image::impl::DlibFaceVerificator>(net, sp, faceVerificatorUserParams);

  dlib::directory dir_test("test_resources/person2");

  for (auto &f : dir_test.get_files()) {
    std::cout << f.full_name() << std::endl;

    // Load image via OpenCV (this is just an example, you can use any image
    // loading method you want, ideally one that supports loading from memory
    // since you'll want to retrieve images from the webcam)
    cv::Mat img = cv::imread(f.full_name(), cv::IMREAD_COLOR);
    auto detectedFaces = faceDetector->detectFaces(img);

    // check if there's exactly one face on the image
    if (detectedFaces.getResult() == trustid::image::ONE_RESULT) {
      // This currently returns the full image, but we can change this to crop
      // it if needed
      auto faceVerificationResult =
          newFaceVerificator->verifyUser(detectedFaces.getEntry());
      std::cout << "Face verification result: "
                << faceVerificationResult.getMatchConfidence() << std::endl;
    } else {
      throw std::runtime_error("There should be exactly one face on the image");
    }
  }

  // Serialize the data and save it to a file
  dlib::serialize("face_verificator.dat") << faceVerificatorUserParams;

  return 0;
}