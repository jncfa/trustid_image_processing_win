/**
 * @file build_simple_face_verifier.cc
 * @author jncfa (jose.faria@isr.uc.pt)
 * @brief Example of how to use a face verifier
 * @version 0.1
 * @date 2022-08-17
 *
 */

#include <iostream>
#include <memory>

#include "trustid_image_processing/client/client_processor.h"
#include "trustid_image_processing/dlib_impl/face_detector.h"
#include "trustid_image_processing/dlib_impl/face_verificator.h"
#include "trustid_image_processing/server/server_processor.h"

int main(int argc, char **argv) {
  trustid::image::impl::DlibFaceVerificatorConfig config;
  dlib::deserialize("face_verificator.dat") >> config;

  // Create a client processor (what will run on the client side)
  auto clientProcessor =
      std::make_unique<trustid::image::ClientImageProcessor>(config);

  // Loading the test images
  dlib::directory dir_test("test_images.bk");

  for (auto &f : dir_test.get_files()) {
    std::cout << f.full_name() << std::endl;

    // Load image via OpenCV (this is just an example, you can use any image
    // loading method you want, ideally one that supports loading from memory
    // since you'll want to retrieve images from the webcam)
    cv::Mat img = cv::imread(f.full_name(), cv::IMREAD_COLOR);
    auto detectedFaces = clientProcessor->detectFaces(img);

    // check if there's exactly one face on the image
    if (detectedFaces.getResult() == trustid::image::ONE_RESULT) {
      // This currently returns the full image, but we can change this to crop
      // it if needed
      auto faceVerificationResult =
          clientProcessor->verifyUser(detectedFaces.getEntry());
      std::cout << "Face verification result: "
                << faceVerificationResult.getMatchConfidence() << std::endl;
    } else {
      std::runtime_error("There should be exactly one face on the image");
    }
  }

  return 0;
}