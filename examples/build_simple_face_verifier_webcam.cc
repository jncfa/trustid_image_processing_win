/**
 * @file build_simple_face_verifier.cc
 * @author jncfa (jose.faria@isr.uc.pt)
 * @brief Example of how to build a simple face verifier
 * @version 0.1
 * @date 2022-08-17
 *
 */

#include <ctime>
#include <iostream>
#include <memory>

#include "trustid_image_processing/client/client_processor.h"
#include "trustid_image_processing/dlib_impl/face_detector.h"
#include "trustid_image_processing/dlib_impl/face_verificator.h"
#include "trustid_image_processing/server/server_processor.h"

int main(int argc, char **argv) {
  // Create a client processor (what will run on the client side)
  // Note that, since we're not loading face verification data, attempting to
  // use face verification methods ~ will result in a failure until you load it
  // via the loadFaceVerificationData method.
  auto clientProcessor =
      std::make_unique<trustid::image::ClientImageProcessor>();

  // Create a server processor (what will run on the server side)
  auto serverProcessor =
      std::make_unique<trustid::image::ServerImageProcessor>();

  // The server will create a face verification model based on the given ground
  // truth results These are the images of the user that will be used to create
  // the model
  std::vector<trustid::image::FaceDetectionResultEntry> facesDetected = {};

  cv::Mat frame;
  //--- INITIALIZE VIDEOCAPTURE
  cv::VideoCapture cap;

  time_t old_time = time(NULL);

  // open the default camera using default API
  // cap.open(0);
  // OR advance usage: select any API backend
  int deviceID = 1;         // 0 = open default camera
  int apiID = cv::CAP_ANY;  // 0 = autodetect default API
  // open selected camera using selected API
  cap.open(deviceID, apiID);
  // check if we succeeded
  if (!cap.isOpened()) {
    std::cerr << "ERROR! Unable to open camera\n";
    return -1;
  }
  //--- GRAB AND WRITE LOOP
  std::cout << "Start grabbing" << std::endl
            << "Press any key to terminate" << std::endl;
  for (;;) {
    // wait for a new frame from camera and store it into 'frame'
    cap.read(frame);
    // check if we succeeded
    if (frame.empty()) {
      std::cerr << "ERROR! blank frame grabbed\n";
      break;
    }

    auto detectedFaces = clientProcessor->detectFaces(frame);

    // check if there's exactly one face on the image
    if (detectedFaces.getResult() == trustid::image::ONE_RESULT) {
      if (facesDetected.size() < 15) {
        time_t current_time = time(NULL);
        if (current_time - old_time > 1) {
          // This currently returns the full image, but we can change this to
          // crop it if needed
          facesDetected.push_back(detectedFaces.getEntry());
          old_time = current_time;
          std::cout << "got image " << facesDetected.size() << std::endl;
        }
      } else {
        if (!clientProcessor->canVerifyFaces()) {
          // Create face verification model and get configuration to be sent to
          // the client
          auto faceVerificator =
              serverProcessor->createVerificationModel(facesDetected);

          // We would serialize and send this back to the client, but here we're
          // just passing the configuration to the client processor directly
          auto faceVerificatorConfig = faceVerificator->getConfig();

          // Now we can use the face verificator to verify faces
          // Let's use the same images we used to create the model
          clientProcessor->loadFaceVerificationData(faceVerificatorConfig);
          std::cout << "face built" << std::endl;
        } else {
          // Now we can use the face verificator to verify faces
          // Let's use the same images we used to create the model
          auto verificationResult =
              clientProcessor->verifyUser(detectedFaces.getEntry());
          cv::putText(
              frame,
              ((verificationResult.getResult() ==
                trustid::image::FaceVerificationResultEnum::SAME_USER)
                   ? std::string("Same User")
                   : std::string("Different User")),
              cv::Point(detectedFaces.getEntry().getBoundingBox().tl().x,
                        detectedFaces.getEntry().getBoundingBox().tl().y - 20),
              cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 0), 2,
              cv::LINE_AA);
        }
      }
      cv::rectangle(frame, detectedFaces.getEntry().getBoundingBox(),
                    cv::Scalar(255, 0, 0), 2);
    } else {
      for (auto face : detectedFaces.getBoundingBoxEntries()) {
        if (clientProcessor->canVerifyFaces()) {
          // Now we can use the face verificator to verify faces
          // Let's use the same images we used to create the model
          auto verificationResult = clientProcessor->verifyUser(face);
          cv::putText(frame,
                      ((verificationResult.getResult() ==
                        trustid::image::FaceVerificationResultEnum::SAME_USER)
                           ? std::string("Same User")
                           : std::string("Different User")),
                      cv::Point(face.getBoundingBox().tl().x,
                                face.getBoundingBox().tl().y - 20),
                      cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 0), 2,
                      cv::LINE_AA);
        }
        cv::rectangle(frame, face.getBoundingBox(), cv::Scalar(255, 0, 0), 2);
      }
    }
    // show live and wait for a key with timeout long enough to show images
    cv::imshow("Live", frame);
    if (cv::waitKey(5) >= 0) break;
  }
  // the camera will be deinitialized automatically in VideoCapture destructor

  return 0;
}