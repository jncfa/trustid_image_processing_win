#include "trustid_image_processing/dlib_impl/face_detector.h"

#include "trustid_image_processing/utils.h"

trustid::image::impl::DlibFaceDetector::DlibFaceDetector() {
  // loads the face detection model
  ffdetector = dlib::get_frontal_face_detector();
}

trustid::image::FaceDetectionResult
trustid::image::impl::DlibFaceDetector::_detectFaces(const cv::Mat image) {
  // run dlib's face detector on the image passed in
  std::vector<dlib::rect_detection> dlibDetections = {};
  ffdetector(dlib::cv_image<dlib::bgr_pixel>(image), dlibDetections);

  // convert dlib's rectangle to our internal detection format
  std::vector<FaceDetectionConfidenceBoundingBox> faces = {};
  for (auto &det : dlibDetections) {
    FaceDetectionConfidenceBoundingBox detectResult;
    detectResult.boundingBox = utils::dlibRectangleToOpenCV(det.rect);
    detectResult.confidenceScore = det.detection_confidence;
    faces.push_back(detectResult);
  }

  // return the faces found
  return FaceDetectionResult(image, faces);
}
