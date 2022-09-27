#ifndef TRUSTID_DLIB_FACE_DETECTOR_H_
#define TRUSTID_DLIB_FACE_DETECTOR_H_

#include <dlib/image_io.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/opencv.h>

#include "trustid_image_processing/face_detector.h"
#include "trustid_image_processing/face_verificator.h"

namespace trustid {
namespace image {
namespace impl {

class DlibFaceDetector : public trustid::image::IFaceDetector {
 public:
  DlibFaceDetector();

 private:
  trustid::image::FaceDetectionResult _detectFaces(
      const cv::Mat image) override;
  dlib::frontal_face_detector ffdetector;
};
}  // namespace impl
}  // namespace image
}  // namespace trustid
#endif  // TRUSTID_DLIB_FACE_DETECTOR_H_
