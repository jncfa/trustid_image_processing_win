/**
 * @file utils.cc
 * @brief Testing the serialization of multiple components of the TRUSTID image
 * processing library.113
 * @author jncfa
 * @date 2022-10-22
 */

#include <dlib/serialize.h>

#include <iostream>
#include <opencv2/core.hpp>

#include "trustid_image_processing/client/client_processor.h"
#include "trustid_image_processing/serialize.h"
#include "trustid_image_processing/server/server_processor.h"

namespace image = trustid::image;
namespace impl = trustid::image::impl;

int main(int argc, char const *argv[]) {
  // open image with opencv, serialize it with dlib, deserialize it and display
  // both side by side
  cv::Mat image(100, 100, cv::CV_64FC1);
  cv::Mat image2;

  cv::randu(image, cv::Scalar(0, 0, 0), cv::Scalar(256, 256, 256));

  // serialize image and deserialize it into a new image
  std::stringstream ss;
  dlib::serialize(ss) << image;
  dlib::deserialize(ss) >> image2;

  auto result = cv::sum(image - image2);
  bool testIfZero =
      result[0] == 0 && result[1] == 0 && result[2] == 0 && result[3] == 0;
  return 0;
}
