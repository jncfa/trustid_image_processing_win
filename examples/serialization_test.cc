/**
 * @file serialization_test.cc
 * @brief Testing the serialization of multiple components of the TRUSTID image
 * processing library.
 * @author jncfa
 * @date 2019-10-30
 */

/**
 * @brief
 *
 * @param argc
 * @param argv
 * @return int
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
  cv::Mat3b random_image(100, 100);
  cv::randu(random_image, cv::Scalar(0, 0, 0), cv::Scalar(256, 256, 256));

  // open image with opencv, serialize it with dlib, deserialize it and display
  // both side by side
  cv::Mat image = cv::imread(
      "C:\\Users\\jncfa\\source\\repos\\trustid_image_processing_"
      "win\\tests\\test_resources\\images\\000288_00925786.jpg");
  cv::Mat image2;

  // serialize image and deserialize it into a new image
  std::stringstream ss;
  dlib::serialize(ss) << image;
  dlib::deserialize(ss) >> image2;

  auto result = cv::sum(image - image2);
  bool testIfZero =
      result[0] == 0 && result[1] == 0 && result[2] == 0 && result[3] == 0;
  return 0;
}
