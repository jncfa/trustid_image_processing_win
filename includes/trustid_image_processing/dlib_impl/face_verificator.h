#ifndef TRUSTID_DLIB_FACE_VERIFICATOR_H_
#define TRUSTID_DLIB_FACE_VERIFICATOR_H_

#include <dlib/dnn.h>
#include <dlib/opencv.h>

#include <memory>
#include <vector>

#include "trustid_image_processing/face_verificator.h"

// TODO: Disabling Protobuf for now, as it causes issues with the build, and we
// don't need it for now #include "face_verificator_config.pb.h"

// The next bit of code defines a ResNet network.  It's basically copied
// and pasted from the dnn_imagenet_ex.cpp example, except we replaced the loss
// layer with loss_metric and made the network somewhat smaller.  Go read the
// introductory dlib DNN examples to learn what all this stuff means.
//
// Also, the dnn_metric_learning_on_images_ex.cpp example shows how to train
// this network. The dlib_face_recognition_resnet_model_v1 model used by this
// example was trained using essentially the code shown in
// dnn_metric_learning_on_images_ex.cpp except the mini-batches were made larger
// (35x15 instead of 5x5), the iterations without progress was set to 10000, and
// the training dataset consisted of about 3 million images instead of
// 55.  Also, the input layer was locked to images of size 150.

template <template <int, template <typename> class, int, typename> class block,
          int N, template <typename> class BN, typename SUBNET>
using residual = dlib::add_prev1<block<N, BN, 1, dlib::tag1<SUBNET>>>;

template <template <int, template <typename> class, int, typename> class block,
          int N, template <typename> class BN, typename SUBNET>
using residual_down = dlib::add_prev2<dlib::avg_pool<
    2, 2, 2, 2, dlib::skip1<dlib::tag2<block<N, BN, 2, dlib::tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block =
    BN<dlib::con<N, 3, 3, 1, 1,
                 dlib::relu<BN<dlib::con<N, 3, 3, stride, stride, SUBNET>>>>>;

template <int N, typename SUBNET>
using ares = dlib::relu<residual<block, N, dlib::affine, SUBNET>>;
template <int N, typename SUBNET>
using ares_down = dlib::relu<residual_down<block, N, dlib::affine, SUBNET>>;

template <typename SUBNET>
using alevel0 = ares_down<256, SUBNET>;
template <typename SUBNET>
using alevel1 = ares<256, ares<256, ares_down<256, SUBNET>>>;
template <typename SUBNET>
using alevel2 = ares<128, ares<128, ares_down<128, SUBNET>>>;
template <typename SUBNET>
using alevel3 = ares<64, ares<64, ares<64, ares_down<64, SUBNET>>>>;
template <typename SUBNET>
using alevel4 = ares<32, ares<32, ares<32, SUBNET>>>;

using ResNet34 = dlib::loss_metric<dlib::fc_no_bias<
    128,
    dlib::avg_pool_everything<
        alevel0<alevel1<alevel2<alevel3<alevel4<dlib::max_pool<
            3, 3, 2, 2,
            dlib::relu<dlib::affine<dlib::con<
                32, 7, 7, 2, 2, dlib::input_rgb_image_sized<150>>>>>>>>>>>>>;

namespace trustid {
namespace image {
namespace impl {
class DlibFaceChipExtractor : public IFaceVerifyImageProcessor {
 public:
  DlibFaceChipExtractor(dlib::shape_predictor sp);
  virtual FaceDetectionResultEntry operator()(
      const FaceDetectionResultEntry detectionResultEntry) override;

 private:
  dlib::shape_predictor sp;
};

struct DlibFaceVerificatorConfig {
  DlibFaceVerificatorConfig() {}
  DlibFaceVerificatorConfig(
      const std::vector<dlib::matrix<float, 0, 1>> groundTruthVecs,
      const ResNet34 net, const float distanceThreshold = 0.6,
      const float votingThreshold = 0.5)
      : groundTruthVecs(groundTruthVecs),
        net(net),
        distanceThreshold(distanceThreshold),
        votingThreshold(votingThreshold) {}

  std::vector<dlib::matrix<float, 0, 1>> groundTruthVecs;
  ResNet34 net;
  float distanceThreshold;
  float votingThreshold;

  dlib::shape_predictor sp;

  // TODO: Replace serialization for Grpc compatible method
  DLIB_DEFINE_DEFAULT_SERIALIZATION(DlibFaceVerificatorConfig, groundTruthVecs,
                                    net, distanceThreshold, votingThreshold,
                                    sp);
};

class DlibFaceVerificator : public IFaceVerificator {
 public:
  DlibFaceVerificator(
      const std::vector<FaceDetectionResultEntry> groundTruthChips,
      const float distanceThreshold = 0.6, const float votingThreshold = 0.5);
  DlibFaceVerificator(const DlibFaceVerificatorConfig config);

  DlibFaceVerificatorConfig getConfig();

 private:
  virtual FaceVerificationResult _verifyUser(
      const FaceDetectionResultEntry detectionResultEntry) override;
  DlibFaceVerificatorConfig config;
};
}  // namespace impl
}  // namespace image
}  // namespace trustid
#endif  // TRUSTID_DLIB_FACE_VERIFICATOR_H_