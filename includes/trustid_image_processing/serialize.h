#ifndef TRUSTID_SERIALIZE_H_
#define TRUSTID_SERIALIZE_H_

#include <dlib/serialize.h>

#include <iostream>
#include <opencv2/core.hpp>

// Add dlib serialization support for cv::Rect and cv::Mat
namespace cv {
void serialize(const Rect& item, std::ostream& out);
void deserialize(Rect& item, std::istream& in);
void serialize(const Mat& item, std::ostream& out);
void deserialize(Mat& item, std::istream& in);
template <typename _Tp, int m, int n>
void serialize(const Matx<_Tp, m, n>& item, std::ostream& out);
template <typename _Tp, int m, int n>
void deserialize(Matx<_Tp, m, n>& item, std::istream& in);
}  // namespace cv

// Add gRPC serialization support for cv::Rect and cv::Mat
// namespace grpc::utils{
//    void serialize_to (const Mat& item, std::ostream& out);
//    void deserialize_from (Mat& item, std::istream& in);
//}
#endif  // TRUSTID_SERIALIZE_H_