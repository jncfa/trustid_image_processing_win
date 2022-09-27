#include "trustid_image_processing/serialize.h"

void cv::serialize(const Rect& item, std::ostream& out) {
  int x = item.x;
  int y = item.y;
  int width = item.width;
  int height = item.height;
  dlib::serialize(x, out);
  dlib::serialize(y, out);
  dlib::serialize(width, out);
  dlib::serialize(height, out);
}

void cv::deserialize(Rect& item, std::istream& in) {
  int x;
  int y;
  int width;
  int height;
  dlib::deserialize(x, in);
  dlib::deserialize(y, in);
  dlib::deserialize(width, in);
  dlib::deserialize(height, in);
  item = Rect(x, y, width, height);
}

void cv::serialize(const Mat& item, std::ostream& out) {
  // convert to a placeholder vector to store byte data
  std::vector<unsigned char> data(item.datastart, item.dataend);
  dlib::serialize(item.type(), out);
  dlib::serialize(item.rows, out);
  dlib::serialize(item.cols, out);
  dlib::serialize(item.step, out);
  dlib::serialize(data, out);
}

void cv::deserialize(Mat& item, std::istream& in) {
  int type, rows, cols;
  uint64 step;
  // use placeholder vector to store byte data
  std::vector<unsigned char> data;
  dlib::deserialize(type, in);
  dlib::deserialize(rows, in);
  dlib::deserialize(cols, in);
  dlib::deserialize(step, in);
  dlib::deserialize(data, in);

  // allocate memory and copy data (because simply moving the data pointer may
  // cause a segfault since std::vector will free the memory when it goes out of
  // scope, regardless of it being moved or not)
  unsigned char* data_ptr = static_cast<unsigned char*>(malloc(data.size()));
  std::copy(data.begin(), data.end(), data_ptr);
  item = Mat(rows, cols, type, std::move(data_ptr), step);
}

template <typename _Tp, int m, int n>
void cv::serialize(const Matx<_Tp, m, n>& item, std::ostream& out) {
  // convert to a placeholder vector to store byte data
  std::vector<_Tp> data(item.val, item.val + m * n);
  dlib::serialize(data, out);
}

template <typename _Tp, int m, int n>
void cv::deserialize(Matx<_Tp, m, n>& item, std::istream& in) {
  // use placeholder vector to store byte data
  std::vector<_Tp> data;
  dlib::deserialize(data, in);
  
  // allocate memory and copy data (because simply moving the data pointer may
  // cause a segfault since std::vector will free the memory when it goes out of
  // scope, regardless of it being moved or not)
  _Tp* data_ptr = static_cast<_Tp*>(malloc(data.size() * sizeof(_Tp)));
  std::copy(data.begin(), data.end(), data_ptr);
  item = Matx<_Tp, m, n>(data_ptr);
}
