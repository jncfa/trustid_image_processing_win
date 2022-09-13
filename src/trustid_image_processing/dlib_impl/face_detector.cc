#include "trustid_image_processing/dlib_impl/face_detector.h"


trustid::image::impl::DlibFaceDetector::DlibFaceDetector()
{
    // loads the face detection model
    ffdetector = dlib::get_frontal_face_detector();
}

trustid::image::FaceDetectionResult trustid::image::impl::DlibFaceDetector::_detectFaces(const cv::Mat image)
{

    // run dlib's face detector on the image passed in
    std::vector<dlib::rect_detection> dlibDetections = {};
    ffdetector(dlib::cv_image<dlib::bgr_pixel>(image), dlibDetections);

    // convert dlib's rectangle to our internal detection format
    std::vector<FaceDetectionConfidenceBoundingBox> faces = {};
    for (auto &det : dlibDetections)
    {
        FaceDetectionConfidenceBoundingBox detectResult;
        detectResult.boundingBox = cv::Rect(det.rect.left(), det.rect.top(), det.rect.width(), det.rect.height());
        detectResult.confidenceScore = det.detection_confidence;
        faces.push_back(detectResult);

        std::cout << detectResult.boundingBox.x << "|" << detectResult.boundingBox.y << "|" << detectResult.boundingBox.width << "|" << detectResult.boundingBox.height << std::endl;
    }

    // return the faces found
    return FaceDetectionResult(image, faces);
}
