#include <vector>
#include <memory>
#include <iostream>
#include <dlib/dnn.h>
#include <dlib/opencv.h>
#include <istream>
#include <dlib/gui_widgets.h>
#include "trustid_image_processing/face_verificator.h"
#include "trustid_image_processing/dlib_impl/face_verificator.h"

trustid::image::impl::DlibFaceChipExtractor::DlibFaceChipExtractor()
{
    // TODO: Make this configurable later
    dlib::deserialize("resources/ERT68.dat") >> sp;
}

trustid::image::FaceDetectionResultEntry trustid::image::impl::DlibFaceChipExtractor::processImage(const FaceDetectionResultEntry detectionResultEntry)
{
    // get the bounding box of the face
    auto detection = detectionResultEntry.getBoundingBox();
    auto image = detectionResultEntry.getImage();

    // convert it to dlib objects
    dlib::cv_image<dlib::bgr_pixel> dlibImage(image);
    auto shape = sp(dlibImage, dlib::rectangle(detection.x, detection.y, detection.x + detection.width, detection.y + detection.height));

    // extract the face chip
    dlib::array2d<dlib::rgb_pixel> face_chip;
    dlib::extract_image_chip(dlibImage, dlib::get_face_chip_details(shape, 150, 0.25), face_chip);

    dlib::image_window wnd(face_chip);

    // convert it back to OpenCV format
    cv::Mat face_chip_mat = dlib::toMat(face_chip);

    // return the new result
    return FaceDetectionResultEntry(face_chip_mat, {FaceDetectionConfidenceBoundingBox{cv::Rect(0, 0, face_chip_mat.size().width, face_chip_mat.size().height), 1.0}});
}


trustid::image::impl::DlibFaceVerificator::DlibFaceVerificator(const std::vector<FaceDetectionResultEntry> groundTruthChips, const float distanceThreshold, const float votingThreshold)
{
    // initialize config
    config = DlibFaceVerificatorConfig();
    config.distanceThreshold = distanceThreshold;
    config.votingThreshold = votingThreshold;

    // load the base network
    dlib::deserialize("resources/dlib_face_recognition_resnet_model_v1.dat") >> config.net;

    // add the preprocessor to extract the face chips
    //this->addPreprocessor(std::make_unique<DlibFaceChipExtractor>());
    this->addPreprocessor(std::make_unique<ResizeImageProcessor>(150, 150));
    for (auto &chip : groundTruthChips)
    {
        // Get the face embedding of the image to test
        std::cout << "hello world " << std::endl;
        cv::imshow("Display window", applyProcessors(chip).getCroppedImage());
        int k = cv::waitKey(0); // Wait for a keystroke in the window
        dlib::matrix<dlib::rgb_pixel> matrix;
        dlib::assign_image(matrix, dlib::cv_image<dlib::bgr_pixel>(applyProcessors(chip).getCroppedImage()));

        // calculate embedding add it to the ground truth vector list
        config.groundTruthVecs.push_back(config.net(matrix));
    }
    std::cout << "Ground truth vector size: " << config.groundTruthVecs.size() << std::endl;
}
trustid::image::impl::DlibFaceVerificator::DlibFaceVerificator(const DlibFaceVerificatorConfig config) : config(config)
{
    // add the preprocessor to extract the face chips
    this->addPreprocessor(std::make_unique<DlibFaceChipExtractor>());
}

trustid::image::impl::DlibFaceVerificatorConfig trustid::image::impl::DlibFaceVerificator::getConfig()
{
    return config;
}

trustid::image::FaceVerificationResult trustid::image::impl::DlibFaceVerificator::_verifyUser(const FaceDetectionResultEntry detectionResultEntry)
{
    // Get the face embedding of the image to test
    dlib::matrix<dlib::rgb_pixel> matrix;
    dlib::assign_image(matrix, dlib::cv_image<dlib::bgr_pixel>(applyProcessors(detectionResultEntry).getCroppedImage()));
    auto testVec = config.net(matrix);

    // Calculate the distance to all ground truth vectors
    int count = 0;
    for (auto groundTruthVec : config.groundTruthVecs)
    {
        // Calculate the distance of
        auto distance = dlib::length(testVec - groundTruthVec);
        if (distance < config.distanceThreshold)
        {
            count++;
        }
    }
    // Calculate the voting percentage and determine if it's the real user based on voting threshold
    return FaceVerificationResult(detectionResultEntry, static_cast<float>(count) / config.groundTruthVecs.size(), static_cast<float>(count) / config.groundTruthVecs.size() > config.votingThreshold ? SAME_USER : DIFFERENT_USER);
}