/**
 * @file build_simple_face_verifier.cc
 * @author jncfa (jose.faria@isr.uc.pt)
 * @brief Example of how to build a simple face verifier
 * @version 0.1
 * @date 2022-08-17
 *
 */

#include "trustid_image_processing/dlib_impl/face_verificator.h"
#include "trustid_image_processing/dlib_impl/face_detector.h"
#include "trustid_image_processing/client/client_processor.h"
#include "trustid_image_processing/server/server_processor.h"
#include <memory>
#include <iostream>

int main(int argc, char **argv)
{

    std::cout << "face built" << std::endl;

    // Create a client processor (what will run on the client side)
    // Note that, since we're not loading face verification data, attempting to use face verification methods ~
    // will result in a failure until you load it via the loadFaceVerificationData method.
    auto clientProcessor = std::make_unique<trustid::image::ClientImageProcessor>();

    // Create a server processor (what will run on the server side)
    auto serverProcessor = std::make_unique<trustid::image::ServerImageProcessor>();

    // The server will create a face verification model based on the given ground truth results
    // These are the images of the user that will be used to create the model
    std::vector<trustid::image::FaceDetectionResultEntry> facesDetected = {};

    // Open directory with images
    dlib::directory dir("images");
    for (auto &f : dir.get_files())
    {
        std::cout << f.full_name() << std::endl;

        // Load image via OpenCV (this is just an example, you can use any image loading method you want,
        // ideally one that supports loading from memory since you'll want to retrieve images from the webcam)
        cv::Mat img = cv::imread(f.full_name(), cv::IMREAD_COLOR);
        auto detectedFaces = clientProcessor->detectFaces(img);

        // check if there's exactly one face on the image
        if (detectedFaces.getResult() == trustid::image::ONE_RESULT)
        {
            // This currently returns the full image, but we can change this to crop it if needed
            facesDetected.push_back(detectedFaces.getEntry());
        }
        else
        {
            
            throw std::runtime_error("There should be exactly one face on the image");
        }
    }
    
    std::cout << "creating model" << std::endl;

    // Create face verification model and get configuration to be sent to the client
    auto faceVerificator = serverProcessor->createVerificationModel(facesDetected);
    std::cout << "face built" << std::endl;

    // We would serialize and send this back to the client, but here we're just passing the configuration to the client processor directly   
    auto faceVerificatorConfig = faceVerificator->getConfig();

    // Now we can use the face verificator to verify faces
    // Let's use the same images we used to create the model
    clientProcessor->loadFaceVerificationData(faceVerificatorConfig);
    
    std::cout << "opening bald guys image" << std::endl;

    // Load image via OpenCV (this is just an example, you can use any image loading method you want,
    // ideally one that supports loading from memory since you'll want to retrieve images from the webcam)
    cv::Mat img = cv::imread("bald_guys.jpg", cv::IMREAD_COLOR);
    auto detectedFaces = clientProcessor->detectFaces(img);

    // check if there's exactly one face on the image
    if (detectedFaces.getResult() == trustid::image::ONE_RESULT)
    {
        // This currently returns the full image, but we can change this to crop it if needed
        auto faceVerificationResult = clientProcessor->verifyUser(detectedFaces.getEntry());
        std::cout << "Face verification result: " << faceVerificationResult.getMatchConfidence() << std::endl;
    }
    else if (detectedFaces.getResult() == trustid::image::MULTIPLE_RESULTS)
    {
        for (auto face : detectedFaces.getBoundingBoxEntries())
        {
            auto faceVerificationResult = clientProcessor->verifyUser(face);
            std::cout << "Face verification result: " << faceVerificationResult.getMatchConfidence() << std::endl;
        }
    }
    else
    {
        throw std::runtime_error("There should be at least one face on the image");
    }
    
    dlib::directory dir_test("test_images");

    for (auto &f : dir_test.get_files())
    {
        std::cout << f.full_name() << std::endl;

        // Load image via OpenCV (this is just an example, you can use any image loading method you want,
        // ideally one that supports loading from memory since you'll want to retrieve images from the webcam)
        cv::Mat img = cv::imread(f.full_name(), cv::IMREAD_COLOR);
        auto detectedFaces = clientProcessor->detectFaces(img);

        // check if there's exactly one face on the image
        if (detectedFaces.getResult() == trustid::image::ONE_RESULT)
        {
            // This currently returns the full image, but we can change this to crop it if needed
            auto faceVerificationResult = clientProcessor->verifyUser(detectedFaces.getEntry());
            std::cout << "Face verification result: " << faceVerificationResult.getMatchConfidence() << std::endl;
        }
        else
        {
            throw std::runtime_error("There should be exactly one face on the image");
        }
    }

    // Serialize the data and save it to a file
    dlib::serialize("face_verificator.dat") << faceVerificatorConfig;

    return 0;
}