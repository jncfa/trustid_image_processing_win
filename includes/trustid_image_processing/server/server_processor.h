#ifndef TRUSTID_SERVER_IMAGE_PROCESSING_PIPELINE_H_
#define TRUSTID_SERVER_IMAGE_PROCESSING_PIPELINE_H_

#include "trustid_image_processing/face_detector.h"
#include "trustid_image_processing/face_verificator.h"
#include "trustid_image_processing/face_normalization.h"
#include "trustid_image_processing/utils.h"
#include "trustid_image_processing/dlib_impl/face_verificator.h"

namespace trustid
{
    namespace image
    {
        class ServerImageProcessor
        {

        public:
            ServerImageProcessor();

            /**
             * Creates a face verification model based in the given face detection result.
             */
            std::unique_ptr<impl::DlibFaceVerificator> createVerificationModel(std::vector<FaceDetectionResultEntry> groundTruthResults);

            /**
             * Check the constrast of a given face image.
             */
            static utils::ImageQualityResultEnum checkImageQuality(const cv::Mat image);

            /**
             * Check the constrast of a given face image.
             */
            static utils::ImageQualityResultEnum checkImageQuality(const FaceDetectionResultEntry detectionResultEntry);
        };
    }
}

#endif // TRUSTID_SERVER_IMAGE_PROCESSING_PIPELINE_H_