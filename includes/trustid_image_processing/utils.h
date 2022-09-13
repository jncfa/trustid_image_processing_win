#ifndef TRUSTID_UTILS_H_
#define TRUSTID_UTILS_H_

#include <opencv2/opencv.hpp>
#include "face_detector.h"

namespace trustid
{
    namespace image
    {
        namespace utils
        {
            enum ImageQualityResultEnum
            {
                LOW_CONTRAST,
                HIGH_CONTRAST,
                UNKNOWN
            };

            /**
             * Checks the image contrast of a given image.
             */

            ImageQualityResultEnum checkImageQuality(const cv::Mat image);
            
            /**
             * Checks the image contrast of a given image.
             */
            ImageQualityResultEnum checkImageQuality(const trustid::image::FaceDetectionResultEntry detectionResultEntry);
        }
    }
}
#endif // TRUSTID_UTILS_H_
