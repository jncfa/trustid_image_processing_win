#ifndef TRUSTID_FACE_VERIFICATOR_H_
#define TRUSTID_FACE_VERIFICATOR_H_

#include "face_detector.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <utility>

namespace trustid
{
    namespace image
    {
        /**
         * Result of a given face verification operation.
         */
        enum FaceVerificationResultEnum
        {
            SAME_USER,
            DIFFERENT_USER,
            UNKNOWN
        };
        /**
         * Stores information regarding a face verification operation.
         */
        class FaceVerificationResult
        {
        public:
            FaceVerificationResult(FaceDetectionResultEntry detectionResultEntry, double matchConfidence, FaceVerificationResultEnum resultValue) : detectionResultEntry(detectionResultEntry), matchConfidence(matchConfidence), resultValue(resultValue)
            {
            }

            /**
             * Returns the confidence score of the match of the given face image to a certain user.
             */
            double getMatchConfidence() const
            {
                return matchConfidence;
            }

            /**
             * Returns the result of the face verification operation.
             */
            FaceVerificationResultEnum getResult() const
            {
                return resultValue;
            }

            /**
             * Returns the face detection result for the given face verification operation.
             */
            FaceDetectionResultEntry getDetectionResult() const
            {
                return detectionResultEntry;
            }

        private:
            double matchConfidence;
            FaceVerificationResultEnum resultValue;
            FaceDetectionResultEntry detectionResultEntry;
        };

        /**
         * Class to implement a processing pipeline for images.
         */
        class IFaceVerifyImageProcessor
        {
        public:
            IFaceVerifyImageProcessor() {}
            virtual FaceDetectionResultEntry processImage(const FaceDetectionResultEntry detectionResultEntry) = 0;
        };

        /**
         * Resizes input image to the specified size.
         */
        class ResizeImageProcessor : public IFaceVerifyImageProcessor
        {
        public:
            ResizeImageProcessor(int width, int height) : width(width), height(height) {}
            virtual FaceDetectionResultEntry processImage(const FaceDetectionResultEntry detectionResultEntry)
            {
                // Resize the image to the specified size
                cv::Mat image = detectionResultEntry.getImage();
                cv::Mat resizedImage;
                auto detectionBoundingBox = detectionResultEntry.getFaceDetBoundingBox();
                cv::resize(image, resizedImage, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);

                // reshape bounding box
                detectionBoundingBox.boundingBox.x = detectionBoundingBox.boundingBox.x * width / resizedImage.cols;
                detectionBoundingBox.boundingBox.y = detectionBoundingBox.boundingBox.y * height / resizedImage.rows;
                detectionBoundingBox.boundingBox.width = detectionBoundingBox.boundingBox.width * width / resizedImage.cols;
                detectionBoundingBox.boundingBox.height = detectionBoundingBox.boundingBox.height * height / resizedImage.rows;
                std::cout << detectionBoundingBox.boundingBox.x << "|" << detectionBoundingBox.boundingBox.y << detectionBoundingBox.boundingBox.width << "|" << detectionBoundingBox.boundingBox.height << std::endl;
                return FaceDetectionResultEntry(resizedImage, detectionBoundingBox);
            }

        private:
            int width;
            int height;
        };

        /**
         * Class to implement a cropping mechanism for images.
         */
        class CropImageProcessor : public IFaceVerifyImageProcessor
        { 
        public:
            CropImageProcessor(cv::Rect cropInfo) : cropInfo(cropInfo) {}

            virtual FaceDetectionResultEntry processImage(const FaceDetectionResultEntry detectionResultEntry) override
            {
                // Crop the image to the specified size
                cv::Mat image = detectionResultEntry.getImage();
                auto cropBoundingBox = detectionResultEntry.getFaceDetBoundingBox();

                // TODO: Add a check when the crop rectangle is outside the image
                cropBoundingBox.boundingBox = cv::Rect(cropBoundingBox.boundingBox.x - cropInfo.x, cropBoundingBox.boundingBox.y - cropInfo.y, cropBoundingBox.boundingBox.width, cropBoundingBox.boundingBox.height);

                return FaceDetectionResultEntry(image(cropInfo), cropBoundingBox);
            }
        private:
            cv::Rect cropInfo;
        };

        /**
         * Detects faces in an image.
         */
        class IFaceVerificator
        {
        public:
            IFaceVerificator() : preprocessors()
            {
            }
            IFaceVerificator(std::vector<std::unique_ptr<IFaceVerifyImageProcessor>> preprocessors) : preprocessors(std::move(preprocessors))
            {
            }

            void addPreprocessor(std::unique_ptr<IFaceVerifyImageProcessor> preprocessor)
            {
                preprocessors.push_back(std::move(preprocessor));
            }

            void removePreprocessors()
            {
                preprocessors.clear();
            }

            /**
             * Detects faces in an image.
             */
            FaceVerificationResult verifyUser(const FaceDetectionResultEntry detectionResultEntry) 
            {
                return _verifyUser(applyProcessors(detectionResultEntry));
            }
        protected:
            FaceDetectionResultEntry applyProcessors(const FaceDetectionResultEntry detectionResultEntry) 
            {
                FaceDetectionResultEntry detectionResultEntryCopy = detectionResultEntry.copy();
                for (auto &processor : preprocessors)
                {
                    detectionResultEntryCopy = processor->processImage(detectionResultEntryCopy);
                }
                return detectionResultEntryCopy;
            }

        private:
            /**
             * Internal procedure for verifying a user.
             * This is split up from the public verifyUser procedure to allow the implementation of pre and/or post-processing steps common to every face verification algorithm.
             */
            virtual FaceVerificationResult _verifyUser(const FaceDetectionResultEntry detectionResultEntry) = 0;
            std::vector<std::unique_ptr<IFaceVerifyImageProcessor>> preprocessors;
        };
    }
}

#endif // TRUSTID_FACE_VERIFICATOR_H_
