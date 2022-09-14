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
            FaceVerificationResult(FaceDetectionResultEntry detectionResultEntry, double matchConfidence, FaceVerificationResultEnum resultValue);

            /**
             * Returns the confidence score of the match of the given face image to a certain user.
             */
            double getMatchConfidence() const;

            /**
             * Returns the result of the face verification operation.
             */
            FaceVerificationResultEnum getResult() const;

            /**
             * Returns the face detection result for the given face verification operation.
             */
            FaceDetectionResultEntry getDetectionResult() const;

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
            IFaceVerifyImageProcessor();
            virtual FaceDetectionResultEntry operator() (const FaceDetectionResultEntry detectionResultEntry) = 0;
        };

        /**
         * Resizes input image to the specified size.
         */
        class ResizeImageProcessor : public IFaceVerifyImageProcessor
        {
        public:
            ResizeImageProcessor(int width, int height);
            virtual FaceDetectionResultEntry operator() (const FaceDetectionResultEntry detectionResultEntry);

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
            CropImageProcessor(cv::Rect cropInfo);
            virtual FaceDetectionResultEntry operator() (const FaceDetectionResultEntry detectionResultEntry) override;

        private:
            cv::Rect cropInfo;
        };

        /**
         * Detects faces in an image.
         */
        class IFaceVerificator
        {
        public:
            IFaceVerificator();
            IFaceVerificator(std::vector<std::unique_ptr<IFaceVerifyImageProcessor>> preprocessors);

            void addPreprocessor(std::unique_ptr<IFaceVerifyImageProcessor> preprocessor);

            void removePreprocessors();

            /**
             * Detects faces in an image.
             */
            FaceVerificationResult verifyUser(const FaceDetectionResultEntry detectionResultEntry);

        protected:
            FaceDetectionResultEntry applyProcessors(const FaceDetectionResultEntry detectionResultEntry);

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
