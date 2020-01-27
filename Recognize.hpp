//
//  Recognize.hpp
//  UITesting
//
//  Created by Grime, John M. on 10/15/19.
//  Copyright © 2019 Grime, John M. All rights reserved.
//

#ifndef Recognize_hpp
#define Recognize_hpp

#include <opencv2/opencv.hpp>

#if defined(HAVE_OPENCV_XFEATURES2D)
	#include <opencv2/xfeatures2d.hpp>
#endif

namespace Recognize
{


using Matches   = std::vector<cv::DMatch>;
using Points    = std::vector<cv::Point2f>;
using Detector  = cv::Ptr<cv::Feature2D>;
using Matcher   = cv::Ptr<cv::DescriptorMatcher>;


//
// Accumulate a change in value over a local scope.
// This is useful for e.g. timing one or more blocks of code.
//
template<typename T>
struct ScopedDeltaAccumulator
{
	T (*f)(), f0, &acc;
	
	ScopedDeltaAccumulator(T& acc, T (*f)()) : f(f), acc(acc) { f0 = f(); }
	~ScopedDeltaAccumulator() { acc += (f()-f0); }
};


//
// Timing routines; int64 is OpenCV's standard tick type.
//
struct Time
{
    using Ticks = int64;

    static Ticks Now()
	{
		return cv::getTickCount();
	}
	static double TicksToSeconds(Ticks t)
	{
		return ((double)t)/cv::getTickFrequency();
	}
	static double SecondsSince(Ticks then)
	{
		return TicksToSeconds(Now()-then);
	}
};


//
// Recognition-relevant metadata regarding an image
//
struct Features
{
	enum class HomographyType {DEFAULT, RANSAC, LMEDS, RHO};

	std::vector<cv::KeyPoint> keypoints;
	cv::Mat descriptors;

	// Find homography transform between Features of two images.
	// RHO seems to be about 4x faster than RANSAC in my testing; runtime is
	// typically dominated by feature detection/matching, though. However,
	// RHO can also provide more "stable" transforms than RANSAC (i.e., image
	// bounds from transform don't twitch around everywhere as much).
	static bool FindHomography(
		const Features& query,
		const Features& reference,
		const Matches& good_matches,
		Points& srcPoints,
		Points& dstPoints,
		cv::Mat& transform,
		HomographyType type = HomographyType::DEFAULT,
		double reprojection = 3.0)
	{
		srcPoints.clear();
		dstPoints.clear();

		for (const auto& m : good_matches) {
			srcPoints.push_back( reference.keypoints[m.trainIdx].pt );
			dstPoints.push_back( query.keypoints[m.queryIdx].pt );
		}

		// def. reproj. value for RANSAC and RHO is 3.0 per OpenCV 4.1.1.
		// sensible range is 1 to 10, according to docs.
		reprojection = (reprojection<3) ? 3 : reprojection;

		switch (type)
		{
			case HomographyType::DEFAULT:
				transform = cv::findHomography(srcPoints, dstPoints);
			break;

			case HomographyType::RANSAC:
				transform = cv::findHomography(srcPoints, dstPoints, cv::RANSAC, reprojection);
			break;

			case HomographyType::LMEDS:
				transform = cv::findHomography(srcPoints, dstPoints, cv::LMEDS);
			break;

			case HomographyType::RHO:
				transform = cv::findHomography(srcPoints, dstPoints, cv::RHO, reprojection);
			break;

			default:
				printf("Unknown homography type.\n");
				exit(-1);
			break;
		}

		return transform.empty() ? false : true;
	}
};


//
// Keeps a paired detector and matcher compatible with one another.
//
struct DetectorMatcherPair
{
	enum class Type {ORB, AKAZE, SIFT, SURF};
	
	Detector detector;
	Matcher matcher;
	
	std::vector<Matches> matches_; // internal, temp
		
	DetectorMatcherPair()
	{
		Prepare(Type::ORB);
	}
	
	// Move this to recognize class, and have it trigger recalculation of 
	// reference data when called?
	bool Prepare(Type type, int param = -1)
	{
		switch (type)
		{
			// Default nFeatures is 500, but this tends not to work so well.
			// OpenCV docs indicate NORM_HAMMING should be used with ORB.
			// If WTA_K 3 or 4 in ORB constructor (def: 2), use NORM_HAMMING2
			case Type::ORB: {
				param = (param<0) ? 500 : param; // number of features
				detector = cv::ORB::create(param);
				matcher = cv::BFMatcher::create(cv::NORM_HAMMING);
				//matcher = cv::FlannBasedMatcher(cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2));
				break;
			}
			
			case Type::AKAZE: {
				detector = cv::AKAZE::create();
				matcher = cv::BFMatcher::create(cv::NORM_HAMMING);
				break;
			}

#if defined(HAVE_OPENCV_XFEATURES2D)
			case Type::SIFT: {
				detector = cv::xfeatures2d::SIFT::create();
				matcher = cv::FlannBasedMatcher::create();
				break;
			}
				
			case Type::SURF: {
				param = (param<0) ? 400 : param; // min Hessian
				detector = cv::xfeatures2d::SURF::create(param);
				matcher = cv::FlannBasedMatcher::create();
				break;
			}
#endif

			default: {
				printf("Unknown recognizer type");
				return false;
			}
		}
		
		return true;
	}
	
	size_t GetFeatures(const cv::Mat& img, Features& f)
	{
		f.keypoints.clear();
		detector->detectAndCompute(img, cv::noArray(), f.keypoints, f.descriptors);
		return f.keypoints.size();
	}
	
	size_t GetMatches(
		const Features& query,
		const Features& reference,
		Matches& good_matches,
		float Lowe_ratio_thresh = 0.7 )
	{
		matches_.clear(); // clear or knnMatch() won't (re)calculate matches
		good_matches.clear(); // clear, as we're appending matches below.

		matcher->knnMatch(query.descriptors, reference.descriptors, matches_, 2);

		for (const auto& m : matches_) {
			if (m[0].distance < Lowe_ratio_thresh * m[1].distance) {
				good_matches.push_back(m[0]);
			}
		}

		return good_matches.size();
	}	
};


//
// Recognizer: store a set of reference images, find them in an input image
//
struct Recognizer
{
	//
	// Describes a reference image to find in scene. Note: does not store the
	// actual pixel data, only features derived from it.
	//
	// For a valid homography transform, several things must be true:
	//
	// 1. Sufficient feature matches of required quality were detected.
	// 2. Homography matrix was successfully calculated.
	//
	// Note that the existance of 1 does not imply 2; homography calculation
	// can fail! It's therefore not enough to simply check matches.size(). We
	// could ALSO check transform.empty(), provided that we explicitly "empty"
	// transform variable of each ReferenceImage in ProcessImage(), but this
	// may introduce some overhead. Simpler and faster to just store a single
	// boolean flag ("present") to check as a single source of truth.
	//
	struct ReferenceImage
	{
		bool ignore, present;
		Features features; // features used for matching
		Matches matches;   // only "good" matches stored
	    cv::Mat transform; // homography; valid where present == true
	};

	//
	// Timings for different aspects of the ProcessImage() call
	//
	struct PerformanceTimings
	{
		Time::Ticks process, detect, match, homography;
	};

	//
	// Fundamental data
	//
	PerformanceTimings perf;
	DetectorMatcherPair dmp;
	std::vector<ReferenceImage> references;
	
	//
	// Internal data, temporary
	//
	Features features_;
	Points srcPoints_, dstPoints_;
    
	bool Prepare(DetectorMatcherPair::Type type, int param = -1)
	{
		return dmp.Prepare(type, param);
	}

	bool AddReferenceImage(cv::Mat& img)
	{
		ReferenceImage ref;
		
		ref.ignore = false;

		auto N = dmp.GetFeatures(img, ref.features);
		
		printf("Reference image %d : %d features\n", (int)references.size(), (int)N);

		if (N < 4) return false; // need min. 4 features for homography calculation
		
		references.push_back(ref);

		return true;
	}
	
	int ProcessImage(const cv::Mat& m, int minMatches)
	{
		using TimeThis = ScopedDeltaAccumulator<Time::Ticks>;
		
		int returnValue = 0;
		
		perf.process = perf.detect = perf.match = perf.homography = 0;
		
		TimeThis _(perf.process, Time::Now);

		for (auto& ref : references) {
			ref.present = false;
			ref.matches.clear();
		}
					
		// Detect features in input image; if too few, bail here (4 features
        // needed for homography calculation)
		{
			TimeThis _(perf.detect, Time::Now);
			if (dmp.GetFeatures(m, features_) < 4) return returnValue;
		}

		// Check image against reference images. Note that this is O(N); we
		// could instead use some sort of fancy tree to search on features
		// to reduce this complexity, but I'm not sure it's worth it yet.
		for (auto& ref : references) {
			// Skip ignored images
			if (ref.ignore == true) continue;

			// Try to get sufficent matches against reference; if too few,
            // bail without homography calculation
			{
				TimeThis _(perf.match, Time::Now);
				dmp.GetMatches(features_, ref.features, ref.matches);
				if ((int)ref.matches.size() < minMatches) continue;
			}
		
			// Try to get homography transform against reference
			{
				TimeThis _(perf.homography, Time::Now);
				if (Features::FindHomography(
					features_, ref.features,
					ref.matches,
					srcPoints_, dstPoints_,
					ref.transform,
					Features::HomographyType::RHO) != true) continue;
			}
			
			// At this point, we have sufficient matches & a valid homography;
			// consider this reference image to be recognized in the input.
			ref.present = true;
			returnValue++;
		}
				
		return returnValue;
	}

};

}

#endif /* Recognize_hpp */
