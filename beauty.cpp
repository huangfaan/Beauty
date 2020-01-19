#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <ctime>
#include <vector>

using namespace std;

cv::Mat add_edge_details(cv::Mat img, cv::Mat buffer_image, double p) {
	cv::Mat highPass, gauss;
	//得到高反差图像
	cv::GaussianBlur(img, gauss, cv::Size(5, 5), 0);
	cv::subtract(img, gauss, highPass);
	
	cv::Mat buff = buffer_image - img + cv::Mat(img.size(), img.type(), cv::Scalar::all(128));
	cv::GaussianBlur(buff, buff, cv::Size(3, 3), 0);
	cv::Mat temp = buff + buff - 256 + img;
	cv::Mat dst = highPass * p + temp;
	//cv::addWeighted(img, 0.1, dst, 0.9, 0, dst);
	return dst;
}


cv::Mat guidedFilter(cv::Mat i_, cv::Mat p_, int r, double eps) {
	cv::Mat I;
	cv::Mat p;
	i_.convertTo(I, CV_32FC1);
	p_.convertTo(p, CV_32FC1);

	//步骤一：计算均值
	cv::Mat mean_I, mean_p, mean_Ip, mean_II, cov_Ip;
	cv::boxFilter(I, mean_I, CV_32FC1, cv::Size(r, r));
	cv::boxFilter(p, mean_p, CV_32FC1, cv::Size(r, r));
	cv::boxFilter(I.mul(p), mean_Ip, CV_32FC1, cv::Size(r, r));
	cv::boxFilter(I.mul(I), mean_II, CV_32FC1, cv::Size(r, r));
	cov_Ip = mean_Ip - mean_I.mul(mean_p);

	//步骤二：计算相关系数 
	cv::Mat var_I = mean_II - mean_I.mul(mean_I);

	//步骤三：计算参数系数a,b
	cv::Mat a = cov_Ip / (var_I + eps);
	cv::Mat b = mean_p - a.mul(mean_I);

	//步骤四：计算系数a\b的均值  
	cv::Mat mean_a, mean_b;
	cv::boxFilter(a, mean_a, CV_32FC1, cv::Size(r, r));
	cv::boxFilter(b, mean_b, CV_32FC1, cv::Size(r, r));

	//步骤五：生成输出矩阵
	cv::Mat q = mean_a.mul(I) + mean_b;
	return q;
}


cv::Mat fastGuidedFilter(cv::Mat I_org, cv::Mat p_org, int r, double eps, int s)
{
	int h = I_org.rows;
	int w = I_org.cols;
	cv::Mat I, p;
	resize(I_org, I, cv::Size(int(round(w / s)), int(round(h / s))), 1);
	resize(I_org, p, cv::Size(int(round(w / s)), int(round(h / s))), 1);

	int small_winSize = int(round(r / s));

	cv::Mat mean_I, mean_p, mean_Ip, mean_II;
	cv::boxFilter(I, mean_I, CV_32FC1, cv::Size(small_winSize, small_winSize));
	cv::boxFilter(p, mean_p, CV_32FC1, cv::Size(small_winSize, small_winSize));
	cv::boxFilter(I.mul(p), mean_Ip, CV_32FC1, cv::Size(small_winSize, small_winSize));
	cv::boxFilter(I.mul(I), mean_II, CV_32FC1, cv::Size(small_winSize, small_winSize));
	cv::Mat cov_Ip = mean_Ip - mean_I.mul(mean_p);
	cv::Mat var_I = mean_II - mean_I.mul(mean_I);

	cv::Mat a = cov_Ip / (var_I + eps);
	cv::Mat b = mean_p - a.mul(mean_I);

	cv::Mat mean_a, rmean_a, mean_b, rmean_b;
	cv::boxFilter(a, mean_a, CV_32FC1, cv::Size(small_winSize, small_winSize));
	resize(mean_a, rmean_a, cv::Size(w, h), 1);
	cv::boxFilter(b, mean_b, CV_32FC1, cv::Size(small_winSize, small_winSize));
	resize(mean_b, rmean_b, cv::Size(w, h), 1);

	cv::Mat q = rmean_a.mul(I_org) + rmean_b;
	return q;
}


vector<int> get_points(std::string txtFile) {
	std::ifstream myfile(txtFile);
	vector<int> res;
	string temp;
	while (getline(myfile, temp)) //按行读取字符串 
	{
		res.push_back(std::stoi(temp));//写文件
	}
	myfile.close();
	return res;
}


cv::Mat get_mask_68lms(cv::Mat src_img, vector<int> points) {
	//cv::Mat mask = cv::Mat(src_img.rows, src_img.cols, CV_32FC1, cv::Scalar(0));
	cv::Mat mask_eye = cv::Mat(src_img.rows, src_img.cols, CV_32FC1, cv::Scalar(0));
	cv::Point right_points[6];
	cv::Point left_points[6];
	//cv::Point face_points[30];
	//
	//face_points[0] = cv::Point(points[0 * 2], points[0 * 2 + 1]); //脸部对应点
	//face_points[1] = cv::Point(points[1 * 2], points[1 * 2 + 1]);
	//face_points[2] = cv::Point(points[2 * 2], points[2 * 2 + 1]);
	//face_points[3] = cv::Point(points[3 * 2], points[3 * 2 + 1]);
	//face_points[4] = cv::Point(points[4 * 2], points[4 * 2 + 1]);
	//face_points[5] = cv::Point(points[5 * 2], points[5 * 2 + 1]);
	//face_points[6] = cv::Point(points[6 * 2], points[6 * 2 + 1]);
	//face_points[7] = cv::Point(points[7 * 2], points[7 * 2 + 1]);
	//face_points[8] = cv::Point(points[8 * 2], points[8 * 2 + 1]);
	//face_points[9] = cv::Point(points[9 * 2], points[9 * 2 + 1]);
	//face_points[10] = cv::Point(points[10 * 2], points[10 * 2 + 1]);
	//face_points[11] = cv::Point(points[11 * 2], points[11 * 2 + 1]);
	//face_points[12] = cv::Point(points[12 * 2], points[12 * 2 + 1]);
	//face_points[13] = cv::Point(points[13 * 2], points[13 * 2 + 1]);
	//face_points[14] = cv::Point(points[14 * 2], points[14 * 2 + 1]);
	//face_points[15] = cv::Point(points[15 * 2], points[15 * 2 + 1]);
	//face_points[16] = cv::Point(points[16 * 2], points[16 * 2 + 1]);

	//int x = points[27 * 2] * 2 - points[51 * 2];
	//int y = points[27 * 2 + 1] * 2 - points[51 * 2 + 1];
	//if (y < 0)
	//	y = 0;
	//face_points[17] = cv::Point(points[27 * 2] + int((points[16 * 2] - points[27 * 2]) * 29 / 30), y + int((points[16 * 2 + 1] - y) / 8));
	//face_points[18] = cv::Point(points[27 * 2] + int((points[16 * 2] - points[27 * 2]) * 9 / 10), y + int((points[16 * 2 + 1] - y) / 10));
	//face_points[19] = cv::Point(points[27 * 2] + int((points[16 * 2] - points[27 * 2]) * 4 / 5), y + int((points[16 * 2 + 1] - y) / 12));
	//face_points[20] = cv::Point(points[27 * 2] + int((points[16 * 2] - points[27 * 2]) * 3 / 5), y + int((points[16 * 2 + 1] - y) / 15));
	//face_points[21] = cv::Point(points[27 * 2] + int((points[16 * 2] - points[27 * 2]) * 2 / 5), y + int((points[16 * 2 + 1] - y) / 20));
	//face_points[22] = cv::Point(points[27 * 2] + int((points[16 * 2] - points[27 * 2]) / 5), y + int((points[16 * 2 + 1] - y) / 60));
	//face_points[23] = cv::Point(x, y);
	//face_points[24] = cv::Point(points[0 * 2] + int((points[27 * 2] - points[0 * 2]) * 4 / 5), y + int((points[0 * 2 + 1] - y) / 60));
	//face_points[25] = cv::Point(points[0 * 2] + int((points[27 * 2] - points[0 * 2]) * 3 / 5), y + int((points[0 * 2 + 1] - y) / 20));
	//face_points[26] = cv::Point(points[0 * 2] + int((points[27 * 2] - points[0 * 2]) * 2 / 5), y + int((points[0 * 2 + 1] - y) / 15));
	//face_points[27] = cv::Point(points[0 * 2] + int((points[27 * 2] - points[0 * 2]) / 5), y + int((points[0 * 2 + 1] - y) / 12));
	//face_points[28] = cv::Point(points[0 * 2] + int((points[27 * 2] - points[0 * 2]) / 10), y + int((points[0 * 2 + 1] - y) / 10));
	//face_points[29] = cv::Point(points[0 * 2] + int((points[27 * 2] - points[0 * 2]) / 30), y + int((points[0 * 2 + 1] - y) / 8));

	//cv::fillConvexPoly(mask, face_points, 30, 1);

	right_points[0] = cv::Point(points[36 * 2], points[36 * 2 + 1]);//右眼对应点
	right_points[1] = cv::Point(points[37 * 2], points[37 * 2 + 1]);
	right_points[2] = cv::Point(points[38 * 2], points[38 * 2 + 1]);
	right_points[3] = cv::Point(points[39 * 2], points[39 * 2 + 1]);
	right_points[4] = cv::Point(points[40 * 2], points[40 * 2 + 1]);
	right_points[5] = cv::Point(points[41 * 2], points[41 * 2 + 1]);

	left_points[0] = cv::Point(points[42 * 2], points[42 * 2 + 1]); // 左眼对应点
	left_points[1] = cv::Point(points[43 * 2], points[43 * 2 + 1]);
	left_points[2] = cv::Point(points[44 * 2], points[44 * 2 + 1]);
	left_points[3] = cv::Point(points[45 * 2], points[45 * 2 + 1]);
	left_points[4] = cv::Point(points[46 * 2], points[46 * 2 + 1]);
	left_points[5] = cv::Point(points[47 * 2], points[47 * 2 + 1]);

	cv::fillConvexPoly(mask_eye, left_points, 6, 1);
	cv::fillConvexPoly(mask_eye, right_points, 6, 1);

	int g_nStructRlementSize = 3;//内核矩阵的尺寸   
	cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2 * g_nStructRlementSize + 1, 2 * g_nStructRlementSize + 1), cv::Point(g_nStructRlementSize, g_nStructRlementSize));
	cv::dilate(mask_eye, mask_eye, element, cv::Point(-1, -1), 1); //腐蚀1次

	//return mask - mask_eye;
	return cv::Scalar(1.0, 1.0, 1.0) - mask_eye;
}


cv::Mat get_mask_72lms(cv::Mat src_img, vector<int> points) {
	//cv::Mat mask = cv::Mat(src_img.rows, src_img.cols, CV_32FC1, cv::Scalar(0));
	cv::Mat mask_eye = cv::Mat(src_img.rows, src_img.cols, CV_32FC1, cv::Scalar(0));
	cv::Point right_points[8];
	cv::Point left_points[8];
	//cv::Point face_points[16];

	//face_points[0] = cv::Point(points[0 * 2], points[0 * 2 + 1]); //脸部对应点
	//face_points[1] = cv::Point(points[1 * 2], points[1 * 2 + 1]);
	//face_points[2] = cv::Point(points[2 * 2], points[2 * 2 + 1]);
	//face_points[3] = cv::Point(points[3 * 2], points[3 * 2 + 1]);
	//face_points[4] = cv::Point(points[4 * 2], points[4 * 2 + 1]);
	//face_points[5] = cv::Point(points[5 * 2], points[5 * 2 + 1]);
	//face_points[6] = cv::Point(points[6 * 2], points[6 * 2 + 1]);
	//face_points[7] = cv::Point(points[7 * 2], points[7 * 2 + 1]);
	//face_points[8] = cv::Point(points[8 * 2], points[8 * 2 + 1]);
	//face_points[9] = cv::Point(points[9 * 2], points[9 * 2 + 1]);
	//face_points[10] = cv::Point(points[10 * 2], points[10 * 2 + 1]);
	//face_points[11] = cv::Point(points[11 * 2], points[11 * 2 + 1]);
	//face_points[12] = cv::Point(points[12 * 2], points[12 * 2 + 1]);
	//face_points[13] = cv::Point(points[13 * 2], points[13 * 2 + 1]);
	//face_points[14] = cv::Point(points[14 * 2], points[14 * 2 + 1]);
	//face_points[15] = cv::Point(points[15 * 2], points[15 * 2 + 1]);

	//cv::fillConvexPoly(mask, face_points, 16, 1);

	right_points[0] = cv::Point(points[30 * 2], points[30 * 2 + 1]);//右眼对应点
	right_points[1] = cv::Point(points[31 * 2], points[31 * 2 + 1]);
	right_points[2] = cv::Point(points[32 * 2], points[32 * 2 + 1]);
	right_points[3] = cv::Point(points[33 * 2], points[33 * 2 + 1]);
	right_points[4] = cv::Point(points[34 * 2], points[34 * 2 + 1]);
	right_points[5] = cv::Point(points[35 * 2], points[35 * 2 + 1]);
	right_points[6] = cv::Point(points[36 * 2], points[36 * 2 + 1]);
	right_points[7] = cv::Point(points[37 * 2], points[37 * 2 + 1]);

	left_points[0] = cv::Point(points[40 * 2], points[40 * 2 + 1]); // 左眼对应点
	left_points[1] = cv::Point(points[41 * 2], points[41 * 2 + 1]);
	left_points[2] = cv::Point(points[42 * 2], points[42 * 2 + 1]);
	left_points[3] = cv::Point(points[43 * 2], points[43 * 2 + 1]);
	left_points[4] = cv::Point(points[44 * 2], points[44 * 2 + 1]);
	left_points[5] = cv::Point(points[45 * 2], points[45 * 2 + 1]);
	left_points[6] = cv::Point(points[46 * 2], points[46 * 2 + 1]);
	left_points[7] = cv::Point(points[47 * 2], points[47 * 2 + 1]);

	cv::fillConvexPoly(mask_eye, left_points, 8, 1);
	cv::fillConvexPoly(mask_eye, right_points, 8, 1);

	int g_nStructRlementSize = 3;//内核矩阵的尺寸   
	cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2 * g_nStructRlementSize + 1, 2 * g_nStructRlementSize + 1), cv::Point(g_nStructRlementSize, g_nStructRlementSize));
	cv::dilate(mask_eye, mask_eye, element, cv::Point(-1, -1), 1); //腐蚀1次

	//return mask - mask_eye;
	return cv::Scalar(1.0, 1.0, 1.0) - mask_eye;
}


cv::Mat buffing(cv::Mat src_img, int r, double eps, int s) {
	cv::Mat resultMat;
	vector<cv::Mat> vSrcImage, vResultImage;
	split(src_img, vSrcImage);
	for (int i = 0; i < 3; i++) {
		cv::Mat tempImage;
		vSrcImage[i].convertTo(tempImage, CV_32FC1, 1.0 / 255);
		cv::Mat p = tempImage.clone();
		cv::Mat resultImage = fastGuidedFilter(tempImage, p, r, eps, s);
		resultImage.convertTo(resultImage, CV_8UC1, 255.0);
		vResultImage.push_back(resultImage);
	}
	merge(vResultImage, resultMat);
	return resultMat;
}


cv::Mat usingMask(cv::Mat src1, cv::Mat src2, cv::Mat mask) {
	cv::GaussianBlur(mask, mask, cv::Size(7, 7), 0);
	cv::Mat mask_c = cv::Scalar(1.0, 1.0, 1.0) - mask;

	cv::Mat result;
	cv::blendLinear(src1, src2, mask_c, mask, result);
	return result;
}



bool beauty(cv::Mat src, vector<int> lms, int lm_count, cv::Mat& result, double weight) {
	/*
	src: 需要磨皮的输入图像
	lms: 人脸特征点  {x0, y0, x1, y1, ... xn, yn}
	lm_count: 人脸特征点个数  68 / 77
	result: 磨皮的输出效果图
	weight: 磨皮权重，范围0-1，正常默认取0.3即可

	return: 成功与否
	*/

	try {
		if (weight < 0)
			weight = 0;
		else if (weight > 1)
			weight = 1.0;
		double esp = weight * weight * 0.02;

		cv::Mat mask;
		if (lm_count == 68)
			mask = get_mask_68lms(src, lms);
		else if (lm_count == 77)
			mask = get_mask_72lms(src, lms);
		else
			return false;

		cv::Mat buffer_img = buffing(src, 10, esp, 2);
		cv::Mat dst = add_edge_details(src, buffer_img, 0.8);
		result = usingMask(src, dst, mask);
		return true;

	}
	catch (exception & e)
	{
		cout << e.what() << endl;
		return false;
	}
}


int main(int argc, char** argv) {
	cv::String pattern = "E:/ND/cpp_beauty/beauty/images/test_1/*.jpg";
	vector<cv::String> fn;
	cv::glob(pattern, fn, false);
	size_t count = fn.size(); //number of png files in images folder
	for (size_t i = 0; i < count; i++)
	{
		cv::Mat src_img = cv::imread(fn[i]);
		if (src_img.empty()) {
			printf("Empty!");
			return -1;
		}

		size_t pos = fn[i].find(".");
		std::string txtFile = fn[i].substr(0, pos) + ".txt";
		vector<int> points = get_points(txtFile);

		clock_t start, end;
		start = clock();
		//cv::Mat buffer_img;
		//cv::Mat mask = get_mask_68lms(src_img, points);
		//buffer_img = buffing(src_img, 10, 0.002, 2);
		//cv::Mat dst = add_edge_details(src_img, buffer_img, 0.8);
		//dst = usingMask(src_img, dst, mask);
		cv::Mat abc;
		bool aaa = beauty(src_img, points, 68, abc, 0.3);
		end = clock();
		double endtime = (double)(end - start) / CLOCKS_PER_SEC;
		std::cout << "Total time: " << endtime * 1000 << "ms" << std::endl;	//ms为单位
		cv::imshow("dst", abc);
		cv::waitKey(0);
		//cv::putText(src_img, "src", cv::Point(10, 30), 1, 2, cv::Scalar(0, 255, 0), 2);
		//string text = "dst " + to_string(int(endtime * 1000)) + " ms";
		//cv::putText(dst, text, cv::Point(10, 30), 1, 2, cv::Scalar(0, 255, 0), 2);
		//// 横向拼接图片
		//cv::Mat res = cv::Mat::zeros(src_img.rows, src_img.cols * 2, src_img.type());
		//cv::Rect rect = cv::Rect(0, 0, src_img.cols, src_img.rows);
		//cv::Mat dstMat = res(rect);
		//src_img.colRange(0, src_img.cols).copyTo(dstMat);

		////rect = cv::Rect(src_img.cols, 0, src_img.cols, src_img.rows);
		////dstMat = res(rect);
		////cv::Mat mask255;
		////vector<cv::Mat> vmask;
		////vmask.push_back(mask);
		////vmask.push_back(mask);
		////vmask.push_back(mask);
		////merge(vmask, mask255);
		////mask255 = mask255 * 255;
		////mask255.colRange(0, mask.cols).copyTo(dstMat);
		////rect = cv::Rect(src_img.cols, 0, src_img.cols, src_img.rows);
		////dstMat = res(rect);
		////buffer_img.colRange(0, buffer_img.cols).copyTo(dstMat);

		//rect = cv::Rect(src_img.cols, 0, src_img.cols, src_img.rows);
		//dstMat = res(rect);
		//dst.colRange(0, dst.cols).copyTo(dstMat);

		//cv::imwrite("E:/ND/cpp_beauty/beauty/images/out/" + to_string(i) + "_.jpg", res);
	}
	_CrtDumpMemoryLeaks();
	return 0;

}
