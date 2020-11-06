package benchmark;

import org.bytedeco.opencv.opencv_core.Mat;

public class VideoFrame {
	
	Mat mat;
	
	VideoFrame(Mat mat)
	{
		this.mat = mat;
	}

	public Mat getData() {
		return mat;
	}
}
