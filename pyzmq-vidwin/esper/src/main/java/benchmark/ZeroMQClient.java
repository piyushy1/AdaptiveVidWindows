package benchmark;

import java.awt.image.DataBufferByte;
import java.io.File;
import java.io.IOException;

import org.bytedeco.javacv.FFmpegFrameGrabber;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.Java2DFrameUtils;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.opencv.opencv_core.Mat;
import org.zeromq.SocketType;
import org.zeromq.ZContext;
import org.zeromq.ZMQ;
import java.util.concurrent.TimeUnit;

public class ZeroMQClient {

	public static void main(String[] args) throws IOException, InterruptedException {
		try (ZContext context = new ZContext()) {

			// Socket to talk to clients
			ZMQ.Socket socket = context.createSocket(SocketType.REQ);
			socket.connect("tcp://localhost:5555");
			
			File f = new File("/home/dhaval/piyush/ViIDWIN/Datasets_VIDWIN/Cropped Datasets for Processing/test2.mp4");

			FFmpegFrameGrabber grabber;
			grabber = new FFmpegFrameGrabber(f);
			grabber.start();
			
			while (true) 
			{
				Frame videoFrame = grabber.grab();
				if (videoFrame == null) {
					System.out.println("Frame is null. Existing...");
					break;
				}
				
				Mat mat = new OpenCVFrameConverter.ToMat().convert(videoFrame);
				if(mat == null)
				{
					//System.out.println("Mat object is null. Retrying...");
					continue;
				}
				byte[] b = new byte[mat.channels() * mat.cols() * mat.rows()];
		        mat.data().get(b);
		        Thread.sleep(35);
		        //byte[] b = ((DataBufferByte) Java2DFrameUtils.toBufferedImage(mat).getRaster().getDataBuffer()).getData();
		        // System.out.println("Sending frame...");
		        socket.send(b, 0);
		        socket.recv(0);
			}
			grabber.close();
		}
	}

}
