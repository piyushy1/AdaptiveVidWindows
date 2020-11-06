package benchmark;
import java.io.File;
import java.io.IOException;

import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.opencv.opencv_core.Mat;
import org.nd4j.common.io.ClassPathResource;
import org.opencv.core.CvType;
import org.zeromq.SocketType;
import org.zeromq.ZContext;
import org.zeromq.ZMQ;

import com.espertech.esper.client.Configuration;
import com.espertech.esper.client.EPAdministrator;
import com.espertech.esper.client.EPRuntime;
import com.espertech.esper.client.EPServiceProvider;
import com.espertech.esper.client.EPServiceProviderManager;
import com.espertech.esper.client.EPStatement;
import com.espertech.esper.client.EventBean;
import com.espertech.esper.client.UpdateListener;

public class EsperBenchmark {

	public static void main(String[] args) throws IOException, ClassNotFoundException {
		EsperBenchmark benchmark =new EsperBenchmark();
		EPRuntime cepRT = benchmark.registerAndStartEsperRuntime();
		benchmark.listenImageEvent(cepRT);
	}
	
	public EPRuntime registerAndStartEsperRuntime() throws IOException
	{
		Configuration cepConfig = new Configuration();
		File resource = new File(new ClassPathResource("configuration.xml").getFile().getPath());
		cepConfig.configure(resource);
		cepConfig.addEventTypeAutoName("benchmark");
		EPServiceProvider cep = EPServiceProviderManager.getProvider("myCEPEngine", cepConfig);
		EPRuntime cepRT = cep.getEPRuntime();

		EPAdministrator cepAdm = cep.getEPAdministrator();
		EPStatement cepStatement = cepAdm.createEPL("select classification(window(*)) from Image.win:length(1)");
		
		cepStatement.addListener(new CEPListener());
		
		return cepRT;
	}
	
	public static class CEPListener implements UpdateListener {

		@Override
		public void update(EventBean[] newEvents, EventBean[] oldEvents) {
			System.out.println("Event processed successfully.");
		}

	}

	public void listenImageEvent(EPRuntime cepRT) throws ClassNotFoundException, IOException {
		try (ZContext context = new ZContext()) {
			
            // Socket to talk to clients
            ZMQ.Socket socket = context.createSocket(SocketType.REP);
            socket.bind("tcp://localhost:5555");
            int counter =0;
            long start_time = System.currentTimeMillis();
            while (!Thread.currentThread().isInterrupted()) {
                // Block until a message is received
                byte[] receivedImage = socket.recv(0);
                Mat mat = new Mat(receivedImage);
                
                // imwrite("/home/dhasal/1.png", mat);

                VideoFrame frame = new VideoFrame(mat);

                counter = counter +1;
                cepRT.sendEvent(frame);

                // Send a response
                String response = "OK.";
                socket.send(response.getBytes(ZMQ.CHARSET), 0);
                long diff = start_time-System.currentTimeMillis();
                System.out.println(
                        "Received event from client...Frame..."+counter+" Time..."+diff
                    );
            }
        }
	}
}
	
