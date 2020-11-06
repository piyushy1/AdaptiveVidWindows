package benchmark;

import java.io.IOException;

import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.ResNet50;
import org.deeplearning4j.zoo.util.imagenet.ImageNetLabels;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.bytedeco.opencv.opencv_video.*;



@SuppressWarnings("unchecked")
public class ObjectClassification {
	
	static ComputationGraph  model = null;
	static{
        try{
        	final ZooModel<ResNet50> zooModel = ResNet50.builder().build();;
        	model  = (ComputationGraph)zooModel.initPretrained(PretrainedType.IMAGENET);
        }catch(Exception e){
            throw new RuntimeException("Exception occured in creating model instance");
        }
    }
	
	public static void classifyObject(Object frameArray) throws IOException{
    	VideoFrame[] frames=(VideoFrame[])frameArray;
    	
    	for (VideoFrame frame: frames)
    	{
    		NativeImageLoader loader = new NativeImageLoader(224, 224, 3);
    		INDArray indArray = loader.asMatrix(frame.getData());
    		ImagePreProcessingScaler imagePreProcessingScaler = new ImagePreProcessingScaler(0, 1);
    		imagePreProcessingScaler.transform(indArray);
    		INDArray results = model.outputSingle(indArray);
        	ImageNetLabels lables = new ImageNetLabels();
        	String out = lables.decodePredictions(results);
    		System.out.println(out);
    	}
    }
}

