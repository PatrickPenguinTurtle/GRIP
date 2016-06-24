package edu.wpi.grip.ui.codegeneration;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.util.ArrayList;
import java.util.Optional;

import org.junit.Test;
import org.opencv.core.Mat;

import edu.wpi.grip.core.ManualPipelineRunner;
import edu.wpi.grip.core.OperationMetaData;
import edu.wpi.grip.core.Step;
import edu.wpi.grip.core.operations.composite.HSVThresholdOperation;
import edu.wpi.grip.core.operations.composite.MaskOperation;
import edu.wpi.grip.core.sockets.InputSocket;
import edu.wpi.grip.core.sockets.OutputSocket;
import edu.wpi.grip.core.sources.ImageFileSource;
import edu.wpi.grip.ui.codegeneration.tools.HelperTools;
import edu.wpi.grip.ui.codegeneration.tools.PipelineInterfacer;
import edu.wpi.grip.util.Files;

public class MaskTest extends AbstractGenerationTest {

	void setup(){
		HSVThresholdSetup.setup(this);
		Step mask = gen.addStep(new OperationMetaData(
				MaskOperation.DESCRIPTION, () -> new MaskOperation(isf,osf)));
		OutputSocket hsvImg = pipeline.getSteps().get(0).getOutputSockets().get(0);
		for(InputSocket sock : mask.getInputSockets()){
			String sockHint = sock.getSocketHint().getIdentifier();
			if(sockHint.equals("Input")){
				gen.connect(pipeline.getSources().get(0).getOutputSockets().get(0), sock);
			}
			else if(sockHint.equals("Mask")){
				gen.connect(hsvImg, sock);
			}
		}
	}
	
	@Test
	public void maskTest(){
		test( () -> {setup(); return true;}, (pip) -> validate(pip), "MaskGripIcon");
	}
	
	void validate(PipelineInterfacer pip){
		new ManualPipelineRunner(eventBus, pipeline).runPipeline();
		Optional out = pipeline.getSteps().get(1).getOutputSockets().get(0).getValue();
		assertTrue("Pipeline did not process", out.isPresent());
		assertFalse("Pipeline output is empty", ((org.bytedeco.javacpp.opencv_core.Mat)out.get()).empty());
		pip.setMatSource(0, Files.imageFile.file);
		pip.setMatSource(1, Files.imageFile.file);
		pip.process();
		Mat genMat = (Mat) pip.getOutput(1);
		Mat gripMat = HelperTools.bytedecoMatToCVMat((org.bytedeco.javacpp.opencv_core.Mat)out.get());
		assertMatWithin(genMat, gripMat, 0.5);
	}
}
