package edu.wpi.grip.ui.codegeneration.cv;

import edu.wpi.grip.core.ManualPipelineRunner;
import edu.wpi.grip.core.OperationMetaData;
import edu.wpi.grip.core.Step;
import edu.wpi.grip.core.operations.composite.DesaturateOperation;
import edu.wpi.grip.core.sockets.InputSocket;
import edu.wpi.grip.core.sockets.OutputSocket;
import edu.wpi.grip.core.sources.ImageFileSource;
import edu.wpi.grip.generated.opencv_imgproc.enumeration.AdaptiveThresholdTypesEnum;
import edu.wpi.grip.generated.opencv_imgproc.enumeration.ThresholdTypesEnum;
import edu.wpi.grip.ui.codegeneration.AbstractGenerationTest;
import edu.wpi.grip.ui.codegeneration.tools.HelperTools;
import edu.wpi.grip.ui.codegeneration.tools.PipelineInterfacer;
import edu.wpi.grip.util.Files;

import org.junit.Test;
import org.opencv.core.Mat;

import java.util.Optional;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

public class CVAdaptiveThreshold extends AbstractGenerationTest {
  private final double thresh = 50;
  private final double maxval = 200;

  boolean setup(AdaptiveThresholdTypesEnum adaptMethod, ThresholdTypesEnum threshMethod) {
    Step desat = gen.addStep(new OperationMetaData(DesaturateOperation.DESCRIPTION, () -> new
        DesaturateOperation(isf, osf)));
    ImageFileSource img = loadImage(Files.gompeiJpegFile);
    OutputSocket imgOut = pipeline.getSources().get(0).getOutputSockets().get(0);
    for (InputSocket sock : desat.getInputSockets()) {
      if (sock.getSocketHint().getIdentifier().equals("Input")) {
        gen.connect(imgOut, sock);
      }
    }

    Step step = gen.addStep(opUtil.getMetaData("CV adaptiveThreshold"));
    OutputSocket deImg = desat.getOutputSockets().get(0);
    gen.connect(deImg, step.getInputSockets().get(0));
    step.getInputSockets().get(1).setValue(new Double(100.0));
    step.getInputSockets().get(2).setValue(adaptMethod);
    step.getInputSockets().get(3).setValue(threshMethod);
    step.getInputSockets().get(4).setValue(new Double(5.0));
    return true;
  }

  @Test
  public void binaryMeanTest() {
    test(() -> setup(AdaptiveThresholdTypesEnum.ADAPTIVE_THRESH_MEAN_C, 
        ThresholdTypesEnum.THRESH_BINARY), (pip) -> validate(pip), "cvBinaryMeanAT");
  }

  @Test
  public void binaryInvMeanTest() {
    test(() -> setup(AdaptiveThresholdTypesEnum.ADAPTIVE_THRESH_MEAN_C,
        ThresholdTypesEnum.THRESH_BINARY_INV), (pip) -> validate(pip), "cvBinaryInvMeanAT");
  }

  @Test
  public void binaryGaussianTest() {
    test(() -> setup(AdaptiveThresholdTypesEnum.ADAPTIVE_THRESH_GAUSSIAN_C,
        ThresholdTypesEnum.THRESH_BINARY), (pip) -> validate(pip), "cvBinaryGaussianAT");
  }

  @Test
  public void binaryInvGaussianTest() {
    test(() -> setup(AdaptiveThresholdTypesEnum.ADAPTIVE_THRESH_GAUSSIAN_C,
        ThresholdTypesEnum.THRESH_BINARY_INV), (pip) -> validate(pip), "cvBinaryInvGaussianAT");
  }

  void validate(PipelineInterfacer pip) {
    ManualPipelineRunner runner = new ManualPipelineRunner(eventBus, pipeline);
    runner.runPipeline();
    pip.setMatSource(0, Files.gompeiJpegFile.file);
    pip.process();
    Optional out = pipeline.getSteps().get(1).getOutputSockets().get(0).getValue();
    assertTrue("Pipeline did not process", out.isPresent());
    assertFalse("Pipeline output is empty", ((org.bytedeco.javacpp.opencv_core.Mat) out.get())
        .empty());
    Mat genMat = (Mat) pip.getOutput(1);
    Mat gripMat = HelperTools.bytedecoMatToCVMat((org.bytedeco.javacpp.opencv_core.Mat) out.get());
    assertMatWithin(genMat, gripMat, 1.0);
  }
}
