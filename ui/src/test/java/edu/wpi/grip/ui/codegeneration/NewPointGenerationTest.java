package edu.wpi.grip.ui.codegeneration;

import edu.wpi.grip.core.ManualPipelineRunner;
import edu.wpi.grip.core.OperationMetaData;
import edu.wpi.grip.core.Step;
import edu.wpi.grip.core.operations.opencv.NewPointOperation;
import edu.wpi.grip.core.sockets.InputSocket;
import edu.wpi.grip.ui.codegeneration.tools.GenType;
import edu.wpi.grip.ui.codegeneration.tools.PipelineInterfacer;

import org.junit.Test;
import org.opencv.core.Point;

import java.util.Optional;

import static org.junit.Assert.assertTrue;

public class NewPointGenerationTest extends AbstractGenerationTest {

  void generatePipeline(double x, double y) {
    Step desat = gen.addStep(new OperationMetaData(NewPointOperation.DESCRIPTION, () -> new
        NewPointOperation(isf, osf)));
    for (InputSocket sock : desat.getInputSockets()) {
      if (sock.getSocketHint().getIdentifier().equals("x")) {
        sock.setValue(x);
      } else if (sock.getSocketHint().getIdentifier().equals("y")) {
        sock.setValue(y);
      }
    }
  }

  @Test
  public void newPointTest() {
    test(() -> {
      generatePipeline(3, 5);
      return true;
    }, (pip) -> testPipeline(pip), "newPointTest");
  }

  void testPipeline(PipelineInterfacer pip) {
    new ManualPipelineRunner(eventBus, pipeline).runPipeline();
    Optional out = pipeline.getSteps().get(0).getOutputSockets().get(0).getValue();
    assertTrue("Output is not present", out.isPresent());
    org.bytedeco.javacpp.opencv_core.Point gripSize = (org.bytedeco.javacpp.opencv_core.Point) out
        .get();
    pip.process();
    Point genSize = (Point) pip.getOutput("New_Point0Output0", GenType.POINT);
    assertTrue("The grip x: " + gripSize.x() + "does not equals the generated x: "
        + genSize.x, gripSize.x() == genSize.x);
    assertTrue("The grip y: " + gripSize.y() + "does not equals the generated y: "
        + genSize.y, gripSize.y() == genSize.y);
  }

}