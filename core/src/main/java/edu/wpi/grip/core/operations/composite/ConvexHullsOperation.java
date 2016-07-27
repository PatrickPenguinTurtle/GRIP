package edu.wpi.grip.core.operations.composite;

import edu.wpi.grip.core.Operation;
import edu.wpi.grip.core.OperationDescription;
import edu.wpi.grip.core.sockets.InputSocket;
import edu.wpi.grip.core.sockets.OutputSocket;
import edu.wpi.grip.core.sockets.SocketHint;

import com.google.common.collect.ImmutableList;

import org.opencv.core.CvType;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfPoint;

import java.util.List;
import java.util.stream.Collectors;

import static org.opencv.imgproc.Imgproc.convexHull;

/**
 * An {@link Operation} that finds the convex hull of each of a list of contours. This can help
 * remove holes in detected shapes, making them easier to analyze.
 */
public class ConvexHullsOperation implements Operation {

  public static final OperationDescription DESCRIPTION =
      OperationDescription.builder()
          .name("Convex Hulls")
          .summary("Compute the convex hulls of contours")
          .category(OperationDescription.Category.FEATURE_DETECTION)
          .build();

  private final SocketHint<ContoursReport> contoursHint =
      new SocketHint.Builder<>(ContoursReport.class)
          .identifier("Contours")
          .initialValueSupplier(ContoursReport::new)
          .build();

  private final InputSocket<ContoursReport> inputSocket;
  private final OutputSocket<ContoursReport> outputSocket;

  @SuppressWarnings("JavadocMethod")
  public ConvexHullsOperation(InputSocket.Factory inputSocketFactory,
                              OutputSocket.Factory outputSocketFactory) {
    this.inputSocket = inputSocketFactory.create(contoursHint);
    this.outputSocket = outputSocketFactory.create(contoursHint);
  }

  @Override
  public List<InputSocket> getInputSockets() {
    return ImmutableList.of(
        inputSocket
    );
  }

  @Override
  public List<OutputSocket> getOutputSockets() {
    return ImmutableList.of(
        outputSocket
    );
  }

  @Override
  public void perform() {
    final List<MatOfPoint> inputContours = inputSocket.getValue().get().getContours();
    final List<MatOfPoint> outputContours =
        inputContours
            .stream()
            .map(ConvexHullsOperation::hull)
            .collect(Collectors.toList());

    outputSocket.setValue(new ContoursReport(outputContours,
        inputSocket.getValue().get().getRows(), inputSocket.getValue().get().getCols()));
  }

  private static MatOfPoint hull(MatOfPoint contour) {
    MatOfInt hull = new MatOfInt();
    convexHull(contour, hull);
    MatOfPoint mopHull = new MatOfPoint();
    mopHull.create((int) hull.size().height, 1, CvType.CV_32SC2);
    for (int hullIdx = 0; hullIdx < hull.size().height; hullIdx++) {
      int contIdx = (int) hull.get(hullIdx, 0)[0];
      double[] pointVals = contour.get(contIdx, 0);
      double[] point = new double[] {pointVals[0], pointVals[1]};
      mopHull.put(hullIdx, 0, point);
    }
    return mopHull;
  }

}
