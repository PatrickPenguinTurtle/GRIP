package edu.wpi.grip.core;

import com.google.common.eventbus.EventBus;

import java.io.InputStream;
import java.util.Optional;

/**
 * The common interface used by <code>Step</code>s in a pipeline to call various operations.  There is usually only one
 * instance of any class that implements <code>Operation</code>, which is called whenever that operation is used.
 */
public interface Operation {

    /**
     * @return The unique user-facing name of the operation, such as "Gaussian Blur"
     */
    String getName();


    /**
     * @return A description of the operation.
     */
    String getDescription();

    /**
     * @return An {@link InputStream} of a 128x128 image to show the user as a representation of the operation.
     */
    default Optional<InputStream> getIcon() {
        return Optional.empty();
    }

    /**
     * @param eventBus The Guava {@link EventBus} used by the application.
     * @return An array of sockets for the inputs that the operation expects.
     */
    InputSocket<?>[] createInputSockets(EventBus eventBus);

    /**
     * @param eventBus The Guava {@link EventBus} used by the application.
     * @return An array of sockets for the outputs that the operation produces.
     */
    OutputSocket<?>[] createOutputSockets(EventBus eventBus);

    /**
     * Perform the operation on the specified inputs, storing the results in the specified outputs.
     *
     * @param inputs  An array obtained from {@link #createInputSockets(EventBus)}. The caller can set the value of
     *                each socket to an actual parameter for the operation.
     * @param outputs An array obtained from {@link #createOutputSockets(EventBus)}. The outputs of the operation will
     *                be stored in these sockets.
     */
    void perform(InputSocket<?>[] inputs, OutputSocket<?>[] outputs);
}
