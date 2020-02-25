import argparse
import cv2
from inference import Network

INPUT_STREAM = "test_video.mp4"
CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"

def get_args():
    '''
    Gets the arguments from the command line.
    '''
    parser = argparse.ArgumentParser("Run inference on an input video")
    # -- Create the descriptions for the commands
    m_desc = "The location of the model XML file"
    i_desc = "The location of the input file"
    d_desc = "The device name, if not 'CPU'"
    ### TODO: Add additional arguments and descriptions for:
    ###       1) Different confidence thresholds used to draw bounding boxes
    ###       2) The user choosing the color of the bounding boxes
    ct_desc = "Set the confidence threshold"
    c_desc = "Set the box color"
    # -- Add required and optional groups
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    # -- Create the arguments
    required.add_argument("-m", help=m_desc, required=True)
    optional.add_argument("-i", help=i_desc, default=INPUT_STREAM)
    optional.add_argument("-d", help=d_desc, default='CPU')
    optional.add_argument("-ct", help=ct_desc, default="0.5")
    optional.add_argument("-c", help=c_desc, default="GREEN")
    args = parser.parse_args()

    return args

def infer_on_video(args):
    ### TODO: Initialize the Inference Engine
    plugin = Network()
    ### TODO: Load the network model into the IE
    plugin.load_model(args.m, device="CPU", cpu_extension=CPU_EXTENSION)
    # Get and open video capture
    cap = cv2.VideoCapture(args.i)
    cap.open(args.i)

    # Grab the shape of the input 
    width = int(cap.get(3))
    height = int(cap.get(4))

    # Create a video writer for the output video
    # The second argument should be `cv2.VideoWriter_fourcc('M','J','P','G')`
    # on Mac, and `0x00000021` on Linux
    out = cv2.VideoWriter('out1.mp4', 0x00000021, 30, (width,height))
    
    in_shape = plugin.get_input_shape()
    
    # Process frames until the video ends, or process is exited
    while cap.isOpened():
        # Read the next frame
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)

        ### TODO: Pre-process the frame
        preprocessed_frame = cv2.resize(frame, (in_shape[3], in_shape[2]))
        preprocessed_frame = preprocessed_frame.transpose((2,0,1))
        preprocessed_frame = preprocessed_frame.reshape(1, *preprocessed_frame.shape)
        
        ### TODO: Perform inference on the frame
        plugin.async_inference(preprocessed_frame)
        
        ### TODO: Get the output of inference
        if plugin.wait() == 0:
            rslt = plugin.extract_output()
            ### TODO: Update the frame to include detected bounding boxes
            if args.c == "BLUE":
                color = (255, 0, 0)
            elif args.c == "GREEN":
                color = (0, 255, 0)
            elif args.c == "RED":
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)
        
            for box in rslt[0][0]:
                ct = box[2]
                if ct >= float(args.ct):
                    xmin = int(box[3] * width)
                    ymin = int(box[4] * height)
                    xmax = int(box[5] * width)
                    ymax = int(box[6] * height)
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                    
            # Write out the frame
            out.write(frame)
            
        # Break if escape key pressed
        if key_pressed == 27:
            break

    # Release the out writer, capture, and destroy any OpenCV windows
    out.release()
    cap.release()
    cv2.destroyAllWindows()


def main():
    args = get_args()
    infer_on_video(args)


if __name__ == "__main__":
    main()
