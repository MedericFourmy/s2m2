import os
import argparse
import numpy as np
import cv2
import torch
import onnxruntime

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from s2m2.s2m2 import load_model
from s2m2.config import S2M2_PRETRAINED_WEIGHTS_PATH
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')


def get_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_type', default='XL', type=str,
                        help='select model type: S,M,L,XL')
    parser.add_argument('--num_refine', default=1, type=int,
                        help='number of local iterative refinement')
    parser.add_argument('--no_export', action='store_true', help='avoid exporting (assumes we already exported this model)')
    parser.add_argument('--allow_negative', action='store_true', help='allow negative disparity for imperfect rectification')
    parser.add_argument('--img_height', default=800, type=int,
                        help='image height')
    parser.add_argument('--img_width', default=1088, type=int,
                        help='image width')
    parser.add_argument('--nb_runs', default=1, type=int)
    return parser

def main(args):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)

    img_height, img_width = args.img_height, args.img_width

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')


    if args.allow_negative:
        left_path = 'samples/Web/64648_pbz98_3D_MPO_70pc_L.jpg'
        right_path = 'samples/Web/64648_pbz98_3D_MPO_70pc_R.jpg'
    else:
        left_path = 'samples/Web/0025_L.png'
        right_path = 'samples/Web/0025_R.png'

    # load stereo images
    left = cv2.cvtColor(cv2.imread(left_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    right = cv2.cvtColor(cv2.imread(right_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


    img_height = (img_height // 32) * 32
    img_width = (img_width // 32) * 32
    print(f"image size: {img_height}, {img_width}")


    left = cv2.resize(left, dsize=(img_width, img_height))
    right = cv2.resize(right, dsize=(img_width, img_height))

    left_torch = (torch.from_numpy(left).permute(-1, 0, 1).unsqueeze(0)).half().to(device)
    right_torch = (torch.from_numpy(right).permute(-1, 0, 1).unsqueeze(0)).half().to(device)

    torch_version = torch.__version__
    onnx_path = os.path.join('onnx_save', f'S2M2_{args.model_type}_{img_width}_{img_height}_v2_torch{torch_version[0]}{torch_version[2]}.onnx')

    print("ONNX_PATH:", onnx_path)
    if not args.no_export:
        os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
        
        model = load_model(
            S2M2_PRETRAINED_WEIGHTS_PATH,
            args.model_type,
            args.allow_negative,
            args.num_refine,
        ).to(device).eval()

        # model = model.half().cpu()
        # left_torch, right_torch = left_torch.half().cpu(), right_torch.half().cpu()

        model = model.half()
        left_torch, right_torch = left_torch.half(), right_torch.half()

        print(f"ONNX conversion takes a long time, even for small model")
        torch.onnx.export(model,
                        (left_torch, right_torch),
                        onnx_path,
                        opset_version=17,
                        do_constant_folding=False,
                        # verbose=False,
                        input_names=['input_left', 'input_right'],
                        output_names=['output_disp', 'output_occ', 'output_conf'],
                        # dynamic_axes=None
        )
        print('success onnx conversion')
    else:
        print("")


    # test onnx file with onnxruntime (provider Tensorrt, fallbcak to CUDA, fallback to CPU)
    # sess = onnxruntime.InferenceSession(onnx_path, providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])
    sess = onnxruntime.InferenceSession(onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    print("Active providers:", sess.get_providers())
    print(f"onnxruntime device: {onnxruntime.get_device()}")

    input_name = [input.name for input in sess.get_inputs()]
    output_name = [output.name for output in sess.get_outputs()]
    print(f"input_name:{input_name}")
    print(f"output_name:{output_name}")

    print("Warmup")
    outputs = sess.run([output_name[0], output_name[1], output_name[2]],
                        {input_name[0]: left_torch.cpu().numpy(),
                        input_name[1]: right_torch.cpu().numpy()})


    print(f"Running {args.nb_runs} times")
    import time
    t1 = time.perf_counter()
    for _ in range(args.nb_runs):
        outputs = sess.run([output_name[0], output_name[1], output_name[2]],
                            {input_name[0]: left_torch.cpu().numpy(),
                            input_name[1]: right_torch.cpu().numpy()})
    print("Avg runtime [s]:", (time.perf_counter()-t1)/args.nb_runs)
    print(f"output shape: {outputs[1].shape}")

    pred_disp, pred_occ, pred_conf = outputs
    pred_disp = np.squeeze(pred_disp)
    pred_occ = np.squeeze(pred_occ)
    pred_conf = np.squeeze(pred_conf)

    # opencv 2D visualization
    valid = ((pred_conf >.1)*(pred_occ >.01))
    d_min = np.min(pred_disp)
    d_max = np.max(pred_disp)
    disp_left_vis = (pred_disp - d_min) / (d_max-d_min) * 255
    disp_left_vis = disp_left_vis.astype("uint8")
    disp_left_vis = cv2.applyColorMap(disp_left_vis, cv2.COLORMAP_JET)
    disp_left_vis_mask = valid[:,:,np.newaxis] * disp_left_vis

    cv2.imshow('left-right', np.hstack((left, right)))
    cv2.imshow(f'left_disparity: min:{round(d_min)}, max:{round(d_max)}', np.hstack((disp_left_vis,disp_left_vis_mask)))
    cv2.waitKey(0)



if __name__ == '__main__':

    parser = get_args_parser()
    args = parser.parse_args()
    print(args)
    main(args)
