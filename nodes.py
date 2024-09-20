import os, glob, sys
import logging

import torch
import torch.nn.functional as torchfn
from torchvision.transforms.functional import normalize
from torchvision.ops import masks_to_boxes

import numpy as np
import cv2
import math
from typing import List
from PIL import Image
from scipy import stats
from insightface.app.common import Face
from segment_anything import sam_model_registry

from reactor.modules.processing import StableDiffusionProcessingImg2Img
from reactor.modules.shared import state
# from comfy_extras.chainner_models import model_loading
import reactor.comfy_utils

import reactor.scripts.reactor_version
from reactor.r_chainner import model_loading
from reactor.scripts.reactor_faceswap import (
    FaceSwapScript,
    get_models,
    get_current_faces_model,
    analyze_faces,
    half_det_size,
    providers
)
from reactor.scripts.reactor_logger import logger
from reactor.reactor_utils import (
    batch_tensor_to_pil,
    batched_pil_to_tensor,
    tensor_to_pil,
    img2tensor,
    tensor2img,
    save_face_model,
    load_face_model,
    download,
    set_ort_session,
    prepare_cropped_face,
    normalize_cropped_face,
    add_folder_path_and_extensions,
    rgba2rgb_tensor
)
from reactor.reactor_log_patch import apply_patch
from reactor.r_facelib.utils.face_restoration_helper import FaceRestoreHelper
from reactor.r_basicsr.utils.registry import ARCH_REGISTRY
import reactor.scripts.r_archs.codeformer_arch
import reactor.scripts.r_masking.subcore as subcore
import reactor.scripts.r_masking.core as core
import reactor.scripts.r_masking.segs as masking_segs


REACTOR_MODELS_PATH = os.environ.get("REACTOR_MODELS_DIR")
FACE_MODELS_PATH = os.path.join(REACTOR_MODELS_PATH, "faces")

if not os.path.exists(REACTOR_MODELS_PATH):
    os.makedirs(REACTOR_MODELS_PATH)
    if not os.path.exists(FACE_MODELS_PATH):
        os.makedirs(FACE_MODELS_PATH)

dir_facerestore_models = os.environ.get("REACTOR_FACERESTORE_MODELS_PATH")
os.makedirs(dir_facerestore_models, exist_ok=True)

BLENDED_FACE_MODEL = None
FACE_SIZE: int = 512
FACE_HELPER = None

# if "ultralytics" not in folder_paths.folder_names_and_paths:
#     add_folder_path_and_extensions("ultralytics_bbox", [os.path.join(models_dir, "ultralytics", "bbox")], folder_paths.supported_pt_extensions)
#     add_folder_path_and_extensions("ultralytics_segm", [os.path.join(models_dir, "ultralytics", "segm")], folder_paths.supported_pt_extensions)
#     add_folder_path_and_extensions("ultralytics", [os.path.join(models_dir, "ultralytics")], folder_paths.supported_pt_extensions)
# if "sams" not in folder_paths.folder_names_and_paths:
#     add_folder_path_and_extensions("sams", [os.path.join(models_dir, "sams")], folder_paths.supported_pt_extensions)

def get_facemodels():
    models_path = os.path.join(FACE_MODELS_PATH, "*")
    models = glob.glob(models_path)
    models = [x for x in models if x.endswith(".safetensors")]
    return models

def get_restorers():
    models_path = os.path.join(models_dir, "facerestore_models/*")
    models = glob.glob(models_path)
    models = [x for x in models if (x.endswith(".pth") or x.endswith(".onnx"))]
    if len(models) == 0:
        fr_urls = [
            "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/facerestore_models/GFPGANv1.3.pth",
            "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/facerestore_models/GFPGANv1.4.pth",
            "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/facerestore_models/codeformer-v0.1.0.pth",
            "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/facerestore_models/GPEN-BFR-512.onnx",
            "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/facerestore_models/GPEN-BFR-1024.onnx",
            "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/facerestore_models/GPEN-BFR-2048.onnx",
        ]
        for model_url in fr_urls:
            model_name = os.path.basename(model_url)
            model_path = os.path.join(dir_facerestore_models, model_name)
            download(model_url, model_path, model_name)
        models = glob.glob(models_path)
        models = [x for x in models if (x.endswith(".pth") or x.endswith(".onnx"))]
    return models

def get_model_names(get_models):
    models = get_models()
    names = []
    for x in models:
        names.append(os.path.basename(x))
    names.sort(key=str.lower)
    names.insert(0, "none")
    return names

def model_names():
    models = get_models()
    return {os.path.basename(x): x for x in models}


class reactor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "enabled": ("BOOLEAN", {"default": True, "label_off": "OFF", "label_on": "ON"}),
                "input_image": ("IMAGE",),
                "swap_model": (list(model_names().keys()),),
                "facedetection": (["retinaface_resnet50", "retinaface_mobile0.25", "YOLOv5l", "YOLOv5n"],),
                "face_restore_model": (get_model_names(get_restorers),),
                "face_restore_visibility": ("FLOAT", {"default": 1, "min": 0.1, "max": 1, "step": 0.05}),
                "codeformer_weight": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1, "step": 0.05}),
                "detect_gender_input": (["no","female","male"], {"default": "no"}),
                "detect_gender_source": (["no","female","male"], {"default": "no"}),
                "input_faces_index": ("STRING", {"default": "0"}),
                "source_faces_index": ("STRING", {"default": "0"}),
                "console_log_level": ([0, 1, 2], {"default": 1}),
            },
            "optional": {
                "source_image": ("IMAGE",),
                "face_model": ("FACE_MODEL",),
                "face_boost": ("FACE_BOOST",),
            },
            "hidden": {"faces_order": "FACES_ORDER"},
        }

    RETURN_TYPES = ("IMAGE","FACE_MODEL")
    FUNCTION = "execute"
    CATEGORY = "🌌 ReActor"

    def __init__(self):
        # self.face_helper = None
        self.faces_order = ["large-small", "large-small"]
        # self.face_size = FACE_SIZE
        self.face_boost_enabled = False
        self.restore = True
        self.boost_model = None
        self.interpolation = "Bicubic"
        self.boost_model_visibility = 1
        self.boost_cf_weight = 0.5

    def restore_face(
            self,
            input_image,
            face_restore_model,
            face_restore_visibility,
            codeformer_weight,
            facedetection,
        ):

        result = input_image

        if face_restore_model != "none":

            global FACE_SIZE, FACE_HELPER

            self.face_helper = FACE_HELPER
            
            faceSize = 512
            if "1024" in face_restore_model.lower():
                faceSize = 1024
            elif "2048" in face_restore_model.lower():
                faceSize = 2048

            logger.status(f"Restoring with {face_restore_model} | Face Size is set to {faceSize}")

            model_path = os.path.join(dir_facerestore_models, face_restore_model)

            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = "cpu" 

            if "codeformer" in face_restore_model.lower():

                codeformer_net = ARCH_REGISTRY.get("CodeFormer")(
                    dim_embd=512,
                    codebook_size=1024,
                    n_head=8,
                    n_layers=9,
                    connect_list=["32", "64", "128", "256"],
                ).to(device)
                checkpoint = torch.load(model_path)["params_ema"]
                codeformer_net.load_state_dict(checkpoint)
                facerestore_model = codeformer_net.eval()

            elif ".onnx" in face_restore_model:

                ort_session = set_ort_session(model_path, providers=providers)
                ort_session_inputs = {}
                facerestore_model = ort_session

            else:

                sd = comfy.utils.load_torch_file(model_path, safe_load=True)
                facerestore_model = model_loading.load_state_dict(sd).eval()
                facerestore_model.to(device)
            
            if faceSize != FACE_SIZE or self.face_helper is None:
                self.face_helper = FaceRestoreHelper(1, face_size=faceSize, crop_ratio=(1, 1), det_model=facedetection, save_ext='png', use_parse=True, device=device)
                FACE_SIZE = faceSize
                FACE_HELPER = self.face_helper

            image_np = 255. * result.numpy()

            total_images = image_np.shape[0]

            out_images = []

            for i in range(total_images):

                if total_images > 1:
                    logger.status(f"Restoring {i+1}")

                cur_image_np = image_np[i,:, :, ::-1]

                original_resolution = cur_image_np.shape[0:2]

                if facerestore_model is None or self.face_helper is None:
                    return result

                self.face_helper.clean_all()
                self.face_helper.read_image(cur_image_np)
                self.face_helper.get_face_landmarks_5(only_center_face=False, resize=640, eye_dist_threshold=5)
                self.face_helper.align_warp_face()

                restored_face = None

                for idx, cropped_face in enumerate(self.face_helper.cropped_faces):

                    # if ".pth" in face_restore_model:
                    cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
                    normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
                    cropped_face_t = cropped_face_t.unsqueeze(0).to(device)

                    try:

                        with torch.no_grad():

                            if ".onnx" in face_restore_model: # ONNX models

                                for ort_session_input in ort_session.get_inputs():
                                    if ort_session_input.name == "input":
                                        cropped_face_prep = prepare_cropped_face(cropped_face)
                                        ort_session_inputs[ort_session_input.name] = cropped_face_prep
                                    if ort_session_input.name == "weight":
                                        weight = np.array([ 1 ], dtype = np.double)
                                        ort_session_inputs[ort_session_input.name] = weight

                                output = ort_session.run(None, ort_session_inputs)[0][0]
                                restored_face = normalize_cropped_face(output)

                            else: # PTH models

                                output = facerestore_model(cropped_face_t, w=codeformer_weight)[0] if "codeformer" in face_restore_model.lower() else facerestore_model(cropped_face_t)[0]
                                restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))

                        del output
                        torch.cuda.empty_cache()

                    except Exception as error:

                        print(f"\tFailed inference: {error}", file=sys.stderr)
                        restored_face = tensor2img(cropped_face_t, rgb2bgr=True, min_max=(-1, 1))

                    if face_restore_visibility < 1:
                        restored_face = cropped_face * (1 - face_restore_visibility) + restored_face * face_restore_visibility

                    restored_face = restored_face.astype("uint8")
                    self.face_helper.add_restored_face(restored_face)

                self.face_helper.get_inverse_affine(None)

                restored_img = self.face_helper.paste_faces_to_input_image()
                restored_img = restored_img[:, :, ::-1]

                if original_resolution != restored_img.shape[0:2]:
                    restored_img = cv2.resize(restored_img, (0, 0), fx=original_resolution[1]/restored_img.shape[1], fy=original_resolution[0]/restored_img.shape[0], interpolation=cv2.INTER_AREA)

                self.face_helper.clean_all()

                # out_images[i] = restored_img
                out_images.append(restored_img)

            restored_img_np = np.array(out_images).astype(np.float32) / 255.0
            restored_img_tensor = torch.from_numpy(restored_img_np)

            result = restored_img_tensor

        return result
    
    def execute(self, enabled, input_image, swap_model, detect_gender_source, detect_gender_input, source_faces_index, input_faces_index, console_log_level, face_restore_model,face_restore_visibility, codeformer_weight, facedetection, source_image=None, face_model=None, faces_order=None, face_boost=None):
        if face_boost is not None:
            self.face_boost_enabled = face_boost["enabled"]
            self.boost_model = face_boost["boost_model"]
            self.interpolation = face_boost["interpolation"]
            self.boost_model_visibility = face_boost["visibility"]
            self.boost_cf_weight = face_boost["codeformer_weight"]
            self.restore = face_boost["restore_with_main_after"]
        else:
            self.face_boost_enabled = False

        if faces_order is None:
            faces_order = self.faces_order

        apply_patch(console_log_level)

        if not enabled:
            return (input_image,face_model)
        elif source_image is None and face_model is None:
            logger.error("Please provide 'source_image' or `face_model`")
            return (input_image,face_model)

        if face_model == "none":
            face_model = None
        
        script = FaceSwapScript()
        pil_images = batch_tensor_to_pil(input_image)
        if source_image is not None:
            source = tensor_to_pil(source_image)
        else:
            source = None
        p = StableDiffusionProcessingImg2Img(pil_images)
        script.process(
            p=p,
            img=source,
            enable=True,
            source_faces_index=source_faces_index,
            faces_index=input_faces_index,
            model=swap_model,
            swap_in_source=True,
            swap_in_generated=True,
            gender_source=detect_gender_source,
            gender_target=detect_gender_input,
            face_model=face_model,
            faces_order=faces_order,
            # face boost:
            face_boost_enabled=self.face_boost_enabled,
            face_restore_model=self.boost_model,
            face_restore_visibility=self.boost_model_visibility,
            codeformer_weight=self.boost_cf_weight,
            interpolation=self.interpolation,
        )
        result = batched_pil_to_tensor(p.init_images)

        if face_model is None:
            current_face_model = get_current_faces_model()
            face_model_to_provide = current_face_model[0] if (current_face_model is not None and len(current_face_model) > 0) else face_model
        else:
            face_model_to_provide = face_model

        if self.restore or not self.face_boost_enabled:
            result = reactor.restore_face(self,result,face_restore_model,face_restore_visibility,codeformer_weight,facedetection)

        return (result,face_model_to_provide)


class ReActorPlusOpt:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "enabled": ("BOOLEAN", {"default": True, "label_off": "OFF", "label_on": "ON"}),
                "input_image": ("IMAGE",),               
                "swap_model": (list(model_names().keys()),),
                "facedetection": (["retinaface_resnet50", "retinaface_mobile0.25", "YOLOv5l", "YOLOv5n"],),
                "face_restore_model": (get_model_names(get_restorers),),
                "face_restore_visibility": ("FLOAT", {"default": 1, "min": 0.1, "max": 1, "step": 0.05}),
                "codeformer_weight": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1, "step": 0.05}),
            },
            "optional": {
                "source_image": ("IMAGE",),
                "face_model": ("FACE_MODEL",),
                "options": ("OPTIONS",),
                "face_boost": ("FACE_BOOST",),
            }
        }

    RETURN_TYPES = ("IMAGE","FACE_MODEL")
    FUNCTION = "execute"
    CATEGORY = "🌌 ReActor"

    def __init__(self):
        # self.face_helper = None
        self.faces_order = ["large-small", "large-small"]
        self.detect_gender_input = "no"
        self.detect_gender_source = "no"
        self.input_faces_index = "0"
        self.source_faces_index = "0"
        self.console_log_level = 1
        # self.face_size = 512
        self.face_boost_enabled = False
        self.restore = True
        self.boost_model = None
        self.interpolation = "Bicubic"
        self.boost_model_visibility = 1
        self.boost_cf_weight = 0.5
    
    def execute(self, enabled, input_image, swap_model, facedetection, face_restore_model, face_restore_visibility, codeformer_weight, source_image=None, face_model=None, options=None, face_boost=None):

        if options is not None:
            self.faces_order = [options["input_faces_order"], options["source_faces_order"]]
            self.console_log_level = options["console_log_level"]
            self.detect_gender_input = options["detect_gender_input"]
            self.detect_gender_source = options["detect_gender_source"]
            self.input_faces_index = options["input_faces_index"]
            self.source_faces_index = options["source_faces_index"]
        
        if face_boost is not None:
            self.face_boost_enabled = face_boost["enabled"]
            self.restore = face_boost["restore_with_main_after"]
        else:
            self.face_boost_enabled = False
        
        result = reactor.execute(
            self,enabled,input_image,swap_model,self.detect_gender_source,self.detect_gender_input,self.source_faces_index,self.input_faces_index,self.console_log_level,face_restore_model,face_restore_visibility,codeformer_weight,facedetection,source_image,face_model,self.faces_order, face_boost=face_boost
        )

        return result


class LoadFaceModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "face_model": (get_model_names(get_facemodels),),
            }
        }
    
    RETURN_TYPES = ("FACE_MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "🌌 ReActor"

    def load_model(self, face_model):
        self.face_model = face_model
        self.face_models_path = FACE_MODELS_PATH
        if self.face_model != "none":
            face_model_path = os.path.join(self.face_models_path, self.face_model)
            out = load_face_model(face_model_path)
        else:
            out = None
        return (out, )


class BuildFaceModel:
    def __init__(self):
        self.output_dir = FACE_MODELS_PATH
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "save_mode": ("BOOLEAN", {"default": True, "label_off": "OFF", "label_on": "ON"}),
                "send_only": ("BOOLEAN", {"default": False, "label_off": "NO", "label_on": "YES"}),
                "face_model_name": ("STRING", {"default": "default"}),
                "compute_method": (["Mean", "Median", "Mode"], {"default": "Mean"}),
            },
            "optional": {
                "images": ("IMAGE",),
                "face_models": ("FACE_MODEL",),
            }
        }

    RETURN_TYPES = ("FACE_MODEL",)
    FUNCTION = "blend_faces"

    OUTPUT_NODE = True

    CATEGORY = "🌌 ReActor"

    def build_face_model(self, image: Image.Image, det_size=(640, 640)):
        logging.StreamHandler.terminator = "\n"
        if image is None:
            error_msg = "Please load an Image"
            logger.error(error_msg)
            return error_msg
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        face_model = analyze_faces(image, det_size)

        if len(face_model) == 0:
            print("")
            det_size_half = half_det_size(det_size)
            face_model = analyze_faces(image, det_size_half)
            if face_model is not None and len(face_model) > 0:
                print("...........................................................", end=" ")
        
        if face_model is not None and len(face_model) > 0:
            return face_model[0]
        else:
            no_face_msg = "No face found, please try another image"
            # logger.error(no_face_msg)
            return no_face_msg
    
    def blend_faces(self, save_mode, send_only, face_model_name, compute_method, images=None, face_models=None):
        global BLENDED_FACE_MODEL
        blended_face: Face = BLENDED_FACE_MODEL

        if send_only and blended_face is None:
            send_only = False

        if (images is not None or face_models is not None) and not send_only:

            faces = []
            embeddings = []

            apply_patch(1)

            if images is not None:
                images_list: List[Image.Image] = batch_tensor_to_pil(images)

                n = len(images_list)

                for i,image in enumerate(images_list):
                    logging.StreamHandler.terminator = " "
                    logger.status(f"Building Face Model {i+1} of {n}...")
                    face = self.build_face_model(image)
                    if isinstance(face, str):
                        logger.error(f"No faces found in image {i+1}, skipping")
                        continue
                    else:
                        print(f"{int(((i+1)/n)*100)}%")
                    faces.append(face)
                    embeddings.append(face.embedding)
            
            elif face_models is not None:

                n = len(face_models)

                for i,face_model in enumerate(face_models):
                    logging.StreamHandler.terminator = " "
                    logger.status(f"Extracting Face Model {i+1} of {n}...")
                    face = face_model
                    if isinstance(face, str):
                        logger.error(f"No faces found for face_model {i+1}, skipping")
                        continue
                    else:
                        print(f"{int(((i+1)/n)*100)}%")
                    faces.append(face)
                    embeddings.append(face.embedding)

            logging.StreamHandler.terminator = "\n"
            if len(faces) > 0:
                # compute_method_name = "Mean" if compute_method == 0 else "Median" if compute_method == 1 else "Mode"
                logger.status(f"Blending with Compute Method '{compute_method}'...")
                blended_embedding = np.mean(embeddings, axis=0) if compute_method == "Mean" else np.median(embeddings, axis=0) if compute_method == "Median" else stats.mode(embeddings, axis=0)[0].astype(np.float32)
                blended_face = Face(
                    bbox=faces[0].bbox,
                    kps=faces[0].kps,
                    det_score=faces[0].det_score,
                    landmark_3d_68=faces[0].landmark_3d_68,
                    pose=faces[0].pose,
                    landmark_2d_106=faces[0].landmark_2d_106,
                    embedding=blended_embedding,
                    gender=faces[0].gender,
                    age=faces[0].age
                )
                if blended_face is not None:
                    BLENDED_FACE_MODEL = blended_face
                    if save_mode:
                        face_model_path = os.path.join(FACE_MODELS_PATH, face_model_name + ".safetensors")
                        save_face_model(blended_face,face_model_path)
                        # done_msg = f"Face model has been saved to '{face_model_path}'"
                        # logger.status(done_msg)
                    logger.status("--Done!--")
                    # return (blended_face,)
                else:
                    no_face_msg = "Something went wrong, please try another set of images"
                    logger.error(no_face_msg)
                    # return (blended_face,)
            # logger.status("--Done!--")
        if images is None and face_models is None:
            logger.error("Please provide `images` or `face_models`")
        return (blended_face,)


class SaveFaceModel:
    def __init__(self):
        self.output_dir = FACE_MODELS_PATH

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "save_mode": ("BOOLEAN", {"default": True, "label_off": "OFF", "label_on": "ON"}),
                "face_model_name": ("STRING", {"default": "default"}),
                "select_face_index": ("INT", {"default": 0, "min": 0}),
            },
            "optional": {
                "image": ("IMAGE",),
                "face_model": ("FACE_MODEL",),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_model"

    OUTPUT_NODE = True

    CATEGORY = "🌌 ReActor"

    def save_model(self, save_mode, face_model_name, select_face_index, image=None, face_model=None, det_size=(640, 640)):
        if save_mode and image is not None:
            source = tensor_to_pil(image)
            source = cv2.cvtColor(np.array(source), cv2.COLOR_RGB2BGR)
            apply_patch(1)
            logger.status("Building Face Model...")
            face_model_raw = analyze_faces(source, det_size)
            if len(face_model_raw) == 0:
                det_size_half = half_det_size(det_size)
                face_model_raw = analyze_faces(source, det_size_half)
            try:
                face_model = face_model_raw[select_face_index]
            except:
                logger.error("No face(s) found")
                return face_model_name
            logger.status("--Done!--")
        if save_mode and (face_model != "none" or face_model is not None):
            face_model_path = os.path.join(self.output_dir, face_model_name + ".safetensors")
            save_face_model(face_model,face_model_path)
        if image is None and face_model is None:
            logger.error("Please provide `face_model` or `image`")
        return face_model_name


class RestoreFace:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),               
                "facedetection": (["retinaface_resnet50", "retinaface_mobile0.25", "YOLOv5l", "YOLOv5n"],),
                "model": (get_model_names(get_restorers),),
                "visibility": ("FLOAT", {"default": 1, "min": 0.0, "max": 1, "step": 0.05}),
                "codeformer_weight": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "🌌 ReActor"

    # def __init__(self):
    #     self.face_helper = None
    #     self.face_size = 512

    def execute(self, image, model, visibility, codeformer_weight, facedetection):
        result = reactor.restore_face(self,image,model,visibility,codeformer_weight,facedetection)
        return (result,)

class ImageDublicator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),               
                "count": ("INT", {"default": 1, "min": 0}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGES",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "execute"
    CATEGORY = "🌌 ReActor"

    def execute(self, image, count):
        images = [image for i in range(count)]        
        return (images,)


class ImageRGBA2RGB:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "🌌 ReActor"

    def execute(self, image):
        out = rgba2rgb_tensor(image)       
        return (out,)


class MakeFaceModelBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "face_model1": ("FACE_MODEL",), 
            },
            "optional": {
                "face_model2": ("FACE_MODEL",),
                "face_model3": ("FACE_MODEL",),
                "face_model4": ("FACE_MODEL",),
                "face_model5": ("FACE_MODEL",),
                "face_model6": ("FACE_MODEL",),
                "face_model7": ("FACE_MODEL",),
                "face_model8": ("FACE_MODEL",),
                "face_model9": ("FACE_MODEL",),
                "face_model10": ("FACE_MODEL",),
            },
        }

    RETURN_TYPES = ("FACE_MODEL",)
    RETURN_NAMES = ("FACE_MODELS",)
    FUNCTION = "execute"

    CATEGORY = "🌌 ReActor"

    def execute(self, **kwargs):
        if len(kwargs) > 0:
            face_models = [value for value in kwargs.values()]
            return (face_models,)
        else:
            logger.error("Please provide at least 1 `face_model`")
            return (None,)


class ReActorOptions:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_faces_order": (
                    ["left-right","right-left","top-bottom","bottom-top","small-large","large-small"], {"default": "large-small"}
                ),
                "input_faces_index": ("STRING", {"default": "0"}),
                "detect_gender_input": (["no","female","male"], {"default": "no"}),
                "source_faces_order": (
                    ["left-right","right-left","top-bottom","bottom-top","small-large","large-small"], {"default": "large-small"}
                ),
                "source_faces_index": ("STRING", {"default": "0"}),
                "detect_gender_source": (["no","female","male"], {"default": "no"}),
                "console_log_level": ([0, 1, 2], {"default": 1}),
            }
        }

    RETURN_TYPES = ("OPTIONS",)
    FUNCTION = "execute"
    CATEGORY = "🌌 ReActor"

    def execute(self,input_faces_order, input_faces_index, detect_gender_input, source_faces_order, source_faces_index, detect_gender_source, console_log_level):
        options: dict = {
            "input_faces_order": input_faces_order,
            "input_faces_index": input_faces_index,
            "detect_gender_input": detect_gender_input,
            "source_faces_order": source_faces_order,
            "source_faces_index": source_faces_index,
            "detect_gender_source": detect_gender_source,
            "console_log_level": console_log_level,
        }
        return (options, )


class ReActorFaceBoost:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "enabled": ("BOOLEAN", {"default": True, "label_off": "OFF", "label_on": "ON"}),
                "boost_model": (get_model_names(get_restorers),),
                "interpolation": (["Nearest","Bilinear","Bicubic","Lanczos"], {"default": "Bicubic"}),
                "visibility": ("FLOAT", {"default": 1, "min": 0.1, "max": 1, "step": 0.05}),
                "codeformer_weight": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1, "step": 0.05}),
                "restore_with_main_after": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("FACE_BOOST",)
    FUNCTION = "execute"
    CATEGORY = "🌌 ReActor"

    def execute(self,enabled,boost_model,interpolation,visibility,codeformer_weight,restore_with_main_after):
        face_boost: dict = {
            "enabled": enabled,
            "boost_model": boost_model,
            "interpolation": interpolation,
            "visibility": visibility,
            "codeformer_weight": codeformer_weight,
            "restore_with_main_after": restore_with_main_after,
        }
        return (face_boost, )
    

NODE_CLASS_MAPPINGS = {
    # --- MAIN NODES ---
    "ReActorFaceSwap": reactor,
    "ReActorFaceSwapOpt": ReActorPlusOpt,
    "ReActorOptions": ReActorOptions,
    "ReActorFaceBoost": ReActorFaceBoost,
    # --- Operations with Face Models ---
    "ReActorSaveFaceModel": SaveFaceModel,
    "ReActorLoadFaceModel": LoadFaceModel,
    "ReActorBuildFaceModel": BuildFaceModel,
    "ReActorMakeFaceModelBatch": MakeFaceModelBatch,
    # --- Additional Nodes ---
    "ReActorRestoreFace": RestoreFace,
    "ReActorImageDublicator": ImageDublicator,
    "ImageRGBA2RGB": ImageRGBA2RGB,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # --- MAIN NODES ---
    "ReActorFaceSwap": "ReActor 🌌 Fast Face Swap",
    "ReActorFaceSwapOpt": "ReActor 🌌 Fast Face Swap [OPTIONS]",
    "ReActorOptions": "ReActor 🌌 Options",
    "ReActorFaceBoost": "ReActor 🌌 Face Booster",
    "ReActorMaskHelper": "ReActor 🌌 Masking Helper",
    # --- Operations with Face Models ---
    "ReActorSaveFaceModel": "Save Face Model 🌌 ReActor",
    "ReActorLoadFaceModel": "Load Face Model 🌌 ReActor",
    "ReActorBuildFaceModel": "Build Blended Face Model 🌌 ReActor",
    "ReActorMakeFaceModelBatch": "Make Face Model Batch 🌌 ReActor",
    # --- Additional Nodes ---
    "ReActorRestoreFace": "Restore Face 🌌 ReActor",
    "ReActorImageDublicator": "Image Dublicator (List) 🌌 ReActor",
    "ImageRGBA2RGB": "Convert RGBA to RGB 🌌 ReActor",
}
