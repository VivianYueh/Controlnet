import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
from diffusers import AutoPipelineForImage2Image, ControlNetModel
from diffusers.utils import load_image, make_image_grid

controlnet=ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth", torch_dtype=torch.float16, variant="fp16", use_safetensors=True)

pipeline =AutoPipelineForImage2Image.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
).to("cuda")

pipeline.enable_model_cpu_offload()

import qrcode
from PIL import Image
def create_code(content: str):
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=16,
        #border=0,
    )
    qr.add_data(content)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    offset_min = 8 * 16
    w, h = img.size
    w = (w + 255 + offset_min) //255*255
    h = (h + 255 + offset_min) //255*255
    if w > 1024:
        raise Exception("QR code is too large, please use a shorter content")
    bg = Image.new('L', (w, h), 128)

    coords = ((w - img.size[0]) // 2 // 16 * 16,
              (h - img.size[1]) // 2 // 16 * 16)
    bg.paste(img, coords)

    return img

#image_url="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-init.png"

img_content = "https://hackmd.io/@ZuOHmaTLS_GQb9ZYidK_6w/rkouZJ_zR"    # 要轉換成 QRCode 的文字
qrcode_image = create_code(img_content)
qrcode_image.save('qrcode_image1.png')    # 儲存圖片
#init_image=load_image(image_url)
init_image=qrcode_image.resize((958,960)) # Note: Be sure the images have the same dimensions

depth_image=load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-init.png")
depth_image=depth_image.resize((958,960))

prompt="Astronaut in Melbourne city, warm color palette, colorful, 8K"
negative_prompt="bad, ugly"
image_control_net=pipeline(prompt, negative_prompt=negative_prompt, image=init_image, control_image=depth_image).images[0]
make_image_grid([init_image, depth_image, image_control_net], rows=1, cols=3)
image_control_net.save("images/image_control_net.png")