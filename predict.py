import os
import time

import torch
from torchvision import transforms
import numpy as np
from PIL import Image
from src import deeplabv3_resnet50
from my_dataset import CityScpates_palette


def colorize_mask(mask, palette):
    zero_pad = 256 * 3 - len(palette)
    for i in range(zero_pad):
        palette.append(0)
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main():
    aux = False  # inference time not need aux_classifier
    classes = 19
    weights_path = "./save_weights/model_29.pth"
    img_path = "./test.jpg"
    # palette_path = "./palette.json"
    assert os.path.exists(weights_path), f"weights {weights_path} not found."
    assert os.path.exists(img_path), f"image {img_path} not found."
    # assert os.path.exists(palette_path), f"palette {palette_path} not found."
    # with open(palette_path, "rb") as f:
    #     pallette_dict = json.load(f)
    #     pallette = []
    #     for v in pallette_dict.values():
    #         pallette += v
    palette = CityScpates_palette

    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    model = deeplabv3_resnet50(aux=aux, num_classes=classes+1)

    # delete weights about aux_classifier
    weights_dict = torch.load(weights_path, map_location='cpu')['model']
    for k in list(weights_dict.keys()):
        if "aux" in k:
            del weights_dict[k]

    # load weights
    model.load_state_dict(weights_dict)
    model.to(device)

    # load image
    original_img = Image.open(img_path)

    # from pil image to tensor and normalize
    data_transform = transforms.Compose([transforms.Resize(1024),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=(0.28689529, 0.32513294, 0.28389176),
                                                              std=(0.17613647, 0.18099176, 0.17772235))])
    img = data_transform(original_img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    model.eval()  # 进入验证模式
    with torch.no_grad():
        # init model
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model(init_img)

        t_start = time_synchronized()
        output = model(img.to(device))
        t_end = time_synchronized()
        print("inference time: {}".format(t_end - t_start))

        prediction = output['out'].argmax(1).squeeze(0)
        prediction = prediction.to("cpu").numpy().astype(np.uint8)
        colorized_mask = colorize_mask(prediction, palette)
        # mask = Image.fromarray(prediction)
        colorized_mask.save("test_result.png")
        # mask.save("test_result.png")


if __name__ == '__main__':
    main()
