from PIL import Image
import torchvision.transforms as transforms
import torch


def process_image(image):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])

    # Применяем преобразования для нормализации
    normalized_image = preprocess(image).unsqueeze(0).cuda() * 2 - 1
    return normalized_image


def torch_tensor_to_pil(tensor):
    tensor = tensor.squeeze().cpu()
    # Перевод тензора в диапазон от 0 до 1
    tensor = (tensor + 1) / 2.0

    # Перевод тензора в формат (C, H, W)
    tensor = tensor.permute(1, 2, 0)
    # Визуализация изображения
    image = Image.fromarray((tensor * 255).type(torch.uint8).numpy())
    return image


def stylize(image, model_name):
    if model_name == "anime":
        model = torch.jit.load('selfie2anime.pt')
    elif model_name == "mone2photo":
        model = torch.jit.load('mone2photo.pt')
    elif model_name == "photo2mone":
        model = torch.jit.load('photo2mone.pt')
    else:
        raise NameError("Unexpected model name. Use 'anime' or 'mone'")
    with torch.no_grad():
        model.eval()
        out = model(process_image(image))
    return torch_tensor_to_pil(out)
