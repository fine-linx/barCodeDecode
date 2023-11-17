import torch
from PIL import Image
from torch import nn
from torchvision import transforms

from decode_network.DecodeNet import DecodeNet

if __name__ == '__main__':
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    model = DecodeNet()
    model.load_state_dict(torch.load('../../decode_network/tune/resnet50_v0.4p_adam_best.pt'))
    model.eval()
    image = Image.open('../temp.png')
    input_image = preprocess(image)
    input_image = input_image.unsqueeze(0)
    with torch.no_grad():
        output = model(input_image)
    output = output.view(-1, 13, 10)
    output = nn.functional.softmax(output, dim=-1)
    _, predicted = torch.max(output, 2)
    arr = predicted.squeeze().cpu().numpy()
    result = "".join(map(str, arr))
    print(result)
