import torch
from main import EmotionRecognizer

PATH = "2024-1-4 12.35.34 b64-e1-a39.3918 CEL-SGD.pt"

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = EmotionRecognizer(7).to(device)
model.load_state_dict(torch.load(PATH))
model.eval()

# # Load the state dictionary
# state_dict = torch.load()

# # Iterate over the state dictionary
# for param_tensor in state_dict:
#     print(param_tensor, "\t", state_dict[param_tensor].size())