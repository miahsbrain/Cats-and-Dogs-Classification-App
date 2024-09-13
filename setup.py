import torch
from torch import nn
from pathlib import Path
from PIL import Image

class ModelV0(nn.Module):
    def __init__(self, input_features: int, hidden_units: int, output_features: int):
        super().__init__()
        self.block_one = nn.Sequential(
            nn.Conv2d(in_channels=input_features, out_channels=hidden_units, kernel_size=[4,4], stride=[1,1], padding=[1,1]),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=[2,2], stride=[2,2]),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=[4,4], stride=[1,1], padding=[1,1]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=[2,2], stride=[2,2]),
            nn.ReLU(),
        )
        self.block_two = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=[4,4], stride=[1,1], padding=[1,1]),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=[2,2], stride=[2,2]),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=[4,4], stride=[1,1], padding=[1,1]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=[2,2], stride=[2,2]),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*9*11, out_features=output_features)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.block_two(self.block_one(x)))

MODEL_SAVE_PATH = Path('models')
MODEL_NAME = 'ModelV0.pth'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

loaded_data = torch.load(MODEL_SAVE_PATH/MODEL_NAME, map_location=device)
input_features, hidden_units, output_features = loaded_data['model_attributes'].values()
model = ModelV0(input_features=input_features, hidden_units=hidden_units, output_features=output_features)
model.load_state_dict(loaded_data['model_weights'])
classes_to_idx = loaded_data['classes_to_idx']
transform = loaded_data['transform']

def make_predictions(model=model, transform=transform, image=None, device=device, class_to_idx=classes_to_idx):
    model.to(device)
    image = Image.open(image)
    trans_image = transform(image).unsqueeze(dim=0).to(device)
    logits = model(trans_image)
    preds = int(logits.sigmoid().round())
    class_name = [(k, v) for k, v in class_to_idx.items() if v == preds][0][0]
    return class_name

# predicted_class = make_predictions(image=Path('data/cat_dog/test/cat/cat.4.jpg'))
# print(predicted_class)