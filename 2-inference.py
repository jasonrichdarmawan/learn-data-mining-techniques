# Inference
import sys
import pandas as pd
import torch
from src.model import Network
from src.preprocess import PreprocessTransforms

print("Usage: python 2-inference.py <input filepath> <model weights filepath> <preprocess params filepath>")

filepath = sys.argv[1] if len(sys.argv) > 1 else "./datasets/pp5i_test.gr.csv"
preprocess_filepath = sys.argv[2] if len(sys.argv) > 2 else "weights/preprocess_params.pt"
model_weights_filepath = sys.argv[3] if len(sys.argv) > 3 else "weights/model_weights_0_4203_5_4_2.pth"

X_test = pd.read_csv(filepath).set_index("SNO").rename_axis(None, axis=0).T
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
preprocess_params = torch.load(preprocess_filepath, weights_only=True)
preprocess_transforms = PreprocessTransforms(mask=preprocess_params['mask'],
                                             mu=preprocess_params['mu'],
                                             sigma=preprocess_params['sigma'])
X_preprocessed = preprocess_transforms(X_test_tensor)
model_weights = torch.load(model_weights_filepath, weights_only=True)
model = Network(input_dim=model_weights['input_dim'],
                output_dim=model_weights['output_dim'],
                num_experts=model_weights['num_experts'],
                top_k=model_weights['top_k'])
model.load_state_dict(model_weights['state_dict'])
model.eval()
weighted_outputs, _, _ = model(X_preprocessed)
pred = torch.softmax(weighted_outputs, dim=-1).sum(dim=0).softmax(dim=-1).argmax(dim=-1)
reverse_mapping = {0: 'MED', 1: 'RHB', 2: 'EPD', 3: 'MGL', 4: 'JPA'}
pred = [reverse_mapping[i.item()] for i in pred]
print("Prediction: \n %s" % pred)