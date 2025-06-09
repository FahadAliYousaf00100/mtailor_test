from torch_model_class import Classifier, BasicBlock
import torch


def create_torch_model(load_state: str) -> Classifier:
    model = Classifier(BasicBlock, [2, 2, 2, 2])
    model.load_state_dict(torch.load(load_state))
    model.eval()
    return model


def convert_to_onnx(model: Classifier, input: torch.Tensor, output_path: str):
    torch.onnx.export(model, input, output_path, opset_version=11)
    print(f"Model has been converted to ONNX format and saved at {output_path}.")
