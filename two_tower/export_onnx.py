"""Export model to ONNX."""
import torch
import argparse
from two_tower.model import TwoTowerModel


def export_candidate_tower(model_path: str, output_path: str, model_name: str, output_dim: int):
    """Export candidate tower to ONNX."""
    model = TwoTowerModel(
        candidate_model_name=model_name,
        output_dim=output_dim
    )
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    dummy_text = "dummy candidate text"
    
    class CandidateEncoderWrapper(torch.nn.Module):
        def __init__(self, tower):
            super().__init__()
            self.tower = tower
        
        def forward(self, text: str) -> torch.Tensor:
            return self.tower([text])[0]
    
    wrapper = CandidateEncoderWrapper(model.candidate_tower)
    
    torch.onnx.export(
        wrapper,
        dummy_text,
        output_path,
        input_names=['text'],
        output_names=['embedding'],
        dynamic_axes={
            'text': {0: 'batch'},
            'embedding': {0: 'batch'}
        },
        opset_version=14
    )
    
    print(f"Exported candidate tower to {output_path}")


def export_job_tower(model_path: str, output_path: str, model_name: str, output_dim: int):
    """Export job tower to ONNX."""
    model = TwoTowerModel(
        candidate_model_name=model_name,
        output_dim=output_dim
    )
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    dummy_text = "dummy job text"
    
    class JobEncoderWrapper(torch.nn.Module):
        def __init__(self, tower):
            super().__init__()
            self.tower = tower
        
        def forward(self, text: str) -> torch.Tensor:
            return self.tower([text])[0]
    
    wrapper = JobEncoderWrapper(model.job_tower)
    
    torch.onnx.export(
        wrapper,
        dummy_text,
        output_path,
        input_names=['text'],
        output_names=['embedding'],
        dynamic_axes={
            'text': {0: 'batch'},
            'embedding': {0: 'batch'}
        },
        opset_version=14
    )
    
    print(f"Exported job tower to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--tower', type=str, choices=['candidate', 'job'], required=True)
    parser.add_argument('--model_name', type=str, default='sentence-transformers/all-MiniLM-L6-v2')
    parser.add_argument('--output_dim', type=int, default=256)
    
    args = parser.parse_args()
    
    if args.tower == 'candidate':
        export_candidate_tower(
            args.model_path,
            args.output_path,
            args.model_name,
            args.output_dim
        )
    else:
        export_job_tower(
            args.model_path,
            args.output_path,
            args.model_name,
            args.output_dim
        )


if __name__ == '__main__':
    main()


