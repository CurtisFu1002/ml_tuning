import xgboost as xgb
import pandas as pd
import numpy as np
import time
import argparse
import re
from pathlib import Path

class TensileOptimizerPerProblem:
    def __init__(self, model_path, yaml_path, output_dir, top_k=20):
        self.model_path = model_path
        self.yaml_path = yaml_path
        self.output_dir = Path(output_dir)
        self.top_k = top_k
        self.model = None
        self.yaml_template = None
        self.mi_configs = []
        
    def load_model(self):
        """Load XGBoost model"""
        print(f"Loading model: {self.model_path}")
        self.model = xgb.Booster()
        self.model.load_model(self.model_path)
        print("✓ Model loaded successfully")
        
    def load_yaml_template(self):
        """Load original YAML as template"""
        print(f"\nLoading YAML template: {self.yaml_path}")
        with open(self.yaml_path, 'r') as f:
            self.yaml_template = f.read()
        print("✓ YAML template loaded successfully")
        
    def extract_mi_configs(self):
        """Extract all MatrixInstruction configurations from YAML"""
        mi_pattern = r'- \[([\d\s,]+)\]\s*#(\d+)'
        matches = re.findall(mi_pattern, self.yaml_template)
        
        mi_list = []
        for config_str, idx in matches:
            config = [int(x.strip()) for x in config_str.split(',')]
            mi_list.append(config)
        
        self.mi_configs = mi_list
        print(f"✓ Extracted {len(mi_list)} MI configurations")
        return mi_list
    
    def extract_problem_sizes(self):
        """Extract all Problem Sizes from YAML"""
        ps_pattern = r'- Exact: \[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]'
        matches = re.findall(ps_pattern, self.yaml_template)
        
        problem_sizes = []
        for m, n, b, k in matches:
            problem_sizes.append({
                'm': int(m), 'n': int(n), 'b': int(b), 'k': int(k)
            })
        
        print(f"✓ Extracted {len(problem_sizes)} Problem Sizes")
        return problem_sizes
    
    def mi_to_features(self, m, n, k, mi_config):
        """Convert Problem Size and MI config to model features"""
        features = [
            m, n, k,
            mi_config[0], mi_config[1], mi_config[2],
            mi_config[3], mi_config[4], mi_config[5],
            mi_config[6], mi_config[7],
            mi_config[8] if len(mi_config) > 8 else 1,
        ]
        return features
    
    def predict_for_problem(self, problem_size, mi_configs):
        """Predict GFLOPS for all MI configurations for a single Problem Size"""
        m, n, k = problem_size['m'], problem_size['n'], problem_size['k']
        
        X = []
        for mi in mi_configs:
            features = self.mi_to_features(m, n, k, mi)
            X.append(features)
        
        X = np.array(X, dtype=np.float32)
        dmatrix = xgb.DMatrix(X)
        predictions = self.model.predict(dmatrix)
        
        results = []
        for idx, (mi, pred_gflops) in enumerate(zip(mi_configs, predictions)):
            results.append({
                'mi_idx': idx,
                'mi_config': mi,
                'predicted_gflops': pred_gflops
            })
        
        results = sorted(results, key=lambda x: x['predicted_gflops'], reverse=True)
        return results
    
    def format_mi_line(self, mi_config, idx, is_last=False):
        """Format a single MI configuration line"""
        # Format: - [16, 16,16, 1,  1,   1, 1,  4,1 ] #0
        formatted = f"          - [{mi_config[0]:2d}, {mi_config[1]:2d},{mi_config[2]:2d}, "
        formatted += f"{mi_config[3]:2d}, {mi_config[4]:2d}, {mi_config[5]:3d}, "
        formatted += f"{mi_config[6]:2d}, {mi_config[7]:2d},{mi_config[8]:2d} ] #{idx}"
        return formatted
    
    def create_optimized_yaml(self, problem_size, top_k_predictions):
        """Create optimized YAML by replacing MI configs and problem size"""
        yaml_content = self.yaml_template
        
        # 1. Replace MatrixInstruction section
        mi_lines = []
        for i, pred in enumerate(top_k_predictions):
            mi_line = self.format_mi_line(
                pred['mi_config'], 
                pred['mi_idx'],
                is_last=(i == len(top_k_predictions) - 1)
            )
            mi_lines.append(mi_line)
        
        new_mi_section = '\n'.join(mi_lines)
        
        # Find and replace MatrixInstruction section
        mi_pattern = r'(        - MatrixInstruction:\n)(.*?)(        # - PrefetchGlobalRead:)'
        
        def replace_mi(match):
            return match.group(1) + new_mi_section + '\n' + match.group(3)
        
        yaml_content = re.sub(mi_pattern, replace_mi, yaml_content, flags=re.DOTALL)
        
        # 2. Replace ProblemSizes section (keep only current problem)
        m, n, b, k = problem_size['m'], problem_size['n'], problem_size['b'], problem_size['k']
        new_ps_section = f"          - Exact: [{m}, {n}, {b}, {k}]"
        
        ps_pattern = r'(        - ProblemSizes:\n)(.*?)(        - BiasTypeArgs:)'
        
        def replace_ps(match):
            return match.group(1) + new_ps_section + '\n          \n' + match.group(3)
        
        yaml_content = re.sub(ps_pattern, replace_ps, yaml_content, flags=re.DOTALL)
        
        return yaml_content
    
    def run_optimization(self):
        """Execute complete optimization workflow"""
        total_start = time.time()
        
        print("=" * 80)
        print("Tensile MI Configuration Optimizer (Per-Problem)")
        print("=" * 80)
        
        self.load_model()
        self.load_yaml_template()
        
        mi_configs = self.extract_mi_configs()
        problem_sizes = self.extract_problem_sizes()
        
        if not mi_configs or not problem_sizes:
            print("Error: Unable to extract MI configs or Problem Sizes")
            return
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        all_results = []
        
        print(f"\nStarting prediction for {len(problem_sizes)} Problem Sizes...")
        print("-" * 80)
        
        for idx, ps in enumerate(problem_sizes):
            m, n, k, b = ps['m'], ps['n'], ps['k'], ps['b']
            print(f"\n[{idx+1}/{len(problem_sizes)}] Problem: M={m}, N={n}, B={b}, K={k}")
            
            # Predict
            start_time = time.time()
            predictions = self.predict_for_problem(ps, mi_configs)
            pred_time = time.time() - start_time
            
            # Select Top-K
            top_k_predictions = predictions[:self.top_k]
            
            print(f"  Prediction time: {pred_time:.4f}s")
            print(f"  Top-{self.top_k} predicted GFLOPS: "
                  f"{top_k_predictions[0]['predicted_gflops']:.2f} (best) ~ "
                  f"{top_k_predictions[-1]['predicted_gflops']:.2f} (worst)")
            print(f"  Top-5 MI indices: {[p['mi_idx'] for p in top_k_predictions[:5]]}")
            
            # Create optimized YAML
            optimized_yaml = self.create_optimized_yaml(ps, top_k_predictions)
            
            # Save
            yaml_filename = f"optimal_m{m}_n{n}_b{b}_k{k}_top{self.top_k}.yaml"
            yaml_path = self.output_dir / yaml_filename
            
            with open(yaml_path, 'w') as f:
                f.write(optimized_yaml)
            
            print(f"  Saved: {yaml_filename}")
            
            # Record results
            all_results.append({
                'problem_idx': idx,
                'm': m, 'n': n, 'b': b, 'k': k,
                'original_mi_count': len(mi_configs),
                'optimized_mi_count': len(top_k_predictions),
                'reduction_ratio': (1 - len(top_k_predictions)/len(mi_configs)) * 100,
                'top1_gflops': top_k_predictions[0]['predicted_gflops'],
                'topk_gflops': top_k_predictions[-1]['predicted_gflops'],
                'top5_mi_indices': [p['mi_idx'] for p in top_k_predictions[:5]],
                'prediction_time': pred_time,
                'output_file': yaml_filename
            })
        
        total_time = time.time() - total_start
        
        self.generate_summary_report(all_results, total_time)
        
        print("\n" + "=" * 80)
        print(f" Optimization completed! Total time: {total_time:.4f}s")
        print(f" Output directory: {self.output_dir}")
        print(f" Each problem reduced from {len(mi_configs)} to {self.top_k} MI configs")
        print(f" Configuration reduction rate: {(1 - self.top_k/len(mi_configs))*100:.1f}%")
        print("=" * 80)
        
        return all_results
    
    def generate_summary_report(self, all_results, total_time):
        """Generate summary report"""
        report_path = self.output_dir / f"optimization_summary_top{self.top_k}.txt"
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("Tensile MI Configuration Optimization Summary\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Model Path: {self.model_path}\n")
            f.write(f"Original YAML: {self.yaml_path}\n")
            f.write(f"Output Directory: {self.output_dir}\n")
            f.write(f"Top-K: {self.top_k}\n\n")
            
            f.write(f"Total Problem Sizes: {len(all_results)}\n")
            f.write(f"Total Execution Time: {total_time:.4f}s\n")
            f.write(f"Average Time per Problem: {total_time/len(all_results):.4f}s\n\n")
            
            avg_reduction = np.mean([r['reduction_ratio'] for r in all_results])
            f.write(f"Average MI Configuration Reduction: {avg_reduction:.2f}%\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("Per Problem Size Details:\n")
            f.write("-" * 80 + "\n\n")
            
            for r in all_results:
                f.write(f"Problem #{r['problem_idx']}: "
                       f"M={r['m']}, N={r['n']}, B={r['b']}, K={r['k']}\n")
                f.write(f"  MI configs: {r['original_mi_count']} → {r['optimized_mi_count']} "
                       f"(reduced {r['reduction_ratio']:.1f}%)\n")
                f.write(f"  Predicted GFLOPS: {r['top1_gflops']:.2f} (top-1) ~ "
                       f"{r['topk_gflops']:.2f} (top-{self.top_k})\n")
                f.write(f"  Top-5 MI indices: {r['top5_mi_indices']}\n")
                f.write(f"  Prediction time: {r['prediction_time']:.4f}s\n")
                f.write(f"  Output file: {r['output_file']}\n\n")
        
        print(f"\n✓ Summary report saved: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="XGBoost-based Tensile MI Optimizer")
    parser.add_argument("--model", type=str, 
                       default="../model/G3_1500round.xgb",
                       help="XGBoost model path")
    parser.add_argument("--yaml", type=str,
                       default="./speedup_test_logic.yaml",
                       help="Input YAML configuration")
    parser.add_argument("--output-dir", type=str,
                       default="./optimal_configs",
                       help="Output directory")
    parser.add_argument("--top-k", type=int, default=20,
                       help="Number of top MI configs to keep")
    
    args = parser.parse_args()
    
    if not Path(args.model).exists():
        print(f"Error: Model file not found: {args.model}")
        return
    
    if not Path(args.yaml).exists():
        print(f"Error: YAML file not found: {args.yaml}")
        return
    
    optimizer = TensileOptimizerPerProblem(
        model_path=args.model,
        yaml_path=args.yaml,
        output_dir=args.output_dir,
        top_k=args.top_k
    )
    
    optimizer.run_optimization()


if __name__ == "__main__":
    main()