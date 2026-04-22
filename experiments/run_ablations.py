import subprocess
import os

def run_cmd(cmd):
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True)

def main():
    # 1. Full Model
    run_cmd("python main.py --dataset 3dident --epochs 50 --lambda_mi 1.0 --samples 5000")
    
    # 2. Ablation: No Minimal Intervention Loss
    run_cmd("python main.py --dataset 3dident --epochs 50 --lambda_mi 0.0 --samples 5000")
    
    # 3. Ablation: No DAG Constraint (Set lambda_graph=0 or dag_rho=0 inside code)
    # For now, I'll just run with different MI weights
    run_cmd("python main.py --dataset 3dident --epochs 50 --lambda_mi 5.0 --samples 5000")

if __name__ == "__main__":
    main()
