"""
RF Cavity Control - Main Entry Point

This script provides a command-line interface for training and testing
the RF cavity control reinforcement learning system.
"""

import os
import sys
import argparse

# Add paths
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(project_root, 'src'))
sys.path.append(os.path.join(project_root, 'scripts'))
sys.path.append(os.path.join(project_root, 'configs'))


def main():
    parser = argparse.ArgumentParser(description='RF Cavity Control RL System')
    parser.add_argument('command', choices=['train', 'test', 'env-test'], 
                       help='Command to execute')
    parser.add_argument('--model-path', type=str, default='./best_model/best_model.zip',
                       help='Path to model file (for test command)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                       help='Device to use for training/testing')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        print("Starting training...")
        os.chdir(os.path.join(project_root, 'scripts'))
        import train_rf_cavity
        train_rf_cavity.main()
        
    elif args.command == 'test':
        print("Starting testing...")
        os.chdir(os.path.join(project_root, 'scripts'))
        import test_rf_cavity
        test_rf_cavity.main()
        
    elif args.command == 'env-test':
        print("Testing environment...")
        os.chdir(os.path.join(project_root, 'scripts'))
        import test_environment
        test_environment.test_environment()


if __name__ == "__main__":
    main()
