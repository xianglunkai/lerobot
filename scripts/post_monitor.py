#!/usr/bin/env python3
"""
离线训练监控脚本
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

class TrainingMonitor:
    def __init__(self, wandb_offline_dir):
        self.wandb_dir = Path(wandb_offline_dir)
        self.df = None
        
    def load_data(self):
        """加载 WandB 离线数据"""
        event_files = list(self.wandb_dir.glob("**/wandb-events.jsonl"))
        if not event_files:
            raise FileNotFoundError(f"No wandb-events.jsonl found in {self.wandb_dir}")
        
        data = []
        for event_file in event_files:
            with open(event_file, 'r') as f:
                for line in f:
                    if line.strip():
                        try:
                            event = json.loads(line)
                            # 只提取训练指标
                            if event.get('_runtime') is not None and 'loss_critic' in event:
                                row = {
                                    'step': event.get('_step', 0),
                                    'timestamp': event.get('_timestamp', 0),
                                    'runtime': event.get('_runtime', 0),
                                }
                                # 添加所有训练指标
                                for key, value in event.items():
                                    if not key.startswith('_') and isinstance(value, (int, float)):
                                        row[key] = value
                                data.append(row)
                        except json.JSONDecodeError:
                            continue
        
        self.df = pd.DataFrame(data)
        if not self.df.empty:
            self.df = self.df.sort_values('step').reset_index(drop=True)
        return self.df
    
    def plot_comprehensive_curves(self, save_path=None):
        """绘制全面的训练曲线"""
        if self.df is None:
            self.load_data()
            
        if self.df.empty:
            print("No training data found")
            return
            
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # 1. 损失曲线
        if 'loss_critic' in self.df.columns:
            axes[0, 0].plot(self.df['step'], self.df['loss_critic'], label='Critic Loss', alpha=0.7)
        if 'loss_actor' in self.df.columns:
            axes[0, 0].plot(self.df['step'], self.df['loss_actor'], label='Actor Loss', alpha=0.7)
        if 'loss_temperature' in self.df.columns:
            axes[0, 0].plot(self.df['step'], self.df['loss_temperature'], label='Temp Loss', alpha=0.7)
        axes[0, 0].set_title('Training Losses')
        axes[0, 0].legend()
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 梯度范数
        if 'critic_grad_norm' in self.df.columns:
            axes[0, 1].plot(self.df['step'], self.df['critic_grad_norm'], label='Critic Grad Norm', alpha=0.7)
        if 'actor_grad_norm' in self.df.columns:
            axes[0, 1].plot(self.df['step'], self.df['actor_grad_norm'], label='Actor Grad Norm', alpha=0.7)
        axes[0, 1].set_title('Gradient Norms')
        axes[0, 1].legend()
        axes[0, 1].set_ylabel('Grad Norm')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 缓冲区大小
        if 'replay_buffer_size' in self.df.columns:
            axes[0, 2].plot(self.df['step'], self.df['replay_buffer_size'], color='green', alpha=0.7)
            axes[0, 2].set_title('Replay Buffer Size')
            axes[0, 2].set_ylabel('Size')
            axes[0, 2].grid(True, alpha=0.3)
        
        # 4. 温度参数
        if 'temperature' in self.df.columns:
            axes[1, 0].plot(self.df['step'], self.df['temperature'], color='purple', alpha=0.7)
            axes[1, 0].set_title('Temperature')
            axes[1, 0].set_ylabel('Temperature')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 5. 优化频率
        if 'Optimization frequency loop [Hz]' in self.df.columns:
            axes[1, 1].plot(self.df['step'], self.df['Optimization frequency loop [Hz]'], color='orange', alpha=0.7)
            axes[1, 1].set_title('Optimization Frequency')
            axes[1, 1].set_ylabel('Hz')
            axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Q值统计（如果有）
        q_value_cols = [col for col in self.df.columns if 'q_value' in col.lower()]
        if q_value_cols:
            for col in q_value_cols:
                axes[1, 2].plot(self.df['step'], self.df[col], label=col, alpha=0.7)
            axes[1, 2].set_title('Q Values')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training curves saved to {save_path}")
        
        plt.show()
    
    def print_training_summary(self):
        """打印训练摘要"""
        if self.df is None:
            self.load_data()
            
        if self.df.empty:
            print("No training data available")
            return
            
        print("=== Training Summary ===")
        print(f"Total optimization steps: {self.df['step'].max()}")
        print(f"Total training time: {self.df['runtime'].max()/3600:.2f} hours")
        
        if 'replay_buffer_size' in self.df.columns:
            print(f"Final replay buffer size: {self.df['replay_buffer_size'].iloc[-1]}")
        
        if 'loss_critic' in self.df.columns:
            final_critic_loss = self.df['loss_critic'].iloc[-1]
            min_critic_loss = self.df['loss_critic'].min()
            print(f"Final critic loss: {final_critic_loss:.4f} (min: {min_critic_loss:.4f})")

def main():
    parser = argparse.ArgumentParser(description='Offline training monitor')
    parser.add_argument('--wandb-dir', type=str, required=True, 
                       help='Path to wandb offline directory')
    parser.add_argument('--output', type=str, default='training_curves.png',
                       help='Output image path')
    
    args = parser.parse_args()
    
    monitor = TrainingMonitor(args.wandb_dir)
    monitor.load_data()
    monitor.print_training_summary()
    monitor.plot_comprehensive_curves(save_path=args.output)

if __name__ == "__main__":
    main()