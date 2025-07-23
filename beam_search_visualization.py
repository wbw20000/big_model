"""
束搜索过程可视化示例
展示候选管理机制的实际工作流程
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np
from typing import List, Tuple, Dict


class BeamSearchVisualizer:
    """束搜索过程可视化器"""
    
    def __init__(self, vocab: List[str], beam_size: int = 3):
        self.vocab = vocab
        self.beam_size = beam_size
        self.steps = []
        
    def add_step(self, beams: List[Dict]):
        """添加一个搜索步骤"""
        self.steps.append(beams)
        
    def visualize(self, filename: str = "beam_search_process.png"):
        """生成可视化图"""
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # 设置坐标轴
        ax.set_xlim(-0.5, len(self.steps) + 0.5)
        ax.set_ylim(-0.5, self.beam_size + 0.5)
        ax.set_xlabel('Time Step', fontsize=12)
        ax.set_ylabel('Beam Index', fontsize=12)
        ax.set_title('Beam Search Candidate Management Process', fontsize=16)
        
        # 颜色映射
        colors = plt.cm.Set3(np.linspace(0, 1, self.beam_size))
        
        # 绘制每个时间步的束
        for step_idx, beams in enumerate(self.steps):
            for beam_idx, beam in enumerate(beams[:self.beam_size]):
                # 绘制节点
                x, y = step_idx, beam_idx
                
                # 创建文本框
                text = f"{' '.join(beam['tokens'])}\n{beam['score']:.3f}"
                
                # 根据是否完成选择样式
                if beam.get('finished', False):
                    box_style = "round,pad=0.3"
                    edge_color = 'red'
                    line_width = 2
                else:
                    box_style = "round,pad=0.3"
                    edge_color = 'black'
                    line_width = 1
                
                # 添加文本框
                bbox = dict(boxstyle=box_style, 
                           facecolor=colors[beam_idx], 
                           edgecolor=edge_color,
                           linewidth=line_width,
                           alpha=0.8)
                
                ax.text(x, y, text, 
                       ha='center', va='center',
                       bbox=bbox,
                       fontsize=10,
                       weight='bold' if beam.get('best', False) else 'normal')
                
                # 绘制连接线
                if step_idx > 0 and 'parent_idx' in beam:
                    parent_idx = beam['parent_idx']
                    ax.plot([step_idx-1, step_idx], 
                           [parent_idx, beam_idx],
                           'k-', alpha=0.3, linewidth=1)
        
        # 添加图例
        legend_elements = [
            patches.Rectangle((0, 0), 1, 1, facecolor='white', 
                            edgecolor='red', linewidth=2, 
                            label='Finished beam'),
            patches.Rectangle((0, 0), 1, 1, facecolor='white', 
                            edgecolor='black', linewidth=1, 
                            label='Active beam')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        # 网格
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(len(self.steps)))
        ax.set_yticks(range(self.beam_size))
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"可视化图已保存到: {filename}")


def simulate_beam_search_process():
    """模拟一个简单的束搜索过程"""
    # 简化的词表
    vocab = ['我', '喜欢', '编程', '学习', '人工智能', '很', '有趣', '<EOS>']
    
    # 创建可视化器
    visualizer = BeamSearchVisualizer(vocab, beam_size=3)
    
    # 步骤1：初始状态
    step1 = [
        {'tokens': ['<START>'], 'score': 0.0, 'parent_idx': None}
    ]
    visualizer.add_step(step1)
    
    # 步骤2：第一个词
    step2 = [
        {'tokens': ['我'], 'score': -0.5, 'parent_idx': 0},
        {'tokens': ['学习'], 'score': -1.2, 'parent_idx': 0},
        {'tokens': ['编程'], 'score': -1.5, 'parent_idx': 0}
    ]
    visualizer.add_step(step2)
    
    # 步骤3：第二个词
    step3 = [
        {'tokens': ['我', '喜欢'], 'score': -0.8, 'parent_idx': 0},
        {'tokens': ['我', '学习'], 'score': -1.1, 'parent_idx': 0},
        {'tokens': ['学习', '人工智能'], 'score': -1.6, 'parent_idx': 1}
    ]
    visualizer.add_step(step3)
    
    # 步骤4：第三个词
    step4 = [
        {'tokens': ['我', '喜欢', '编程'], 'score': -1.0, 'parent_idx': 0},
        {'tokens': ['我', '喜欢', '学习'], 'score': -1.2, 'parent_idx': 0},
        {'tokens': ['我', '学习', '人工智能'], 'score': -1.5, 'parent_idx': 1}
    ]
    visualizer.add_step(step4)
    
    # 步骤5：结束
    step5 = [
        {'tokens': ['我', '喜欢', '编程', '<EOS>'], 'score': -1.1, 'parent_idx': 0, 'finished': True, 'best': True},
        {'tokens': ['我', '喜欢', '学习', '<EOS>'], 'score': -1.4, 'parent_idx': 1, 'finished': True},
        {'tokens': ['我', '学习', '人工智能', '很'], 'score': -1.8, 'parent_idx': 2}
    ]
    visualizer.add_step(step5)
    
    # 生成可视化
    visualizer.visualize()


def demonstrate_beam_pruning():
    """演示束剪枝过程"""
    print("\n=== 束剪枝演示 ===")
    print("假设beam_size=3，在某个时间步有6个候选：")
    
    candidates = [
        ("我喜欢编程", -1.2),
        ("我学习AI", -1.5),
        ("我喜欢学习", -1.8),
        ("编程很有趣", -2.0),
        ("学习很重要", -2.3),
        ("AI很强大", -2.5)
    ]
    
    print("\n扩展后的所有候选（按分数排序）：")
    for i, (text, score) in enumerate(candidates):
        print(f"{i+1}. {text}: {score:.2f}")
    
    print(f"\n剪枝后保留前{3}个候选：")
    for i, (text, score) in enumerate(candidates[:3]):
        print(f"{i+1}. {text}: {score:.2f} ✓")
    
    print("\n被剪枝的候选：")
    for i, (text, score) in enumerate(candidates[3:]):
        print(f"{i+4}. {text}: {score:.2f} ✗")


def demonstrate_score_calculation():
    """演示评分计算过程"""
    print("\n=== 评分计算演示 ===")
    
    # 模拟token概率
    tokens = ['我', '喜欢', '编程']
    log_probs = [-0.5, -0.3, -0.4]
    
    print("Token序列及其对数概率：")
    for token, log_prob in zip(tokens, log_probs):
        print(f"  {token}: {log_prob:.2f}")
    
    # 基础分数
    base_score = sum(log_probs)
    print(f"\n基础累积分数: {base_score:.2f}")
    
    # 长度归一化
    length = len(tokens)
    length_penalty = 0.7
    normalized_score = base_score / ((5 + length) / 6) ** length_penalty
    print(f"长度归一化后分数: {normalized_score:.2f}")
    print(f"  (长度惩罚因子: {length_penalty})")
    
    # 重复惩罚示例
    repetition_penalty = 1.2
    if '编程' in tokens[:-1]:  # 检查是否重复
        adjusted_score = normalized_score - 0.5
        print(f"应用重复惩罚后: {adjusted_score:.2f}")
        print(f"  (重复惩罚因子: {repetition_penalty})")


if __name__ == "__main__":
    # 生成束搜索过程可视化
    simulate_beam_search_process()
    
    # 演示剪枝过程
    demonstrate_beam_pruning()
    
    # 演示评分计算
    demonstrate_score_calculation()
    
    print("\n✅ 所有演示完成！")