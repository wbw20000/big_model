"""
束搜索（Beam Search）候选管理机制详解与实现

束搜索是大模型生成文本时常用的解码策略，通过维护固定数量的最优候选序列来平衡生成质量和计算效率。
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional
import heapq
from dataclasses import dataclass
import numpy as np


@dataclass
class BeamCandidate:
    """束搜索候选项"""
    tokens: List[int]          # token序列
    score: float              # 累积对数概率
    attention_cache: Optional[torch.Tensor] = None  # 注意力缓存
    
    def __lt__(self, other):
        # 用于优先队列比较，分数越高越好
        return self.score > other.score


class BeamSearchManager:
    """束搜索候选管理器"""
    
    def __init__(self, 
                 beam_size: int = 5,
                 max_length: int = 100,
                 length_penalty: float = 1.0,
                 temperature: float = 1.0,
                 repetition_penalty: float = 1.0):
        """
        初始化束搜索管理器
        
        Args:
            beam_size: 束宽度，同时维护的候选序列数量
            max_length: 最大生成长度
            length_penalty: 长度惩罚因子，控制生成长度偏好
            temperature: 温度参数，控制概率分布的平滑程度
            repetition_penalty: 重复惩罚因子，减少重复生成
        """
        self.beam_size = beam_size
        self.max_length = max_length
        self.length_penalty = length_penalty
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        
        # 当前活跃的候选序列
        self.active_beams: List[BeamCandidate] = []
        # 已完成的候选序列
        self.finished_beams: List[BeamCandidate] = []
        
    def initialize(self, start_token: int):
        """初始化束搜索，创建初始候选"""
        initial_candidate = BeamCandidate(
            tokens=[start_token],
            score=0.0
        )
        self.active_beams = [initial_candidate]
        
    def update_candidates(self, 
                         logits: torch.Tensor, 
                         eos_token_id: int) -> bool:
        """
        更新候选序列
        
        Args:
            logits: 模型输出的logits，shape: (beam_size, vocab_size)
            eos_token_id: 结束token的ID
            
        Returns:
            是否所有候选都已完成
        """
        # 存储所有新的候选
        all_candidates = []
        
        for beam_idx, beam in enumerate(self.active_beams):
            # 获取当前beam的logits
            beam_logits = logits[beam_idx]
            
            # 应用温度缩放
            if self.temperature != 1.0:
                beam_logits = beam_logits / self.temperature
            
            # 应用重复惩罚
            beam_logits = self._apply_repetition_penalty(
                beam_logits, beam.tokens
            )
            
            # 计算概率分布
            probs = F.softmax(beam_logits, dim=-1)
            log_probs = F.log_softmax(beam_logits, dim=-1)
            
            # 获取top-k个候选token
            # 使用2*beam_size确保有足够的候选
            top_k = min(2 * self.beam_size, probs.size(-1))
            top_probs, top_indices = torch.topk(probs, top_k)
            
            for i in range(top_k):
                token_id = top_indices[i].item()
                token_log_prob = log_probs[token_id].item()
                
                # 创建新的候选序列
                new_tokens = beam.tokens + [token_id]
                new_score = beam.score + token_log_prob
                
                # 应用长度归一化
                normalized_score = self._length_normalize(
                    new_score, len(new_tokens)
                )
                
                new_candidate = BeamCandidate(
                    tokens=new_tokens,
                    score=normalized_score
                )
                
                # 检查是否是结束token
                if token_id == eos_token_id:
                    self.finished_beams.append(new_candidate)
                else:
                    all_candidates.append(new_candidate)
        
        # 候选剪枝：保留得分最高的beam_size个候选
        self.active_beams = self._prune_candidates(all_candidates)
        
        # 检查是否达到最大长度
        if self.active_beams and len(self.active_beams[0].tokens) >= self.max_length:
            self.finished_beams.extend(self.active_beams)
            self.active_beams = []
        
        return len(self.active_beams) == 0
    
    def _apply_repetition_penalty(self, 
                                 logits: torch.Tensor, 
                                 generated_tokens: List[int]) -> torch.Tensor:
        """应用重复惩罚"""
        if self.repetition_penalty == 1.0:
            return logits
            
        # 对已生成的token施加惩罚
        for token in set(generated_tokens):
            if logits[token] < 0:
                logits[token] = logits[token] * self.repetition_penalty
            else:
                logits[token] = logits[token] / self.repetition_penalty
                
        return logits
    
    def _length_normalize(self, score: float, length: int) -> float:
        """长度归一化，避免偏向短序列"""
        # 使用Wu et al. (2016)提出的长度惩罚公式
        length_penalty = ((5 + length) / 6) ** self.length_penalty
        return score / length_penalty
    
    def _prune_candidates(self, candidates: List[BeamCandidate]) -> List[BeamCandidate]:
        """剪枝候选序列，保留得分最高的beam_size个"""
        if len(candidates) <= self.beam_size:
            return sorted(candidates, key=lambda x: x.score, reverse=True)
        
        # 使用堆来高效获取top-k
        return heapq.nlargest(self.beam_size, candidates, key=lambda x: x.score)
    
    def get_best_sequence(self) -> List[int]:
        """获取最佳生成序列"""
        all_candidates = self.finished_beams + self.active_beams
        if not all_candidates:
            return []
            
        best_candidate = max(all_candidates, key=lambda x: x.score)
        return best_candidate.tokens


class DiverseBeamSearch(BeamSearchManager):
    """多样性束搜索，增强生成的多样性"""
    
    def __init__(self, 
                 beam_size: int = 5,
                 num_groups: int = 5,
                 diversity_penalty: float = 0.5,
                 **kwargs):
        """
        Args:
            num_groups: 束组数量
            diversity_penalty: 多样性惩罚强度
        """
        super().__init__(beam_size=beam_size, **kwargs)
        self.num_groups = num_groups
        self.diversity_penalty = diversity_penalty
        self.group_size = beam_size // num_groups
        
    def update_candidates_with_diversity(self, 
                                       logits: torch.Tensor,
                                       eos_token_id: int) -> bool:
        """带多样性的候选更新"""
        vocab_size = logits.size(-1)
        
        # 将beams分组
        beam_groups = [[] for _ in range(self.num_groups)]
        for i, beam in enumerate(self.active_beams):
            group_idx = i % self.num_groups
            beam_groups[group_idx].append((i, beam))
        
        all_candidates = []
        
        # 对每个组依次处理
        for group_idx in range(self.num_groups):
            group_beams = beam_groups[group_idx]
            
            for beam_idx, beam in group_beams:
                beam_logits = logits[beam_idx].clone()
                
                # 应用多样性惩罚
                if group_idx > 0:
                    # 对前面组已选择的token施加惩罚
                    for prev_group_idx in range(group_idx):
                        for _, prev_beam in beam_groups[prev_group_idx]:
                            if prev_beam.tokens:
                                last_token = prev_beam.tokens[-1]
                                beam_logits[last_token] -= self.diversity_penalty
                
                # 继续正常的束搜索流程
                probs = F.softmax(beam_logits, dim=-1)
                log_probs = F.log_softmax(beam_logits, dim=-1)
                
                top_k = min(self.group_size * 2, vocab_size)
                top_probs, top_indices = torch.topk(probs, top_k)
                
                for i in range(top_k):
                    token_id = top_indices[i].item()
                    new_candidate = BeamCandidate(
                        tokens=beam.tokens + [token_id],
                        score=beam.score + log_probs[token_id].item()
                    )
                    
                    if token_id == eos_token_id:
                        self.finished_beams.append(new_candidate)
                    else:
                        all_candidates.append(new_candidate)
        
        self.active_beams = self._prune_candidates(all_candidates)
        return len(self.active_beams) == 0


# 使用示例
def beam_search_example():
    """束搜索使用示例"""
    # 模拟词表大小
    vocab_size = 50000
    beam_size = 5
    
    # 创建束搜索管理器
    beam_manager = BeamSearchManager(
        beam_size=beam_size,
        max_length=50,
        length_penalty=0.7,
        temperature=0.8,
        repetition_penalty=1.2
    )
    
    # 初始化
    start_token_id = 1  # <START>
    eos_token_id = 2    # <EOS>
    beam_manager.initialize(start_token_id)
    
    # 模拟生成过程
    for step in range(20):
        # 模拟模型输出logits
        # 实际使用时这里应该是模型的forward pass
        logits = torch.randn(beam_size, vocab_size)
        
        # 更新候选
        finished = beam_manager.update_candidates(logits, eos_token_id)
        
        if finished:
            print(f"生成在第{step}步完成")
            break
    
    # 获取最佳序列
    best_sequence = beam_manager.get_best_sequence()
    print(f"最佳生成序列: {best_sequence}")
    
    # 展示所有完成的候选
    print(f"\n所有完成的候选序列（共{len(beam_manager.finished_beams)}个）:")
    for i, candidate in enumerate(beam_manager.finished_beams[:5]):
        print(f"候选{i+1}: 分数={candidate.score:.4f}, 长度={len(candidate.tokens)}")


if __name__ == "__main__":
    # 运行示例
    beam_search_example()
    
    # 测试多样性束搜索
    print("\n" + "="*50)
    print("多样性束搜索示例:")
    
    diverse_manager = DiverseBeamSearch(
        beam_size=10,
        num_groups=5,
        diversity_penalty=1.0,
        max_length=30
    )
    
    diverse_manager.initialize(1)
    
    # 模拟一步更新
    logits = torch.randn(10, 50000)
    diverse_manager.update_candidates_with_diversity(logits, 2)
    
    print(f"活跃候选数: {len(diverse_manager.active_beams)}")