"""
束搜索候选管理机制的简化演示
不需要外部依赖，展示核心概念
"""

import random
from typing import List, Tuple, Dict
import heapq


class SimpleBeamSearch:
    """简化的束搜索实现，用于演示候选管理机制"""
    
    def __init__(self, vocab: List[str], beam_size: int = 3):
        self.vocab = vocab
        self.beam_size = beam_size
        self.vocab_to_id = {word: i for i, word in enumerate(vocab)}
        self.id_to_vocab = {i: word for i, word in enumerate(vocab)}
        
    def get_mock_probabilities(self, prefix: List[str]) -> Dict[str, float]:
        """模拟获取下一个词的概率分布"""
        # 这里使用简单的规则来模拟概率
        probs = {}
        
        if not prefix or prefix == ['<START>']:
            # 开始时的概率分布
            probs = {
                '我': 0.4,
                '学习': 0.3,
                '编程': 0.2,
                '人工智能': 0.1
            }
        elif prefix[-1] == '我':
            probs = {
                '喜欢': 0.5,
                '学习': 0.3,
                '是': 0.2
            }
        elif prefix[-1] == '喜欢':
            probs = {
                '编程': 0.4,
                '学习': 0.3,
                '人工智能': 0.3
            }
        elif prefix[-1] == '学习':
            probs = {
                '人工智能': 0.5,
                '编程': 0.3,
                '很': 0.2
            }
        elif prefix[-1] in ['编程', '人工智能']:
            probs = {
                '很': 0.4,
                '<EOS>': 0.6
            }
        elif prefix[-1] == '很':
            probs = {
                '有趣': 0.5,
                '重要': 0.3,
                '困难': 0.2
            }
        elif prefix[-1] in ['有趣', '重要', '困难']:
            probs = {
                '<EOS>': 1.0
            }
        else:
            # 默认结束
            probs = {'<EOS>': 1.0}
            
        # 为词表中其他词分配小概率
        remaining_prob = 1.0 - sum(probs.values())
        other_words = [w for w in self.vocab if w not in probs and w != '<START>']
        if other_words and remaining_prob > 0:
            for word in other_words:
                probs[word] = remaining_prob / len(other_words)
                
        return probs
    
    def beam_search(self, max_length: int = 10) -> List[Tuple[List[str], float]]:
        """执行束搜索"""
        print("=== 束搜索过程演示 ===")
        print(f"束宽度: {self.beam_size}")
        print(f"最大长度: {max_length}\n")
        
        # 初始化：只有一个候选 <START>
        active_beams = [(['<START>'], 0.0)]
        finished_beams = []
        
        step = 0
        while active_beams and step < max_length:
            print(f"\n--- 步骤 {step + 1} ---")
            print(f"当前活跃候选数: {len(active_beams)}")
            
            # 存储所有新的候选
            all_candidates = []
            
            # 对每个活跃的束进行扩展
            for beam_idx, (tokens, score) in enumerate(active_beams):
                print(f"\n扩展候选 {beam_idx + 1}: {' '.join(tokens)} (分数: {score:.3f})")
                
                # 获取下一个词的概率分布
                probs = self.get_mock_probabilities(tokens)
                
                # 选择概率最高的几个词
                top_words = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]
                
                print("  可能的下一个词:")
                for word, prob in top_words[:3]:  # 只显示前3个
                    log_prob = -abs(random.gauss(1.5, 0.5))  # 模拟对数概率
                    new_score = score + log_prob
                    print(f"    - {word}: 概率={prob:.3f}, 对数概率={log_prob:.3f}, 新分数={new_score:.3f}")
                    
                    # 创建新候选
                    new_tokens = tokens[1:] + [word] if tokens[0] == '<START>' else tokens + [word]
                    
                    if word == '<EOS>':
                        finished_beams.append((new_tokens[:-1], new_score))
                        print(f"      → 序列完成！")
                    else:
                        all_candidates.append((new_tokens, new_score))
            
            # 候选剪枝：保留分数最高的beam_size个
            if all_candidates:
                print(f"\n剪枝前候选总数: {len(all_candidates)}")
                all_candidates.sort(key=lambda x: x[1], reverse=True)
                active_beams = all_candidates[:self.beam_size]
                
                print(f"剪枝后保留的候选:")
                for i, (tokens, score) in enumerate(active_beams):
                    print(f"  {i + 1}. {' '.join(tokens)} (分数: {score:.3f})")
                
                if len(all_candidates) > self.beam_size:
                    print(f"\n被剪枝的候选:")
                    for i, (tokens, score) in enumerate(all_candidates[self.beam_size:self.beam_size+3]):
                        print(f"  - {' '.join(tokens)} (分数: {score:.3f})")
                    if len(all_candidates) > self.beam_size + 3:
                        print(f"  ... 还有 {len(all_candidates) - self.beam_size - 3} 个候选被剪枝")
            else:
                active_beams = []
            
            step += 1
        
        # 合并所有候选并排序
        all_results = finished_beams + [(tokens, score) for tokens, score in active_beams]
        all_results.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n\n=== 最终结果 ===")
        print(f"完成的序列数: {len(finished_beams)}")
        print(f"未完成的序列数: {len(active_beams)}")
        print(f"\n最佳序列 (前5个):")
        for i, (tokens, score) in enumerate(all_results[:5]):
            print(f"  {i + 1}. {' '.join(tokens)} (分数: {score:.3f})")
        
        return all_results


def demonstrate_score_normalization():
    """演示分数归一化的重要性"""
    print("\n\n=== 长度归一化演示 ===")
    print("为什么需要长度归一化？")
    print("因为累积对数概率总是负数，越长的序列分数越低\n")
    
    sequences = [
        (['我', '喜欢', '编程'], -3.6),
        (['我', '喜欢', '学习', '人工智能', '和', '编程'], -7.2),
        (['编程', '很', '有趣'], -3.5)
    ]
    
    print("未归一化的分数:")
    for tokens, score in sequences:
        print(f"  {' '.join(tokens)}")
        print(f"    长度: {len(tokens)}, 分数: {score:.2f}")
    
    print("\n应用长度归一化 (α=0.7):")
    alpha = 0.7
    for tokens, score in sequences:
        length = len(tokens)
        normalized = score / ((5 + length) / 6) ** alpha
        print(f"  {' '.join(tokens)}")
        print(f"    归一化分数: {normalized:.2f}")


def demonstrate_beam_width_impact():
    """演示束宽度对结果的影响"""
    print("\n\n=== 束宽度影响演示 ===")
    
    vocab = ['<START>', '我', '喜欢', '学习', '编程', '人工智能', 
             '很', '有趣', '重要', '困难', '<EOS>']
    
    for beam_size in [1, 3, 5]:
        print(f"\n使用束宽度 = {beam_size}:")
        searcher = SimpleBeamSearch(vocab, beam_size=beam_size)
        
        # 简化输出，只显示最终结果
        import sys
        from io import StringIO
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        results = searcher.beam_search(max_length=8)
        
        sys.stdout = old_stdout
        
        print(f"最佳结果: {' '.join(results[0][0])} (分数: {results[0][1]:.3f})")
        print(f"找到的不同序列数: {len(results)}")


if __name__ == "__main__":
    # 主要演示
    vocab = ['<START>', '我', '喜欢', '学习', '编程', '人工智能', 
             '很', '有趣', '重要', '困难', '是', '<EOS>']
    
    searcher = SimpleBeamSearch(vocab, beam_size=3)
    searcher.beam_search(max_length=8)
    
    # 其他演示
    demonstrate_score_normalization()
    demonstrate_beam_width_impact()
    
    print("\n\n✅ 演示完成！")