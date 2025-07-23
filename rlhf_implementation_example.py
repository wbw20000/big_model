"""
RLHF中 (1-rewards) 权重系数的实现示例
展示奖励与损失反向关系的实际应用
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


class RLHFTrainer:
    """RLHF训练器，展示(1-rewards)权重的使用"""
    
    def __init__(self, model, reward_model, learning_rate=1e-4):
        self.model = model
        self.reward_model = reward_model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
    def compute_loss_with_reward_weighting(self, logits, labels, rewards):
        """
        计算带有(1-rewards)权重的损失
        
        Args:
            logits: 模型输出的logits
            labels: 真实标签
            rewards: 奖励值 (0到1之间)
        
        Returns:
            weighted_loss: 加权后的损失
            metrics: 包含各种指标的字典
        """
        # 计算基础策略损失（交叉熵）
        policy_loss = F.cross_entropy(logits, labels, reduction='none')
        
        # 应用(1-rewards)权重
        # 高奖励 -> 低权重 -> 小损失
        # 低奖励 -> 高权重 -> 大损失
        weights = 1 - rewards
        weighted_loss = weights * policy_loss
        
        # 计算平均损失
        mean_loss = weighted_loss.mean()
        
        # 收集指标
        metrics = {
            'policy_loss': policy_loss.mean().item(),
            'weighted_loss': mean_loss.item(),
            'mean_reward': rewards.mean().item(),
            'mean_weight': weights.mean().item(),
            'max_reward': rewards.max().item(),
            'min_reward': rewards.min().item()
        }
        
        return mean_loss, metrics
    
    def train_step(self, inputs, labels):
        """执行一步训练"""
        # 前向传播
        logits = self.model(inputs)
        
        # 获取奖励（这里模拟奖励模型的输出）
        with torch.no_grad():
            rewards = self.reward_model(inputs, logits)
            rewards = torch.sigmoid(rewards)  # 确保在[0,1]范围内
        
        # 计算加权损失
        loss, metrics = self.compute_loss_with_reward_weighting(logits, labels, rewards)
        
        # 反向传播和优化
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return metrics


def visualize_reward_weight_relationship():
    """可视化奖励与权重的关系"""
    rewards = np.linspace(0, 1, 100)
    
    # 基础线性关系
    weights_linear = 1 - rewards
    
    # 平方关系（加强效果）
    weights_squared = (1 - rewards) ** 2
    
    # 开方关系（减弱效果）
    weights_sqrt = np.sqrt(1 - rewards)
    
    # 分段函数
    weights_piecewise = np.where(rewards > 0.5, 
                                0.2 * (1 - rewards),  # 高奖励区域
                                1.5 * (1 - rewards))  # 低奖励区域
    
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, weights_linear, label='Linear: (1-r)', linewidth=2)
    plt.plot(rewards, weights_squared, label='Squared: (1-r)²', linewidth=2)
    plt.plot(rewards, weights_sqrt, label='Square root: √(1-r)', linewidth=2)
    plt.plot(rewards, weights_piecewise, label='Piecewise', linewidth=2, linestyle='--')
    
    plt.xlabel('Reward', fontsize=12)
    plt.ylabel('Weight', fontsize=12)
    plt.title('Reward-Weight Relationship in RLHF', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('reward_weight_relationship.png', dpi=150)
    plt.close()


def demonstrate_gradient_scaling():
    """演示梯度缩放效果"""
    # 创建示例数据
    batch_size = 4
    vocab_size = 100
    
    # 模拟不同质量的输出及其对应的奖励
    scenarios = [
        ("优秀输出", 0.95),
        ("良好输出", 0.75),
        ("一般输出", 0.50),
        ("较差输出", 0.25)
    ]
    
    print("梯度缩放演示：")
    print("-" * 50)
    
    for name, reward in scenarios:
        # 模拟logits和标签
        logits = torch.randn(1, vocab_size, requires_grad=True)
        labels = torch.randint(0, vocab_size, (1,))
        
        # 计算基础损失
        base_loss = F.cross_entropy(logits, labels)
        
        # 计算加权损失
        weight = 1 - reward
        weighted_loss = weight * base_loss
        
        # 计算梯度
        base_loss.backward(retain_graph=True)
        base_grad_norm = torch.norm(logits.grad)
        logits.grad.zero_()
        
        weighted_loss.backward()
        weighted_grad_norm = torch.norm(logits.grad)
        
        print(f"{name} (reward={reward:.2f}):")
        print(f"  权重: {weight:.2f}")
        print(f"  基础损失: {base_loss.item():.4f}")
        print(f"  加权损失: {weighted_loss.item():.4f}")
        print(f"  梯度范数比例: {weighted_grad_norm/base_grad_norm:.2f}")
        print()


class AdvancedRLHFLoss(nn.Module):
    """高级RLHF损失函数，支持多种权重策略"""
    
    def __init__(self, weight_type='linear', temperature=1.0, 
                 min_weight=0.01, max_weight=2.0):
        super().__init__()
        self.weight_type = weight_type
        self.temperature = temperature
        self.min_weight = min_weight
        self.max_weight = max_weight
        
    def compute_weights(self, rewards):
        """根据不同策略计算权重"""
        if self.weight_type == 'linear':
            weights = 1 - rewards
            
        elif self.weight_type == 'squared':
            weights = (1 - rewards) ** 2
            
        elif self.weight_type == 'exponential':
            weights = torch.exp(self.temperature * (1 - rewards)) - 1
            
        elif self.weight_type == 'adaptive':
            # 自适应权重：低奖励时更激进，高奖励时更保守
            weights = torch.where(
                rewards > 0.7,
                0.1 * (1 - rewards),  # 高奖励区域
                torch.where(
                    rewards > 0.3,
                    1.0 * (1 - rewards),  # 中等奖励区域
                    2.0 * (1 - rewards)   # 低奖励区域
                )
            )
        else:
            raise ValueError(f"Unknown weight type: {self.weight_type}")
        
        # 限制权重范围
        weights = torch.clamp(weights, self.min_weight, self.max_weight)
        
        return weights
    
    def forward(self, logits, labels, rewards):
        """计算RLHF损失"""
        # 基础策略损失
        policy_loss = F.cross_entropy(logits, labels, reduction='none')
        
        # 计算权重
        weights = self.compute_weights(rewards)
        
        # 加权损失
        weighted_loss = weights * policy_loss
        
        return weighted_loss.mean(), {
            'weights': weights,
            'policy_loss': policy_loss,
            'weighted_loss': weighted_loss
        }


def compare_weight_strategies():
    """比较不同权重策略的效果"""
    # 生成示例数据
    batch_size = 100
    rewards = torch.linspace(0, 1, batch_size)
    
    # 不同的权重策略
    strategies = {
        'linear': AdvancedRLHFLoss(weight_type='linear'),
        'squared': AdvancedRLHFLoss(weight_type='squared'),
        'exponential': AdvancedRLHFLoss(weight_type='exponential', temperature=2.0),
        'adaptive': AdvancedRLHFLoss(weight_type='adaptive')
    }
    
    plt.figure(figsize=(12, 8))
    
    for name, loss_fn in strategies.items():
        weights = loss_fn.compute_weights(rewards).numpy()
        plt.plot(rewards.numpy(), weights, label=name, linewidth=2)
    
    plt.xlabel('Reward', fontsize=12)
    plt.ylabel('Weight', fontsize=12)
    plt.title('Comparison of Weight Strategies in RLHF', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('weight_strategies_comparison.png', dpi=150)
    plt.close()


if __name__ == "__main__":
    print("RLHF (1-rewards) 权重系数演示\n")
    
    # 1. 可视化奖励-权重关系
    print("1. 生成奖励-权重关系图...")
    visualize_reward_weight_relationship()
    print("   已保存到: reward_weight_relationship.png\n")
    
    # 2. 演示梯度缩放
    print("2. 梯度缩放效果演示：")
    demonstrate_gradient_scaling()
    
    # 3. 比较不同权重策略
    print("3. 生成权重策略比较图...")
    compare_weight_strategies()
    print("   已保存到: weight_strategies_comparison.png\n")
    
    # 4. 实际应用示例
    print("4. 实际应用示例：")
    print("-" * 50)
    
    # 创建简单的模型和奖励模型（仅用于演示）
    class SimpleModel(nn.Module):
        def __init__(self, input_dim, output_dim):
            super().__init__()
            self.fc = nn.Linear(input_dim, output_dim)
            
        def forward(self, x):
            return self.fc(x)
    
    class SimpleRewardModel(nn.Module):
        def __init__(self):
            super().__init__()
            
        def forward(self, inputs, logits):
            # 模拟奖励：基于logits的某种度量
            return torch.randn_like(logits[:, 0])
    
    # 初始化模型
    model = SimpleModel(10, 50)
    reward_model = SimpleRewardModel()
    trainer = RLHFTrainer(model, reward_model)
    
    # 模拟训练步骤
    for i in range(3):
        inputs = torch.randn(8, 10)  # batch_size=8, input_dim=10
        labels = torch.randint(0, 50, (8,))  # vocab_size=50
        
        metrics = trainer.train_step(inputs, labels)
        
        print(f"步骤 {i+1}:")
        print(f"  策略损失: {metrics['policy_loss']:.4f}")
        print(f"  加权损失: {metrics['weighted_loss']:.4f}")
        print(f"  平均奖励: {metrics['mean_reward']:.4f}")
        print(f"  平均权重: {metrics['mean_weight']:.4f}")
        print()
    
    print("\n演示完成！")
    print("\n关键要点总结：")
    print("1. (1-rewards) 实现了奖励与损失的反向关系")
    print("2. 高奖励导致低权重，减少梯度更新")
    print("3. 低奖励导致高权重，增加梯度更新")
    print("4. 这种机制帮助模型保持好的行为，改进差的行为")