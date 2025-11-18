"""
E-puck机器人遗传算法适应度函数模块
用于Webots仿真环境中的循迹和避障任务

作者: Genetic Algorithm Training System
日期: 2025
"""

import numpy as np


class FitnessFunctions:
    """
    适应度函数集合类
    包含四个主要的适应度评估函数，用于训练e-puck机器人
    """

    def __init__(self, max_speed=1.0):
        """
        初始化适应度函数参数
        
        Args:
            max_speed: 机器人最大速度 (m/s)
        """
        self.max_speed = max_speed
        self.SENSOR_MAX = 4096.0  # e-puck传感器最大值
        self.GROUND_SENSOR_THRESHOLD = 500  # 地面传感器阈值（黑线检测）
        self.DANGER_THRESHOLD = 90  # 障碍物危险距离阈值
        
    # ============================================================================
    # FITNESS FUNCTION 1: FORWARD FITNESS
    # ============================================================================
    def forwardFitness(self, left_speed, right_speed, real_speed=None, is_on_edge=False):
        """
        前进适应度函数
        
        目标：鼓励机器人快速直线前进，避免倒退和原地不动
        
        设计原理：
        1. 速度奖励：两轮速度越快越好（鼓励快速移动）
        2. 直线奖励：两轮速度差异越小越好（鼓励直线行驶）
        3. 方向惩罚：惩罚倒退行为
        4. 静止惩罚：惩罚原地不动或速度过慢
        5. 边界惩罚：惩罚接近或超出边界的行为
        
        Args:
            left_speed: 左轮速度 [-max_speed, max_speed]
            right_speed: 右轮速度 [-max_speed, max_speed]
            real_speed: 机器人实际速度（可选，用于更精确的评估）
            is_on_edge: 是否在边界上（True表示超出安全区域）
            
        Returns:
            fitness: 适应度得分 [0, 1]
        """
        
        # 1. 速度奖励：鼓励快速移动
        # 使用绝对值的平均速度，归一化到[0,1]
        speed_reward = (abs(left_speed) + abs(right_speed)) / (2 * self.max_speed)
        
        # 2. 直线奖励：两轮速度差异越小，直线行驶越好
        speed_difference = abs(left_speed - right_speed) / self.max_speed
        straightness_reward = 1.0 - speed_difference
        
        # 3. 方向惩罚：惩罚倒退
        direction_penalty = 0.0
        if left_speed < 0 or right_speed < 0:
            # 如果任一轮倒退，施加惩罚
            backward_ratio = (abs(min(left_speed, 0)) + abs(min(right_speed, 0))) / self.max_speed
            direction_penalty = 0.5 * backward_ratio
        
        # 4. 静止惩罚：惩罚速度过慢
        static_penalty = 0.0
        if real_speed is not None and real_speed < 0.01:
            # 实际速度过慢（几乎静止）
            static_penalty = 0.3
        
        # # 5. 边界惩罚：严重惩罚超出边界
        # edge_penalty = 0.0
        # if is_on_edge:
        #     edge_penalty = 0.5
        
        # 综合适应度计算
        # 基础得分 = 速度奖励 × 直线奖励
        base_fitness = speed_reward * straightness_reward
        
        # 应用所有惩罚
        fitness = base_fitness - direction_penalty - static_penalty
        
        # 确保适应度在[0, 1]范围内
        return max(0.0, min(1.0, fitness))
    
    
    # ============================================================================
    # FITNESS FUNCTION 2: FOLLOW LINE FITNESS
    # ============================================================================
    def followLineFitness(self, ground_sensors, left_speed, right_speed, 
                         real_speed=None, is_on_edge=False):
        """
        循线适应度函数
        
        目标：使机器人能够准确跟随地面上的黑线
        
        设计原理：
        1. 线条检测：奖励检测到线的情况
        2. 位置评估：中心传感器检测到线得分最高
        3. 方向修正：根据线的位置调整转向
        4. 速度保持：在线上时保持较高速度
        5. 丢线惩罚：完全丢失线的严重惩罚
        
        e-puck地面传感器布局：
        - gs0 (left): 左侧传感器
        - gs1 (center): 中心传感器  
        - gs2 (right): 右侧传感器
        传感器值 < 500 表示检测到黑线
        
        Args:
            ground_sensors: 地面传感器数组 [left, center, right]
            left_speed: 左轮速度
            right_speed: 右轮速度
            real_speed: 机器人实际速度（可选）
            is_on_edge: 是否在边界上
            
        Returns:
            fitness: 适应度得分 [0, 1]
        """
        
        left_sensor, center_sensor, right_sensor = ground_sensors
        
        # 1. 线条检测奖励
        line_detection_reward = 0.0
        on_line = False
        
        if center_sensor < self.GROUND_SENSOR_THRESHOLD:
            # 最佳情况：中心传感器在线上
            line_detection_reward = 1.0
            on_line = True
        elif left_sensor < self.GROUND_SENSOR_THRESHOLD or right_sensor < self.GROUND_SENSOR_THRESHOLD:
            # 次优情况：偏离中心但仍能检测到线
            line_detection_reward = 0.6
            on_line = True
        else:
            # 最差情况：完全丢线
            line_detection_reward = 0.0
            on_line = False
        
        # 2. 方向修正奖励：评估转向是否正确
        correction_reward = 0.0
        
        if left_sensor < self.GROUND_SENSOR_THRESHOLD and right_sensor > self.GROUND_SENSOR_THRESHOLD:
            # 线在左侧，应该左转（左轮慢，右轮快）
            if right_speed > left_speed:
                speed_diff_ratio = (right_speed - left_speed) / self.max_speed
                correction_reward = min(speed_diff_ratio, 1.0) * 0.9
                
        elif right_sensor < self.GROUND_SENSOR_THRESHOLD and left_sensor > self.GROUND_SENSOR_THRESHOLD:
            # 线在右侧，应该右转（右轮慢，左轮快）
            if left_speed > right_speed:
                speed_diff_ratio = (left_speed - right_speed) / self.max_speed
                correction_reward = min(speed_diff_ratio, 1.0) * 0.9
                
        elif center_sensor < self.GROUND_SENSOR_THRESHOLD:
            # 线在中心，应该直行（两轮速度相近）
            speed_similarity = 1.0 - abs(left_speed - right_speed) / self.max_speed
            correction_reward = speed_similarity
        
        # 3. 速度奖励：在线上时保持速度
        speed_reward = 0.0
        if on_line:
            avg_speed = (abs(left_speed) + abs(right_speed)) / (2 * self.max_speed)
            speed_reward = avg_speed * 0.8
        
        # 4. 丢线惩罚
        lost_line_penalty = 0.0
        if not on_line:
            lost_line_penalty = 0.8
        
        # 5. 静止惩罚
        static_penalty = 0.0
        if real_speed is not None and real_speed < 0.01:
            static_penalty = 0.5
        
        # 6. 边界惩罚
        edge_penalty = 0.0
        if is_on_edge:
            edge_penalty = 1.0  # 循线时超出边界是严重错误
        
        # 综合适应度计算
        # 权重分配：检测(40%) + 修正(30%) + 速度(30%)
        fitness = (line_detection_reward * 0.4 + 
                  correction_reward * 0.3 + 
                  speed_reward * 0.3 - 
                  lost_line_penalty - 
                  static_penalty - 
                  edge_penalty)
        
        return max(-0.3, min(1.0, fitness))
    
    
    # ============================================================================
    # FITNESS FUNCTION 3: AVOID COLLISION FITNESS
    # ============================================================================
    def avoidCollisionFitness(self, proximity_sensors, left_speed, right_speed, 
                             real_speed=None, is_on_edge=False):
        """
        避障适应度函数
        
        目标：使机器人能够检测并避开障碍物
        
        设计原理：
        1. 危险检测：识别前方和侧面的障碍物
        2. 安全距离：保持与障碍物的安全距离
        3. 避障响应：根据障碍物位置正确调整轮速
        4. 碰撞惩罚：传感器值过高严重惩罚
        5. 减速奖励：接近障碍物时适当减速
        
        e-puck接近传感器布局（8个传感器）：
        - ps0, ps1: 右前方（最重要）
        - ps2, ps3: 右侧
        - ps4, ps5: 后方
        - ps6, ps7: 左侧/左前方（最重要）
        传感器值越高表示障碍物越近
        
        Args:
            proximity_sensors: 8个接近传感器读数 [ps0-ps7]
            left_speed: 左轮速度
            right_speed: 右轮速度
            real_speed: 机器人实际速度（可选）
            is_on_edge: 是否在边界上
            
        Returns:
            fitness: 适应度得分 [0, 1]
        """
        
        if len(proximity_sensors) < 8:
            return 0.0
        
        # 传感器权重（前方传感器最重要）
        sensor_weights = np.array([
            0.20,  # ps0 - 右前
            0.20,  # ps1 - 右前
            0.10,  # ps2 - 右侧
            0.05,  # ps3 - 右后侧
            0.05,  # ps4 - 后方
            0.05,  # ps5 - 左后侧
            0.10,  # ps6 - 左侧
            0.25   # ps7 - 左前（最重要）
        ])
        
        # 归一化传感器读数到[0, 1]
        norm_sensors = np.array(proximity_sensors) / self.SENSOR_MAX
        
        # 计算加权危险程度
        danger_level = np.sum(norm_sensors * sensor_weights)
        
        # 1. 检测前方障碍物（最关键的三个传感器）
        front_sensors = [proximity_sensors[0], proximity_sensors[1], proximity_sensors[7]]
        max_front = max(front_sensors)
        
        # 2. 碰撞检测：如果前方传感器值过高，表示即将碰撞
        if max_front > self.DANGER_THRESHOLD * 3:
            return -5.0  # 严重碰撞，零分
        
        # 3. 安全距离评分
        if max_front < self.DANGER_THRESHOLD:
            # 安全距离，高分
            safety_score = 1.0
        else:
            # 有障碍物，根据距离评分
            # 距离越近，分数越低
            danger_ratio = (max_front - self.DANGER_THRESHOLD) / (self.DANGER_THRESHOLD * 2)
            safety_score = 1.0 - min(danger_ratio, 0.8)
        
        # 4. 避障行为评估
        left_obstacle = proximity_sensors[7] > self.DANGER_THRESHOLD  # 左前方有障碍
        right_obstacle = proximity_sensors[0] > self.DANGER_THRESHOLD  # 右前方有障碍
        
        avoidance_score = 0.5  # 默认中等分数
        
        if left_obstacle and not right_obstacle:
            # 左侧有障碍，应该右转（右轮慢，左轮快）
            if left_speed > right_speed:
                turn_strength = (left_speed - right_speed) / self.max_speed
                avoidance_score = min(turn_strength, 1.0)
            else:
                avoidance_score = 0.2  # 转向方向错误
                
        elif right_obstacle and not left_obstacle:
            # 右侧有障碍，应该左转（左轮慢，右轮快）
            if right_speed > left_speed:
                turn_strength = (right_speed - left_speed) / self.max_speed
                avoidance_score = min(turn_strength, 1.0)
            else:
                avoidance_score = 0.2  # 转向方向错误
                
        elif left_obstacle and right_obstacle:
            # 两侧都有障碍，应该后退或急转
            if left_speed < 0 or right_speed < 0:
                avoidance_score = 0.6  # 后退是合理的避障策略
            else:
                avoidance_score = 0.3
        else:
            # 没有明显障碍，保持前进
            avoidance_score = 1.0
        
        # 5. 减速奖励：接近障碍物时应该减速
        speed_adjustment_score = 1.0
        if max_front > self.DANGER_THRESHOLD:
            avg_speed = (abs(left_speed) + abs(right_speed)) / (2 * self.max_speed)
            # 障碍物越近，速度应该越慢
            expected_speed = 1.0 - (max_front - self.DANGER_THRESHOLD) / (self.DANGER_THRESHOLD * 2)
            speed_diff = abs(avg_speed - expected_speed)
            speed_adjustment_score = 1.0 - speed_diff * 0.5
        
        # 6. 静止惩罚（避免卡住不动）
        static_penalty = 0.0
        if real_speed is not None and real_speed < 0.02:
            # 如果设置了较高的轮速但实际速度很低，说明可能卡住了
            commanded_speed = max(abs(left_speed), abs(right_speed))
            if commanded_speed > 0.5:
                static_penalty = 0.5
        
        # 7. 边界惩罚
        edge_penalty = 0.0
        if is_on_edge:
            edge_penalty = 0.3
        
        # 综合适应度计算
        # 权重分配：安全距离(40%) + 避障行为(35%) + 速度调整(25%)
        fitness = (safety_score * 0.40 + 
                  avoidance_score * 0.35 + 
                  speed_adjustment_score * 0.25 - 
                  static_penalty - 
                  edge_penalty)
        
        return max(-5.0, min(1.0, fitness))
    
    
    # ============================================================================
    # FITNESS FUNCTION 4: SPINNING FITNESS (PENALTY)
    # ============================================================================
    def spinningFitness(self, left_speed, right_speed, real_speed=None, 
                       angular_velocity_history=None):
        """
        旋转惩罚函数
        
        目标：惩罚原地旋转和无效的振荡行为
        
        设计原理：
        1. 原地旋转检测：两轮速度大小相等方向相反
        2. 振荡检测：频繁改变转向方向
        3. 允许必要转向：小幅度转向不惩罚
        4. 卡住检测：轮速高但实际速度低
        
        Args:
            left_speed: 左轮速度
            right_speed: 右轮速度
            real_speed: 机器人实际速度（可选）
            angular_velocity_history: 历史角速度记录（可选，用于检测振荡）
            
        Returns:
            fitness: 惩罚得分 [0, 1]，1表示无惩罚，0表示最大惩罚
        """
        
        # 1. 基本速度检查
        speed_sum = abs(left_speed) + abs(right_speed)
        
        if speed_sum < 0.1:
            # 几乎静止，不惩罚（可能是合理的停止）
            return 1.0
        
        # 2. 原地旋转检测
        # 特征：两轮速度方向相反且大小相近
        if left_speed * right_speed < 0:  # 符号相反
            speed_diff = abs(abs(left_speed) - abs(right_speed))
            similarity = 1.0 - speed_diff / (speed_sum + 1e-6)
            
            if similarity > 0.8:
                # 明显的原地旋转
                # 检查是否有实际移动
                if real_speed is not None and real_speed < 0.005:
                    # 原地旋转且没有实际移动，严重惩罚
                    return 0.0
                else:
                    # 有一定移动的旋转，中等惩罚
                    spinning_penalty = similarity * 0.7
                    return 1.0 - spinning_penalty
        
        # 3. 卡住检测：轮速高但实际速度低
        if real_speed is not None:
            max_wheel_speed = max(abs(left_speed), abs(right_speed))
            if max_wheel_speed > 0.5 and real_speed < 0.01:
                # 轮子在转但机器人不动，可能卡住了
                return 0.0
        
        # 4. 振荡行为检测
        if angular_velocity_history is not None and len(angular_velocity_history) > 5:
            recent_history = angular_velocity_history[-10:]
            
            # 计算方向改变次数
            direction_changes = 0
            for i in range(1, len(recent_history)):
                if recent_history[i] * recent_history[i-1] < 0:
                    direction_changes += 1
            
            # 频繁改变方向表示振荡
            if direction_changes > 5:
                oscillation_penalty = min(direction_changes / 10.0, 0.6)
                return 1.0 - oscillation_penalty
        
        # 5. 转向程度评估
        # 计算转向比率
        angular_velocity = right_speed - left_speed
        turn_ratio = abs(angular_velocity) / (speed_sum + 1e-6)
        
        if turn_ratio < 0.3:
            # 轻微转向，不惩罚
            return 1.0
        elif turn_ratio < 0.6:
            # 中等转向，轻微惩罚
            return 1.0 - (turn_ratio - 0.3) * 0.3
        else:
            # 剧烈转向，中等惩罚
            return 1.0 - turn_ratio * 0.4
    
    
    # ============================================================================
    # COMBINED FITNESS FUNCTION
    # ============================================================================
    def combinedFitness(self, ground_sensors, proximity_sensors, 
                       left_speed, right_speed, real_speed=None, 
                       is_on_edge=False, angular_velocity_history=None,
                       weights=None):
        """
        组合适应度函数
        
        将四个子适应度函数组合成最终的适应度评分
        
        Args:
            ground_sensors: 地面传感器数组 [left, center, right]
            proximity_sensors: 接近传感器数组 [ps0-ps7]
            left_speed: 左轮速度
            right_speed: 右轮速度
            real_speed: 机器人实际速度
            is_on_edge: 是否在边界上
            angular_velocity_history: 角速度历史
            weights: 权重字典，格式：
                    {
                        'forward': 0.3,
                        'follow_line': 0.4,
                        'avoid_collision': 0.2,
                        'spinning': 0.1
                    }
                    如果为None，使用默认权重
                    
        Returns:
            fitness: 综合适应度得分 [0, 1]
            components: 各组件得分字典（用于调试）
        """
        
        # 默认权重
        if weights is None:
            weights = {
                'forward': 0.25,
                'follow_line': 0.40,
                'avoid_collision': 0.25,
                'spinning': 0.10
            }
        
        # 计算各个子适应度
        forward_fit = self.forwardFitness(
            left_speed, right_speed, real_speed, is_on_edge
        )
        
        follow_line_fit = self.followLineFitness(
            ground_sensors, left_speed, right_speed, real_speed, is_on_edge
        )
        
        avoid_collision_fit = self.avoidCollisionFitness(
            proximity_sensors, left_speed, right_speed, real_speed, is_on_edge
        )
        
        spinning_fit = self.spinningFitness(
            left_speed, right_speed, real_speed, angular_velocity_history
        )
        
        # 组合适应度
        combined_fit = (
            forward_fit * weights['forward'] +
            follow_line_fit * weights['follow_line'] +
            avoid_collision_fit * weights['avoid_collision'] +
            spinning_fit * weights['spinning']
        )
        
        # 返回综合得分和各组件得分
        components = {
            'forward': forward_fit,
            'follow_line': follow_line_fit,
            'avoid_collision': avoid_collision_fit,
            'spinning': spinning_fit,
            'combined': combined_fit
        }
        
        return combined_fit, components


# ============================================================================
# 动态权重策略
# ============================================================================
class AdaptiveWeightStrategy:
    """
    自适应权重策略
    根据训练阶段动态调整各适应度函数的权重
    """
    
    @staticmethod
    def get_weights_by_generation(current_generation, total_generations):
        """
        根据当前代数返回适应度权重
        
        训练策略：
        - 早期（0-30%）：注重前进和基本避障
        - 中期（30-70%）：注重循线能力
        - 后期（70-100%）：注重避障和精细控制
        
        Args:
            current_generation: 当前代数
            total_generations: 总代数
            
        Returns:
            weights: 权重字典
        """
        progress = current_generation / total_generations
        
        if progress <= 0.3:
            # 早期：学习基本移动
            return {
                'forward': 0.50,
                'follow_line': 0.20,
                'avoid_collision': 0.25,
                'spinning': 0.05
            }
        elif progress <= 0.7:
            # 中期：学习循线
            return {
                'forward': 0.25,
                'follow_line': 0.50,
                'avoid_collision': 0.20,
                'spinning': 0.05
            }
        else:
            # 后期：精细控制和避障
            return {
                'forward': 0.20,
                'follow_line': 0.40,
                'avoid_collision': 0.35,
                'spinning': 0.05
            }


# ============================================================================
# 使用示例
# ============================================================================
if __name__ == "__main__":
    """
    适应度函数使用示例
    """
    
    # 创建适应度函数实例
    fitness_funcs = FitnessFunctions(max_speed=1.0)
    
    # 模拟传感器数据
    ground_sensors = [600, 400, 600]  # 中心传感器检测到线
    proximity_sensors = [50, 60, 30, 20, 15, 20, 30, 80]  # 左前方有障碍
    left_speed = 0.8
    right_speed = 0.7
    real_speed = 0.06
    is_on_edge = False
    
    # 计算各个适应度
    print("=" * 60)
    print("适应度函数测试")
    print("=" * 60)
    
    forward_fit = fitness_funcs.forwardFitness(left_speed, right_speed, real_speed, is_on_edge)
    print(f"前进适应度: {forward_fit:.4f}")
    
    follow_line_fit = fitness_funcs.followLineFitness(
        ground_sensors, left_speed, right_speed, real_speed, is_on_edge
    )
    print(f"循线适应度: {follow_line_fit:.4f}")
    
    avoid_collision_fit = fitness_funcs.avoidCollisionFitness(
        proximity_sensors, left_speed, right_speed, real_speed, is_on_edge
    )
    print(f"避障适应度: {avoid_collision_fit:.4f}")
    
    spinning_fit = fitness_funcs.spinningFitness(left_speed, right_speed, real_speed)
    print(f"旋转惩罚: {spinning_fit:.4f}")
    
    # 计算组合适应度
    combined_fit, components = fitness_funcs.combinedFitness(
        ground_sensors, proximity_sensors, left_speed, right_speed,
        real_speed, is_on_edge
    )
    
    print("\n" + "=" * 60)
    print("组合适应度结果")
    print("=" * 60)
    for key, value in components.items():
        print(f"{key}: {value:.4f}")
    
    # 测试自适应权重
    print("\n" + "=" * 60)
    print("自适应权重策略")
    print("=" * 60)
    
    for gen_ratio in [0.1, 0.5, 0.9]:
        gen = int(gen_ratio * 100)
        weights = AdaptiveWeightStrategy.get_weights_by_generation(gen, 100)
        print(f"\n第 {gen} 代（{gen_ratio*100:.0f}%）:")
        for key, value in weights.items():
            print(f"  {key}: {value:.2f}")
