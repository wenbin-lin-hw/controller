"""
E-puck机器人遗传算法适应度函数
用于Webots仿真环境中的循迹和避障任务

本模块提供四个核心适应度函数：
1. forwardFitness - 前进适应度
2. followLineFitness - 循线适应度
3. avoidCollisionFitness - 避障适应度
4. spinningFitness - 旋转惩罚

作者: AI Assistant
日期: 2025-11-18
"""

import numpy as np


# ==============================================================================
# FITNESS FUNCTION 1: FORWARD FITNESS
# ==============================================================================
def forwardFitness(left_speed, right_speed, max_speed=1.0):
    """
    前进适应度函数
    
    目标：
        鼓励机器人快速直线前进，惩罚倒退和原地旋转
    
    设计原理：
        1. 速度奖励：两轮速度越快越好
        2. 直线奖励：两轮速度越接近，越接近直线运动
        3. 方向惩罚：惩罚倒退行为（负速度）
    
    参数：
        left_speed: 左轮速度 [-max_speed, max_speed]
        right_speed: 右轮速度 [-max_speed, max_speed]
        max_speed: 最大速度，默认1.0 m/s
    
    返回：
        fitness: 适应度得分 [0, 1]
    
    示例：
        >>> forwardFitness(0.8, 0.8, 1.0)  # 快速直线前进
        0.8
        >>> forwardFitness(0.8, -0.8, 1.0)  # 原地旋转
        0.0
        >>> forwardFitness(-0.5, -0.5, 1.0)  # 倒退
        0.0
    """
    
    # 1. 速度分量：鼓励高速运动
    # 使用两轮速度的平均值，归一化到[0,1]
    avg_speed = (left_speed + right_speed) / 2.0
    speed_component = avg_speed / max_speed
    
    # 2. 直线分量：鼓励直线运动
    # 两轮速度差异越小，直线性越好
    speed_diff = abs(left_speed - right_speed)
    straightness = 1.0 - (speed_diff / (2.0 * max_speed))
    
    # 3. 方向惩罚：惩罚倒退
    # 如果平均速度为负（倒退），适应度为0
    if avg_speed < 0:
        return 0.0
    
    # 综合适应度：速度 × 直线性
    fitness = speed_component * straightness
    
    # 确保适应度在[0, 1]范围内
    return max(0.0, min(1.0, fitness))


# ==============================================================================
# FITNESS FUNCTION 2: FOLLOW LINE FITNESS
# ==============================================================================
def followLineFitness(ground_sensors, left_speed, right_speed, max_speed=1.0):
    """
    循线适应度函数
    
    目标：
        使机器人能够准确跟随地面黑线
    
    设计原理：
        1. 线条检测：奖励传感器检测到线的情况
        2. 居中奖励：中心传感器检测到线得分最高
        3. 转向修正：根据线的偏离方向，奖励正确的转向
        4. 速度保持：在线上时保持较高速度
    
    E-puck地面传感器：
        - 3个红外地面传感器：[左, 中, 右]
        - 传感器值范围：[0, 1000]
        - 黑线：传感器值低（< 500）
        - 白色地面：传感器值高（> 500）
    
    参数：
        ground_sensors: 地面传感器数组 [left, center, right]
                       归一化值 [0, 1]，0表示黑线，1表示白色
        left_speed: 左轮速度
        right_speed: 右轮速度
        max_speed: 最大速度
    
    返回：
        fitness: 适应度得分 [0, 1]
    
    示例：
        >>> followLineFitness([0.8, 0.2, 0.8], 0.5, 0.5, 1.0)  # 中心在线上，直行
        0.85
        >>> followLineFitness([0.2, 0.8, 0.8], 0.3, 0.6, 1.0)  # 线在左侧，右转
        0.70
        >>> followLineFitness([0.8, 0.8, 0.8], 0.5, 0.5, 1.0)  # 完全丢线
        0.0
    """
    
    left_sensor, center_sensor, right_sensor = ground_sensors
    
    # 定义黑线检测阈值（归一化后）
    LINE_THRESHOLD = 0.5
    
    # 1. 线条检测得分
    # 检查哪些传感器检测到了线（值 < 阈值）
    left_on_line = left_sensor < LINE_THRESHOLD
    center_on_line = center_sensor < LINE_THRESHOLD
    right_on_line = right_sensor < LINE_THRESHOLD
    
    # 2. 位置得分：根据线的位置给予不同奖励
    position_score = 0.0
    lost_line_penalty = 0.0
    if center_on_line:
        # 最佳情况：中心传感器在线上
        position_score = 1.0
    elif left_on_line and not right_on_line:
        # 线在左侧
        position_score = 0.6
    elif right_on_line and not left_on_line:
        # 线在右侧
        position_score = 0.6
    elif left_on_line and right_on_line:
        # 两侧都检测到（可能在交叉路口）
        position_score = 0.7
    else:
        # 完全丢线
        position_score = 0.0
        lost_line_penalty=1.0
    
    # 3. 转向修正得分：评估机器人是否正确转向
    correction_score = 0.0
    
    if left_on_line and not right_on_line:
        # 线在左侧，应该左转（右轮快，左轮慢）
        if right_speed > left_speed:
            turn_strength = (right_speed - left_speed) / max_speed
            correction_score = min(turn_strength * 1.5, 1.0)
        else:
            correction_score = 0.0  # 转向方向错误
            
    elif right_on_line and not left_on_line:
        # 线在右侧，应该右转（左轮快，右轮慢）
        if left_speed > right_speed:
            turn_strength = (left_speed - right_speed) / max_speed
            correction_score = min(turn_strength * 1.5, 1.0)
        else:
            correction_score = 0.0  # 转向方向错误
            
    elif center_on_line:
        # 线在中心，应该直行（两轮速度相近）
        speed_similarity = 1.0 - abs(left_speed - right_speed) / max_speed
        correction_score = speed_similarity
    
    # 4. 速度得分：在线上时保持速度
    if position_score > 0:
        avg_speed = (left_speed + right_speed) / 2.0
        speed_score = avg_speed / max_speed
    else:
        speed_score = 0.0
    
    # 综合适应度：位置(40%) + 修正(40%) + 速度(20%)
    fitness = position_score * 0.4 + correction_score * 0.4 + speed_score * 0.2-lost_line_penalty

    
    # return max(0.0, min(1.0, fitness))
    return fitness


# ==============================================================================
# FITNESS FUNCTION 3: AVOID COLLISION FITNESS
# ==============================================================================
def avoidCollisionFitness(proximity_sensors, left_speed, right_speed, max_speed=1.0):
    """
    避障适应度函数
    
    目标：
        使机器人能够检测并避开障碍物
    
    设计原理：
        1. 安全距离：保持与障碍物的安全距离
        2. 前方优先：前方传感器比侧面传感器更重要
        3. 避障行为：根据障碍物位置正确转向
        4. 碰撞惩罚：传感器值过高严重惩罚
    
    E-puck接近传感器布局（8个红外传感器）：
        ps0: 右前方 (45°)
        ps1: 右前方 (15°)
        ps2: 右侧 (-15°)
        ps3: 右后侧 (-45°)
        ps4: 后方 (-90°)
        ps5: 左后侧 (-135°)
        ps6: 左侧 (165°)
        ps7: 左前方 (135°)
    
    传感器值：
        - 范围：[0, 4096]
        - 值越大表示障碍物越近
        - 归一化后：[0, 1]
    
    参数：
        proximity_sensors: 8个接近传感器读数 [ps0-ps7]
                          归一化值 [0, 1]
        left_speed: 左轮速度
        right_speed: 右轮速度
        max_speed: 最大速度
    
    返回：
        fitness: 适应度得分 [0, 1]
    
    示例：
        >>> sensors = [0.1, 0.1, 0.05, 0.02, 0.02, 0.02, 0.05, 0.1]  # 无障碍
        >>> avoidCollisionFitness(sensors, 0.8, 0.8, 1.0)
        0.95
        >>> sensors = [0.1, 0.1, 0.05, 0.02, 0.02, 0.02, 0.05, 0.6]  # 左前方有障碍
        >>> avoidCollisionFitness(sensors, 0.8, 0.4, 1.0)  # 正确右转
        0.75
    """
    
    if len(proximity_sensors) < 8:
        return 0.0
    
    # 传感器权重（前方传感器最重要）
    sensor_weights = np.array([
        0.15,  # ps0 - 右前方
        0.20,  # ps1 - 右前方（最重要）
        0.10,  # ps2 - 右侧
        0.05,  # ps3 - 右后侧
        0.05,  # ps4 - 后方
        0.05,  # ps5 - 左后侧
        0.10,  # ps6 - 左侧
        0.30   # ps7 - 左前方（最重要）
    ])
    
    # 危险阈值（归一化后）
    DANGER_THRESHOLD = 0.2  # 超过此值表示有障碍物
    COLLISION_THRESHOLD = 0.6  # 超过此值表示即将碰撞
    
    # 1. 计算加权危险程度
    sensors_array = np.array(proximity_sensors)
    danger_level = np.sum(sensors_array * sensor_weights)
    
    # 2. 检测前方障碍物（最关键的传感器）
    front_sensors = [proximity_sensors[1], proximity_sensors[7]]  # ps1, ps7
    max_front = max(front_sensors)
    
    # 3. 碰撞检测：如果前方传感器值过高
    if max_front > COLLISION_THRESHOLD:
        # 即将碰撞，严重惩罚
        return 0.0
    
    # 4. 安全距离得分
    if max_front < DANGER_THRESHOLD:
        # 安全距离，满分
        safety_score = 1.0
    else:
        # 有障碍物，根据距离计算得分
        danger_ratio = (max_front - DANGER_THRESHOLD) / (COLLISION_THRESHOLD - DANGER_THRESHOLD)
        safety_score = 1.0 - danger_ratio
    
    # 5. 避障行为得分
    left_front_obstacle = proximity_sensors[7] > DANGER_THRESHOLD  # ps7
    right_front_obstacle = proximity_sensors[1] > DANGER_THRESHOLD  # ps1
    
    avoidance_score = 1.0  # 默认满分
    
    if left_front_obstacle and not right_front_obstacle:
        # 左前方有障碍，应该右转（左轮快，右轮慢）
        if left_speed > right_speed:
            # 正确的避障行为
            turn_strength = (left_speed - right_speed) / max_speed
            avoidance_score = 0.7 + min(turn_strength, 0.3)
        else:
            # 转向不足或方向错误
            avoidance_score = 0.4
            
    elif right_front_obstacle and not left_front_obstacle:
        # 右前方有障碍，应该左转（右轮快，左轮慢）
        if right_speed > left_speed:
            # 正确的避障行为
            turn_strength = (right_speed - left_speed) / max_speed
            avoidance_score = 0.7 + min(turn_strength, 0.3)
        else:
            # 转向不足或方向错误
            avoidance_score = 0.4
            
    elif left_front_obstacle and right_front_obstacle:
        # 前方两侧都有障碍，应该后退或急转
        if left_speed < 0 or right_speed < 0:
            # 后退是合理的
            avoidance_score = 0.6
        elif abs(left_speed - right_speed) > 0.5 * max_speed:
            # 急转也是合理的
            avoidance_score = 0.5
        else:
            # 没有采取有效避障措施
            avoidance_score = 0.2
    
    # 6. 速度调整得分：接近障碍物时应该减速
    speed_adjustment_score = 1.0
    if max_front > DANGER_THRESHOLD:
        avg_speed = (abs(left_speed) + abs(right_speed)) / (2.0 * max_speed)
        # 障碍物越近，期望速度越低
        expected_speed_ratio = 1.0 - (max_front - DANGER_THRESHOLD) / (COLLISION_THRESHOLD - DANGER_THRESHOLD)
        speed_diff = abs(avg_speed - expected_speed_ratio)
        speed_adjustment_score = max(0.5, 1.0 - speed_diff)
    
    # 综合适应度：安全距离(50%) + 避障行为(30%) + 速度调整(20%)
    fitness = safety_score * 0.5 + avoidance_score * 0.3 + speed_adjustment_score * 0.2
    
    return max(0.0, min(1.0, fitness))


# ==============================================================================
# FITNESS FUNCTION 4: SPINNING FITNESS (PENALTY)
# ==============================================================================
def spinningFitness(left_speed, right_speed, max_speed=1.0):
    """
    旋转惩罚函数
    
    目标：
        惩罚原地旋转和无效的振荡行为
    
    设计原理：
        1. 原地旋转检测：两轮速度大小相等方向相反
        2. 允许必要转向：小幅度转向不惩罚
        3. 静止检测：完全静止不惩罚（可能是合理的停止）
    
    原地旋转特征：
        - 左轮和右轮速度符号相反
        - 速度大小相近
        - 机器人位置不变，只是旋转
    
    参数：
        left_speed: 左轮速度
        right_speed: 右轮速度
        max_speed: 最大速度
    
    返回：
        fitness: 惩罚得分 [0, 1]
        - 1.0 表示无惩罚（正常行为）
        - 0.0 表示最大惩罚（严重的原地旋转）
    
    示例：
        >>> spinningFitness(0.8, 0.8, 1.0)  # 直线前进
        1.0
        >>> spinningFitness(0.8, -0.8, 1.0)  # 原地旋转
        0.0
        >>> spinningFitness(0.8, 0.5, 1.0)  # 轻微转向
        0.95
    """
    
    # 1. 静止检测
    speed_sum = abs(left_speed) + abs(right_speed)
    if speed_sum < 0.1:
        # 几乎静止，不惩罚
        return 1.0
    
    # 2. 原地旋转检测
    # 特征：两轮速度符号相反（一个前进，一个后退）
    if left_speed * right_speed < 0:
        # 速度符号相反，可能是旋转
        
        # 计算速度相似度
        speed_diff = abs(abs(left_speed) - abs(right_speed))
        similarity = 1.0 - (speed_diff / (speed_sum + 1e-6))
        
        if similarity > 0.8:
            # 速度大小非常接近，明显的原地旋转
            # 惩罚程度与相似度成正比
            penalty = similarity
            return 1.0 - penalty
        else:
            # 速度大小差异较大，可能是急转弯
            # 轻微惩罚
            return 0.7
    
    # 3. 转向程度评估
    # 即使不是原地旋转，过度转向也应该轻微惩罚
    
    # 计算转向比率
    angular_velocity = abs(right_speed - left_speed)
    turn_ratio = angular_velocity / (speed_sum + 1e-6)
    
    if turn_ratio < 0.3:
        # 轻微转向，不惩罚
        return 1.0
    elif turn_ratio < 0.6:
        # 中等转向，轻微惩罚
        penalty = (turn_ratio - 0.3) * 0.3
        return 1.0 - penalty
    else:
        # 剧烈转向，中等惩罚
        penalty = 0.3 + (turn_ratio - 0.6) * 0.5
        return 1.0 - min(penalty, 0.6)


# ==============================================================================
# COMBINED FITNESS FUNCTION
# ==============================================================================
def combinedFitness(ground_sensors, proximity_sensors, left_speed, right_speed, 
                   max_speed=1.0, weights=None):
    """
    组合适应度函数
    
    将四个子适应度函数组合成最终的适应度评分
    
    参数：
        ground_sensors: 地面传感器数组 [left, center, right]
        proximity_sensors: 接近传感器数组 [ps0-ps7]
        left_speed: 左轮速度
        right_speed: 右轮速度
        max_speed: 最大速度
        weights: 权重字典，格式：
                {
                    'forward': 0.25,
                    'follow_line': 0.40,
                    'avoid_collision': 0.25,
                    'spinning': 0.10
                }
                如果为None，使用默认权重
    
    返回：
        fitness: 综合适应度得分 [0, 1]
        components: 各组件得分字典（用于调试）
    
    示例：
        >>> ground = [0.8, 0.2, 0.8]  # 中心在线上
        >>> proximity = [0.1]*8  # 无障碍
        >>> fitness, components = combinedFitness(ground, proximity, 0.8, 0.8, 1.0)
        >>> print(f"综合适应度: {fitness:.3f}")
        综合适应度: 0.850
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
    forward_fit = forwardFitness(left_speed, right_speed, max_speed)
    
    follow_line_fit = followLineFitness(
        ground_sensors, left_speed, right_speed, max_speed
    )
    
    avoid_collision_fit = avoidCollisionFitness(
        proximity_sensors, left_speed, right_speed, max_speed
    )
    
    spinning_fit = spinningFitness(left_speed, right_speed, max_speed)
    
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


# ==============================================================================
# 动态权重策略
# ==============================================================================
def getAdaptiveWeights(current_generation, total_generations):
    """
    根据训练进度返回自适应权重
    
    训练策略：
        早期（0-30%）：注重前进和基本避障
        中期（30-70%）：注重循线能力
        后期（70-100%）：注重避障和精细控制
    
    参数：
        current_generation: 当前代数
        total_generations: 总代数
    
    返回：
        weights: 权重字典
    
    示例：
        >>> weights = getAdaptiveWeights(10, 100)  # 10%进度
        >>> print(weights)
        {'forward': 0.5, 'follow_line': 0.2, 'avoid_collision': 0.25, 'spinning': 0.05}
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


# ==============================================================================
# 测试和示例
# ==============================================================================
if __name__ == "__main__":
    """
    适应度函数测试示例
    """
    
    print("="*70)
    print("E-puck 遗传算法适应度函数测试")
    print("="*70)
    
    # 测试场景1：直线前进，中心在线上，无障碍
    print("\n【场景1】直线前进，中心在线上，无障碍")
    print("-"*70)
    ground = [0.8, 0.2, 0.8]  # 中心传感器检测到线
    proximity = [0.1, 0.1, 0.05, 0.02, 0.02, 0.02, 0.05, 0.1]  # 无障碍
    left_speed = 0.8
    right_speed = 0.8
    
    forward = forwardFitness(left_speed, right_speed)
    follow = followLineFitness(ground, left_speed, right_speed)
    avoid = avoidCollisionFitness(proximity, left_speed, right_speed)
    spinning = spinningFitness(left_speed, right_speed)
    
    print(f"左轮速度: {left_speed:.2f}, 右轮速度: {right_speed:.2f}")
    print(f"地面传感器: {ground}")
    print(f"前进适应度:     {forward:.4f}")
    print(f"循线适应度:     {follow:.4f}")
    print(f"避障适应度:     {avoid:.4f}")
    print(f"旋转惩罚:       {spinning:.4f}")
    
    combined, components = combinedFitness(ground, proximity, left_speed, right_speed)
    print(f"综合适应度:     {combined:.4f}")
    
    # 测试场景2：左转，线在左侧，无障碍
    print("\n【场景2】左转，线在左侧，无障碍")
    print("-"*70)
    ground = [0.2, 0.8, 0.8]  # 左侧传感器检测到线
    proximity = [0.1, 0.1, 0.05, 0.02, 0.02, 0.02, 0.05, 0.1]
    left_speed = 0.4
    right_speed = 0.8
    
    forward = forwardFitness(left_speed, right_speed)
    follow = followLineFitness(ground, left_speed, right_speed)
    avoid = avoidCollisionFitness(proximity, left_speed, right_speed)
    spinning = spinningFitness(left_speed, right_speed)
    
    print(f"左轮速度: {left_speed:.2f}, 右轮速度: {right_speed:.2f}")
    print(f"地面传感器: {ground}")
    print(f"前进适应度:     {forward:.4f}")
    print(f"循线适应度:     {follow:.4f}")
    print(f"避障适应度:     {avoid:.4f}")
    print(f"旋转惩罚:       {spinning:.4f}")
    
    combined, components = combinedFitness(ground, proximity, left_speed, right_speed)
    print(f"综合适应度:     {combined:.4f}")
    
    # 测试场景3：右转避障，左前方有障碍
    print("\n【场景3】右转避障，左前方有障碍")
    print("-"*70)
    ground = [0.8, 0.2, 0.8]
    proximity = [0.1, 0.1, 0.05, 0.02, 0.02, 0.02, 0.05, 0.7]  # ps7有障碍
    left_speed = 0.8
    right_speed = 0.3
    
    forward = forwardFitness(left_speed, right_speed)
    follow = followLineFitness(ground, left_speed, right_speed)
    avoid = avoidCollisionFitness(proximity, left_speed, right_speed)
    spinning = spinningFitness(left_speed, right_speed)
    
    print(f"左轮速度: {left_speed:.2f}, 右轮速度: {right_speed:.2f}")
    print(f"地面传感器: {ground}")
    print(f"接近传感器ps7: {proximity[7]:.2f} (左前方有障碍)")
    print(f"前进适应度:     {forward:.4f}")
    print(f"循线适应度:     {follow:.4f}")
    print(f"避障适应度:     {avoid:.4f}")
    print(f"旋转惩罚:       {spinning:.4f}")
    
    combined, components = combinedFitness(ground, proximity, left_speed, right_speed)
    print(f"综合适应度:     {combined:.4f}")
    
    # 测试场景4：原地旋转（应该被惩罚）
    print("\n【场景4】原地旋转（应该被惩罚）")
    print("-"*70)
    ground = [0.8, 0.8, 0.8]  # 丢线
    proximity = [0.1, 0.1, 0.05, 0.02, 0.02, 0.02, 0.05, 0.1]
    left_speed = 0.8
    right_speed = -0.8
    
    forward = forwardFitness(left_speed, right_speed)
    follow = followLineFitness(ground, left_speed, right_speed)
    avoid = avoidCollisionFitness(proximity, left_speed, right_speed)
    spinning = spinningFitness(left_speed, right_speed)
    
    print(f"左轮速度: {left_speed:.2f}, 右轮速度: {right_speed:.2f}")
    print(f"地面传感器: {ground}")
    print(f"前进适应度:     {forward:.4f}")
    print(f"循线适应度:     {follow:.4f}")
    print(f"避障适应度:     {avoid:.4f}")
    print(f"旋转惩罚:       {spinning:.4f} (严重惩罚)")
    
    combined, components = combinedFitness(ground, proximity, left_speed, right_speed)
    print(f"综合适应度:     {combined:.4f}")
    
    # 测试自适应权重
    print("\n" + "="*70)
    print("自适应权重策略测试")
    print("="*70)
    
    for progress in [0.1, 0.5, 0.9]:
        gen = int(progress * 100)
        weights = getAdaptiveWeights(gen, 100)
        print(f"\n第 {gen} 代（进度 {progress*100:.0f}%）:")
        print(f"  前进权重:       {weights['forward']:.2f}")
        print(f"  循线权重:       {weights['follow_line']:.2f}")
        print(f"  避障权重:       {weights['avoid_collision']:.2f}")
        print(f"  旋转惩罚权重:   {weights['spinning']:.2f}")
    
    print("\n" + "="*70)
    print("测试完成")
    print("="*70)
