import re
import random
import jiwer
import numpy as np
import itertools
from swift.plugin import ORM, orms

# ==================== 正则解析模块 ====================
_COT_PATTERN = re.compile(r"<think>.*?estimated speaker-number:\s*(\d+).*?</think>", re.DOTALL | re.IGNORECASE)
_TAG1 = re.compile(r"<spk1>(.*?)</spk1>", re.DOTALL | re.IGNORECASE)
_TAG2 = re.compile(r"<spk2>(.*?)</spk2>", re.DOTALL | re.IGNORECASE)
_TAG3 = re.compile(r"<spk3>(.*?)</spk3>", re.DOTALL | re.IGNORECASE)
_TIME_PATTERN = re.compile(r"\[([\d\.]+)s\s*-\s*([\d\.]+)s\]")

def _norm(s: str) -> str:
    s = (s or "").lower()
    return re.sub(r"\s+", " ", s).strip()

def get_time(text: str) -> list:
    matches = _TIME_PATTERN.findall(text)
    if matches:
        return [float(matches[0][0]), float(matches[0][1])]
    return None

# ==================== 核心打分引擎 ====================
def compute_mixed_mcar_reward(pred_text: str, ref_a: str, ref_b: str, ref_c: str, true_spk_count: int):
    # --- 1. CoT 推理人数打分 (极其关键的初期引导信号) ---
    cot_match = _COT_PATTERN.search(pred_text)
    pred_spk_count = int(cot_match.group(1)) if cot_match else -1
    
    r_cot = 0.0
    if pred_spk_count == true_spk_count:
        r_cot = 1.0  # 猜对人数直接给大奖，稳住大模型心态
    elif pred_spk_count != -1:
        r_cot = -0.5 # 猜错了扣分
    else:
        r_cot = -1.0 # 连 CoT 格式都不遵守，重罚
        
    # --- 2. 文本解析 ---
    ra, rb, rc = _norm(ref_a), _norm(ref_b), _norm(ref_c)
    
    m1, m2, m3 = _TAG1.search(pred_text), _TAG2.search(pred_text), _TAG3.search(pred_text)
    p1 = _norm(m1.group(1)) if m1 else ""
    p2 = _norm(m2.group(1)) if m2 else ""
    p3 = _norm(m3.group(1)) if m3 else ""
    
    refs = [ra, rb, rc]
    preds = [p1, p2, p3]
    
    # --- 3. 统一 3! PI-WER 计算 (优雅处理人数不匹配与 jiwer 空值 Bug) ---
    best_wer = float('inf')
    best_perm = (0, 1, 2)
    
    for perm in itertools.permutations([0, 1, 2]):
        err, length = 0, 0
        for ref_idx, pred_idx in enumerate(perm):
            r_text = re.sub(r'\[[\d\.]+s\s*-\s*[\d\.]+s\]\s*', '', refs[ref_idx]).strip()
            p_text = re.sub(r'\[[\d\.]+s\s*-\s*[\d\.]+s\]\s*', '', preds[pred_idx]).strip()
            
            # --- 🔥 修复 jiwer 空字符串报错的完美分流逻辑 ---
            if not r_text and p_text:
                # 纯幻觉 (Insertion): 参考为空，预测有字。错误数直接加上预测词数
                err += len(p_text.split())
            elif r_text and not p_text:
                # 纯漏听 (Deletion): 预测为空，参考有字。错误数等于参考词数
                err += len(r_text.split())
                length += len(r_text.split())
            elif r_text and p_text:
                # 都有文本，安全调用 jiwer
                out = jiwer.process_words(r_text, p_text)
                err += out.substitutions + out.deletions + out.insertions
                length += len(r_text.split())
                
        # 防止除以 0 的极端情况
        wer = err / max(length, 1)
        if wer < best_wer:
            best_wer = wer
            best_perm = perm

    r_sem = np.exp(-2.0 * best_wer) 

    # --- 4. 时间戳误差对齐计算 ---
    time_error = 0.0
    matched_time_tags = 0
    ref_times = [get_time(r) for r in refs]
    pred_times = [get_time(p) for p in preds]
    
    for ref_idx, pred_idx in enumerate(best_perm):
        r_t = ref_times[ref_idx]
        p_t = pred_times[pred_idx]
        if r_t and p_t:
            time_error += abs(p_t[0] - r_t[0]) + abs(p_t[1] - r_t[1])
            matched_time_tags += 1
            
    avg_time_error = time_error / max(matched_time_tags, 1)
    r_time_acc = np.exp(-1.0 * max(0, avg_time_error - 0.2)) if matched_time_tags > 0 else 0.0

    # --- 5. 格式与逻辑惩罚 ---
    r_logic = 0.0
    # 惩罚时间倒挂
    matches = _TIME_PATTERN.findall(pred_text)
    for start_str, end_str in matches:
        try:
            if float(start_str) >= float(end_str): r_logic -= 1.0
        except: pass
        
    # 惩罚知行不一 (CoT 猜了 3 个人，却只输出了 2 个标签)
    actual_tag_count = sum(1 for m in [m1, m2, m3] if m)
    if pred_spk_count != -1 and pred_spk_count != actual_tag_count:
        r_logic -= 0.5 

    # --- 综合计分 ---
    # CoT 得分占了非常大的比重，引导模型先学会数人头！
    total_reward = 0.4 * r_sem + 0.3 * r_cot + 0.2 * r_time_acc + r_logic
    total_reward = float(max(min(total_reward, 1.0), 0.0))
    
    return total_reward, best_wer, r_sem, r_cot, r_time_acc, r_logic

# ==================== 插件包装 ====================
def reward_func(completions, **kwargs) -> list[float]:
    ref_a_list = kwargs.get('ref_a', [""] * len(completions))
    ref_b_list = kwargs.get('ref_b', [""] * len(completions))
    ref_c_list = kwargs.get('ref_c', [""] * len(completions))
    # 💡 获取底层的标准答案人数
    spk_counts = kwargs.get('spk_count', [2] * len(completions)) 
    
    rewards = []
    for i, (pred, ra, rb, rc, count) in enumerate(zip(completions, ref_a_list, ref_b_list, ref_c_list, spk_counts)):
        total_reward, wer, r_sem, r_cot, r_time, r_logic = compute_mixed_mcar_reward(pred, ra, rb, rc, count)
        rewards.append(total_reward)
        
        if random.random() < 0.005:
            print(f"\n[MCAR Mixed CoT] 🎯 True Spk: {count} | 🏆 Tot: {total_reward:.3f} | 📉 PI-WER: {wer:.3f}")
            print(f"   ├─ Sem: {r_sem:.2f} | CoT: {r_cot:.2f} | Time: {r_time:.2f} | Logic: {r_logic:.2f}")
            print(f"   └─ Pred: {pred.strip()[:120]}...")
            
    return rewards

class MCARRewardPlugin:
    def __init__(self, **kwargs): pass
    def __call__(self, completions, **kwargs): return reward_func(completions, **kwargs)

orms['mcar_mixed_reward'] = MCARRewardPlugin
