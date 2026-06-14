import numpy as np

def levenshtein_distance(s1, s2):
    """
    计算两个字符串之间的 Levenshtein 编辑距离
    用动态规划法实现
    """
    # 转换为字符列表以支持 Unicode 字符的正确索引（如中文汉字）
    s1 = list(s1)
    s2 = list(s2)
    
    m, n = len(s1), len(s2)
    dp = np.zeros((m + 1, n + 1), dtype=int)
    
    # 初始化边界条件
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
        
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                cost = 0
            else:
                cost = 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # 删除操作
                dp[i][j - 1] + 1,      # 插入操作
                dp[i - 1][j - 1] + cost # 替换操作
            )
            
    return int(dp[m][n])

def character_error_rate(reference, hypothesis):
    """
    计算字符错误率 (Character Error Rate, CER)
    公式: CER = Levenshtein(Ref, Hyp) / len(Ref)
    如果 Ref 为空，则当 Hyp 也为空时返回 0，否则返回 1.0
    """
    reference = reference.strip()
    hypothesis = hypothesis.strip()
    
    if len(reference) == 0:
        return 0.0 if len(hypothesis) == 0 else 1.0
        
    dist = levenshtein_distance(reference, hypothesis)
    return float(dist) / len(reference)
