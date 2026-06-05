作业 10：Sinusoidal Position Encoding 与 RoPE
```powershell
python rope_position_encoding_demo.py
```

1. Transformer 为什么需要位置编码

Transformer 的 self-attention 本身对输入顺序不敏感。如果只给模型一组 token embedding，交换 token 顺序后，attention 计算没有天然办法知道“谁在前、谁在后”。因此需要位置编码把序列顺序信息注入模型。

2. Sinusoidal Position Encoding 如何生成

传统正弦位置编码对每个位置 `pos` 和维度对 `2i, 2i+1` 使用不同频率：

```text
PE(pos, 2i)     = sin(pos / 10000^(2i / d_model))
PE(pos, 2i + 1) = cos(pos / 10000^(2i / d_model))
```

代码实现见 `sinusoidal_position_encoding()`。它生成形状为 `[seq_len, dim]` 的位置矩阵，然后传统 Transformer 会使用：

```text
hidden = token_embedding + position_encoding
```

3. E + pos 的问题：内容和位置混合

`E + pos` 把 token 内容向量和位置向量直接相加。这样做简单有效，但会把“内容信息”和“位置信息”混到同一个表示里。后续线性层和 attention 看到的是混合后的向量，无法明确区分某一部分来自 token 内容，某一部分来自位置。

从点积角度看：

```text
(E_m + P_m)^T (E_n + P_n)
= E_m^T E_n + E_m^T P_n + P_m^T E_n + P_m^T P_n
```

其中 `E_m^T P_n` 和 `P_m^T E_n` 是内容与位置的交叉项，所以位置注入不够“干净”。

4. RoPE 不是加法，而是旋转

RoPE（Rotary Position Embedding）把向量的每两个维度看成一个二维平面，在每个位置上按不同角频率旋转。二维旋转矩阵为：

```text
R(theta) = [[cos(theta), -sin(theta)],
            [sin(theta),  cos(theta)]]
```

代码实现：
`rotate_2d()`：二维向量旋转。
`apply_rope_one()`：对一个高维向量按位置进行 RoPE 旋转。
`apply_rope_batch()`：对整个序列应用 RoPE。

5. RoPE 作用在 Q 和 K 上

传统加法位置编码通常在输入 embedding 阶段执行：

```text
X' = E + pos
Q = X' W_Q
K = X' W_K
V = X' W_V
```

RoPE 通常作用在 attention 的 `Q` 和 `K` 上：

```text
Q'_m = R_m Q_m
K'_n = R_n K_n
score(m, n) = (Q'_m)^T K'_n
```

一般不对 `V` 做 RoPE，因为 attention score 需要位置信息来决定“看哪里”，而 `V` 主要负责提供被加权汇聚的内容。

6. RoPE 的点积天然包含相对位置

由于旋转矩阵满足：

```text
R_m^T R_n = R_(n-m)
```

所以 RoPE 后的 attention score 为：

```text
(R_m q)^T (R_n k)
= q^T R_m^T R_n k
= q^T R_(n-m) k
```

这说明 score 中自然出现了相对位置 `n - m`，而不是只依赖绝对位置 `m` 和 `n`。

脚本中的 `verify_rope_relative_property()` 会用随机向量验证：

```text
<R_m q, R_n k> == <q, R_(n-m) k>
```

运行结果的最大误差通常在 `1e-15` 量级，说明数值上成立。

7. 为什么 RoPE 比简单 E + pos 更巧妙

RoPE 的优势在于：
它不把位置向量直接加进内容向量，而是通过旋转改变 `Q` 和 `K` 的几何关系。
attention score 中会自然出现相对位置 `n - m`。
内容向量本身不需要和位置向量硬混合，位置关系主要体现在查询和键的匹配过程里。
对长序列和相对位置建模更友好，因此被很多大语言模型采用。

