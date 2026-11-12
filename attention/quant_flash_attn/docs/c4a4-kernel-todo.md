## C4A4 待完成的事项


### 功能支持问题

1. KV cache 的格式支持

    当前只支持了PA_BBND，PA_BNBD和PA_NZ 还不支持


### 精度验证标准化

1. 构建CPU和triton的参考实现，和算法侧的人对齐

    量化的实现方式、vector使用bf16还是fp32

2. 接入模型验收精度

    和模型侧对齐测试Qwen3.6/Qwen3.8的精度


### 性能优化问题

1. 流水、buffer的pipeline，重叠度如何

2. 寄存器代码的优化问题，当前vf代码可能还不符合寄存器高性能写法

3. M/N/K的选择，不同shape的模版特化，不规则形状的处理

    一些小shape，不对齐shape的特化处理

4. 是否合轴的选择标准
    prefill/decode 选择是否合轴


5. 标准的性能测试case


6. 接入模型的性能

### 辅助算子问题

要接入vllm-ascend，还要有mxfp4的在线量化算子，kv cache的写入算子
