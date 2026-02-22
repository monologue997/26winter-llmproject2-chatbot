# 评估系统使用说明

这个文件夹包含MP2 Part-4的评估系统实现。

## 文件结构

```
evulation/
├── generate_dataset.py   # 测试数据集生成器
├── evaluate.py           # 评估管道和LLM评判者
├── test_prompts.json         # 生成的测试数据集（运行后生成）
└── README.md            # 本说明文件
```

## 组件说明

### 1. `generate_dataset.py` - 测试数据集生成器

**包含的类：**
- `TestDatasetGenerator`: 使用LLM生成6种类别的测试用例

**测试类别：**
1. **obnoxious** (无理/冒犯性查询)
   - 例子：粗鲁的问题、冒犯性语言
   - 期望行为：礼貌拒绝

2. **irrelevant** (不相关查询)
   - 例子：与机器学习无关的问题
   - 期望行为：说明问题不在书本范围内

3. **relevant** (相关查询)
   - 例子：机器学习相关的正常问题
   - 期望行为：基于书本内容准确回答

4. **small_talk** (闲聊)
   - 例子："你好"、"谢谢"等
   - 期望行为：简短友好回复

5. **hybrid** (混合型查询)
   - 例子：同时包含相关和不相关内容的问题
   - 期望行为：只回答相关部分

6. **multi_turn** (多轮对话)
   - 例子：需要上下文的连续对话
   - 期望行为：正确理解和使用对话历史

### 2. `evaluate.py` - 评估管道

**包含的类：**
- `LLM_Judge`: LLM-as-a-Judge评估器
- `EvaluationPipeline`: 评估流程编排器

**评估流程：**
1. 加载测试数据集
2. 将测试用例发送给chatbot
3. 使用LLM评判响应质量
4. 汇总统计结果

## 使用步骤

### 步骤1：生成测试数据集

```python
from openai import OpenAI
from generate_dataset import TestDatasetGenerator

# 初始化
client = OpenAI(api_key="your-api-key")
generator = TestDatasetGenerator(client)

# 生成数据集
generator.build_full_dataset()
generator.save_dataset("test_set.json")
```

### 步骤2：运行评估

```python
from evaluate import LLM_Judge, EvaluationPipeline
from agents.head_agent import Head_Agent  # 你的Part-3实现

# 初始化组件
head_agent = Head_Agent(...)
judge = LLM_Judge(client)
pipeline = EvaluationPipeline(head_agent, judge)

# 加载数据
data = generator.load_dataset("test_set.json")

# 运行测试
pipeline.run_single_turn_test("obnoxious", data["obnoxious"])
pipeline.run_single_turn_test("irrelevant", data["irrelevant"])
pipeline.run_single_turn_test("relevant", data["relevant"])
pipeline.run_single_turn_test("small_talk", data["small_talk"])
pipeline.run_single_turn_test("hybrid", data["hybrid"])
pipeline.run_multi_turn_test(data["multi_turn"])

# 查看结果
pipeline.calculate_metrics()
```

## 评估指标

评估系统会计算：
- **每个类别的准确率**：该类别通过的测试用例比例
- **总体准确率**：所有测试用例的通过率

评分为二元分类：
- **1 (PASS)**: chatbot响应符合期望
- **0 (FAIL)**: chatbot响应不符合期望

## 实现提示

### TestDatasetGenerator.generate_synthetic_prompts()
- 构造详细的prompt，说明每个类别的特征
- 要求LLM生成多样化的测试用例
- 解析JSON格式的输出

### LLM_Judge.construct_judge_prompt()
- 为每个类别定义清晰的评判标准
- 包含用户输入和bot响应
- 要求输出结构化的评分（PASS/FAIL）

### EvaluationPipeline.run_single_turn_test()
- 遍历测试用例
- 调用head_agent处理查询
- 记录响应和使用的agent路径
- 调用judge进行评分

### EvaluationPipeline.run_multi_turn_test()
- 维护对话历史
- 可能需要重置chatbot状态
- 评估上下文理解能力

## 与其他部分的集成

这个评估系统依赖于：
- **Part-3的Head_Agent**: 需要导入并初始化
- **所有的agents**: Obnoxious, Query, Relevant, Answering等
- **OpenAI API**: 用于生成测试数据和评判

## 注意事项

1. **API费用**: 生成数据集和评估都会调用OpenAI API，注意费用
2. **模型选择**: 建议使用`gpt-4.1-nano`以保持一致性
3. **数据多样性**: 确保生成的测试用例覆盖各种情况
4. **评判一致性**: LLM Judge的prompt需要精心设计以保证评判公平

## Part-4要求

根据MP2 Part-4的要求，你需要：
1. 实现测试数据集生成（6个类别）
2. 实现LLM-as-a-Judge评估系统
3. 运行完整评估并生成报告
4. 在PDF报告中包含评估结果和分析
