from typing import List, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
import os

_ = load_dotenv(find_dotenv())


class Chain:
    def __init__(self, client: OpenAI):
        self.client = client

    def run(self, chain_config: List[Dict[str, Any]], initial_input: str = "") -> str:
        """
        执行 chain 工作流
        
        Args:
            chain_config: 配置列表，每个配置包含 system, prompt, model 和 temperature
            initial_input: 初始输入文本
        
        Returns:
            最后一个 LLM 的输出结果
        """
        current_output = initial_input

        for step in chain_config:
            # 获取配置参数
            system = step.get('system', '')
            prompt_template = step.get('prompt', '')
            model = step.get('model', 'gpt-4o')
            temperature = step.get('temperature', 0.7)

            # 将上一步的输出插入到当前步骤的提示中
            current_prompt = prompt_template.format(input=current_output)

            # 调用 OpenAI API
            response = self.client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": current_prompt}
                ]
            )

            # 获取输出并更新 current_output
            current_output = response.choices[0].message.content
            # 
            print("step!")

        return current_output

# 创建 Chain 实例
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)
chain = Chain(client)


# 配置 chain 工作流
chain_config = [
    {
        "system": """你是一位文本大纲生成专家，擅长根据用户的需求创建一个有条理且易于扩展成完整文章的大纲，
                    你拥有强大的主题分析能力，能准确提取关键信息和核心要点。具备丰富的文案写作知识储备，熟悉各种文体和题材的文案大纲构建方法。
                    可根据不同的主题需求，如商业文案、文学创作、学术论文等，生成具有针对性、逻辑性和条理性的文案大纲，并且能确保大纲结构合理、逻辑通顺。
                    该大纲应该包含以下部分：引言：介绍主题背景，阐述撰写目的，并吸引读者兴趣。
                    主体部分：第一段落：详细说明第一个关键点或论据，支持观点并引用相关数据或案例。
                    第二段落：深入探讨第二个重点，继续论证或展开叙述，保持内容的连贯性和深度。
                    第三段落：如果有必要，进一步讨论其他重要方面，或者提供不同的视角和证据。
                    结论：总结所有要点，重申主要观点，并给出有力的结尾陈述，可以是呼吁行动、提出展望或其他形式的收尾。创意性标题：为文章构思一个引人注目的标题，确保它既反映了文章的核心内容又能激发读者的好奇心。 
                  """,
        "prompt": "{input}",
        "model": "deepseek-r1",
        "temperature": 0.5
    },
    {
        "system": """
        ### 角色设定
        你是一位专业的内容开发专家，擅长将结构化大纲转化为逻辑严谨、内容充实的完整文章。你精通不同文体的表达规范，能够根据目标读者的认知水平调整内容深度，并擅长运用案例增强说服力
        ### 任务说明
        请基于以下大纲进行专业级内容扩写，要求：
        1. 结构优化：检查各级标题的逻辑递进关系，必要时调整框架顺序
        2. 内容深化：为每个核心论点补充3种支持要素（数据/案例/权威引述）
        3. 衔接处理：在段落间添加过渡句，确保内容自然流畅
        4. 风格适配：根据[商业文案/学术论文/新媒体文章]的文体特征调整语言风格
        5. 互动设计：在适当位置设置读者互动点（提问/场景化描述/悬念设置）
        ### 输出要求
        1. 段落内的主题句+支撑内容+小结句结构
        2. 关键数据的来源标注（格式：[数据来源]）
        3. 专业术语的通俗化解释框
        4. 可视化元素的插入建议（图表/信息图位置标记）
        """,
        "prompt": "{input}",
        "model": "claude-3-5-sonnet-20240620",
        "temperature": 0.6
    },
    # {
    #     "system": """
    #     # 角色设定
    #     你是一位资深文案外科医生，拥有10年以上专业编辑经验。
    #     精通语法规范、逻辑校验和说服力增强技巧，擅长通过精准的文字调整提升内容传播效能
    #     # 润色维度
    #     1. 逻辑校验：检测论点-论据链条的完整性，标记需要强化的薄弱环节
    #     2. 语言优化：消除冗余表达（重复率控制在<5%）；专业术语与通俗表达的平衡调整；句式多样性优化（长短句比例3:7）；情感词密度检测（每千字15-20个情感锚点）
    #     3. 说服力增强：检查SCQA故事模型的应用情况；评估FAB法则的运用效果；验证AIDA公式的达成度
    #     4. 可读性提升：段落长度控制（移动端阅读适配）；重点信息的视觉强调方案；复杂信息的拆解建议
    #     # 输出模板
    #     采用三栏对照形式呈现：
    #     """,
    #     "prompt": "{input}",
    #     "model": "claude-3-5-sonnet-20240620",
    #     "temperature": 0.7
    # }
]

# 运行 chain
result = chain.run(chain_config, "以孤独的夜行者为题写一篇750字的散文，描绘一个人在城市中夜晚漫无目的行走的心情与所见所感，以及夜的寂静给予的独特感悟。")
print(result)