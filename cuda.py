from langchain.llms import HuggingFacePipeline
from transformers import LlamaForCausalLM, LlamaTokenizer, pipeline
from langchain.agents import initialize_agent, AgentType, load_tools
from langchain.tools import Tool
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

cb_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# model_path = "../llama-65b-hf"
model_path = '../wizard-vicuna-13b/'
# model_path = "../llama-7b-hf"
tokenizer = LlamaTokenizer.from_pretrained(model_path)
model = LlamaForCausalLM.from_pretrained(model_path, offload_folder=model_path, load_in_8bit=False, device_map="auto")

pipe = pipeline(
     task="text-generation", model=model, tokenizer=tokenizer, max_new_tokens=64, model_kwargs={})
llm = HuggingFacePipeline(pipeline=pipe, verbose=True)

tool_names = [
            "python_repl" 
            #   ,"llm-math"
              ]
tools = load_tools(tool_names=tool_names, llm=llm)

def search_metric(keyword: str) -> str:
    """"根据关键字查询相关指标的元信息"""
    return f"指标\"{keyword}\"的元信息如下：指标code为0x1234，指标名称为{keyword}"

def send2Freezer(something: str) -> str:
    """把物品放入冰箱的步骤"""
    return f"把{something}放入冰箱分3步:\n1.把冰箱门打开\n2.把东西放进去\n3.把冰箱门关上\n\n"

tools.append(Tool.from_function(func=search_metric, name="msearch", description="根据关键字查询相关指标的元信息，输入查询关键字，输出的元信息包含指标code和指标名称"))
tools.append(Tool.from_function(func=send2Freezer, name="send2Freezer", description="把物品放入冰箱的步骤"))

agent = initialize_agent(tools, llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, callback_manager=cb_manager)

while True:
    question = input("You:")
    # print(llm(question))
    result = agent.run(question)
    # result =llmchain.run(question)
    print(result)
    print('\n\n以上\n\n')
# response, _ = model.chat(tokenizer, text)
# print(response)
