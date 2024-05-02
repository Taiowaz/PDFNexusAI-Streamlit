from db_api.api.api_pinecone import query
from db_api.models import ChatSession, Message, VectorBaseInfo
from db_api.api.api_embedding import get_single_embedding
from db_api.api.api_qwen import call_stream_with_messages, call_qwen
import db_api.sqlite as db


# 获取所有会话名
def get_all_chat_session():
    chat_sessions = db.get_all_chat_session()
    chat_session_names = [chat_session.name for chat_session in chat_sessions]

    return chat_session_names

# 利用知识库构建带有提示词内容


def query_integrate_content(content: str, vectorbase_name):
    # 数字转英文
    def number_to_words(n):
        num_to_word = {
            1: 'one',
            2: 'two',
            3: 'three',
            4: 'four',
            5: 'five',
            6: 'six',
            7: 'seven',
            8: 'eight',
            9: 'nine',
            10: 'ten'
        }
        return num_to_word.get(n, 'Number out of range')

    # 获取输入文本对应向量
    content_embedding = get_single_embedding(content)
    # 获取相关文本列表
    text_list = query(vectorbase_name, content_embedding)
    prompt = ""
    i = 1
    for text in text_list:
        prompt += "This is the "+number_to_words(i)+" information:"+text + "//"
        i += 1

    # 构建输入内容
    content = f'''
    You are a helpful assistant,
    This is my question:
    {content}
    The following is relevant information.
    Please combine it with my question and decide whether to reply based on the relevant information:
    {prompt}
    '''

    return content


# 与千问对话 --保存对话时，不带提示词，输入到模型的是带提示词的内容
def talk_stream_with_qwen(chat_session_name, vectorbase_name, messages: list):
    input_content = messages[-1]['content']
    # 保存当前消息,但不带提示词存入数据库
    db.save_message_by_chat_session_name(
        chat_session_name=chat_session_name,
        message=messages[-1]
    )

    # 选定知识库，则根据知识库构建内容
    if vectorbase_name != None:
        # 判断是否需要查询知识库
        flag = is_need_query_knowledgebase(messages, vectorbase_name)
        """ 测试！！！ """
        # print("\nisQuery:"+str(flag)+"\n")
        if flag.lower() == "yes":
            input_content = query_integrate_content(
                content=input_content,
                vectorbase_name=vectorbase_name
            )
    """ 测试！！！ """
    # print("\ninput_content:"+str(input_content)+"\n")
    # 构建messages 输入到LLM的内容，带有提示词
    messages[-1]['content'] = input_content
    """ 测试！！！ """
    # print("\nmessages:"+str(messages)+"\n")
    # 向qwenAPI发送请求并流式输出
    stream_gen = call_stream_with_messages(chat_session_name, messages)

    # 返回流响应流
    return stream_gen


# 判断当前问题是否需要查询知识库
def is_need_query_knowledgebase(messages: list, vectorbase_name: str):
    # 获取当前知识库描述
    vectornase_info = db.get_vector_base_info_by_name(vectorbase_name)
    knowledge_base_detail = vectornase_info.detail
    # 获取用户输入
    current_input = messages[-1]['content']
    # 构建提示词，让模型选择是否需要查询知识库
    prompt = f'''
    Current input content: {current_input}
    Details from my current knowledge base: {knowledge_base_detail}
    Constraints:
    1. Consider our chat history, your own knowledge and my knowledge base detail to decide whether you need me to query some information from this knowledge base for you to facilitate a reply.
    2. Your answer must be either 'yes' or 'no'.
    '''
    # 将提示词更新到messages中
    messages[-1]['content'] = prompt
    # 调用qwenAPI,返回模型回复
    return call_qwen(messages)

# 创建会话


def create_chat_session():
    chat_session = ChatSession()
    chat_session_name = db.save_chat_session_return_name(chat_session)
    return chat_session_name


# 获取会话对应的对话内容


def get_messages_by_chat_session_name(chat_session_name):
    # 获取指定ID的会话的所有信息
    message_infos = db.get_messages_by_chat_session_name(chat_session_name)
    # 获取每个消息的角色与内容
    messages = [{'role': message.role, 'content': message.content}
                for message in message_infos]
    return messages

# 删除会话


def delete_chat_session_by_chat_session_name(chat_session_name):
    db.delete_chat_session(chat_session_name)

# 更新会话名字


def update_chat_session_name(chat_session_name, chat_session_new_name):
    db.update_chat_session_name(
        chat_session_name, chat_session_new_name)

# 判断会话名是否存在


def chat_session_exists(chat_session_name):
    return db.chat_session_exists(chat_session_name)
