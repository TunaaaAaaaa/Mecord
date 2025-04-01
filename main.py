import streamlit as st
import requests
import json
import os
from dotenv import load_dotenv
from knowledge_base import KnowledgeBase

# 加载环境变量
load_dotenv()

# 获取API密钥和URL
API_KEY = os.getenv("API_KEY")

API_URL = os.getenv("API_URL")

# 设置页面配置
st.set_page_config(page_title="情绪分析助手", page_icon="😊")

# 初始化会话状态
if "messages" not in st.session_state:
    st.session_state.messages = []

# 初始化知识库
if "knowledge_base" not in st.session_state:
    st.session_state.knowledge_base = KnowledgeBase(API_KEY)
    st.session_state.knowledge_base.initialize_vector_store()

# 分析情绪函数
def analyze_emotion(text):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    payload = {
        "model": "deepseek-ai/DeepSeek-V3",
        "messages": [
            {"role": "system", "content": "你是一个情绪分析专家。请分析用户输入的情绪，并从以下选项中选择一个最匹配的情绪类别：开心、悲伤、愤怒、焦虑、困惑、中性。只需回复情绪类别，不要添加其他内容。"}, 
            {"role": "user", "content": text}
        ],
        "temperature": 0.3
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()  # 检查请求是否成功
        result = response.json()
        emotion = result["choices"][0]["message"]["content"].strip()
        return emotion
    except Exception as e:
        st.error(f"分析情绪时出错: {str(e)}")
        return "中性"  # 出错时返回默认情绪

# 生成回应函数
def generate_response(text, emotion):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    # 从知识库中检索相关内容
    relevant_docs = st.session_state.knowledge_base.search_knowledge(text)
    knowledge_context = "\n".join(relevant_docs) if relevant_docs else ""
    
    prompt = f"用户说: '{text}'\n用户的情绪似乎是: {emotion}\n"
    if knowledge_context:
        prompt += f"根据知识库中的相关信息：\n{knowledge_context}\n"
    prompt += f"请告知用户他的情绪类型为{emotion}，并结合知识库信息（如果有）给出适合这种情绪的回应，提供安慰、建议或积极的反馈。"
    
    payload = {
        "model": "deepseek-ai/DeepSeek-V3",
        "messages": [
            {"role": "system", "content": "你是一个善解人意的心理咨询师，擅长根据用户的情绪状态和相关知识提供恰当的回应和建议。"}, 
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()  # 检查请求是否成功
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        st.error(f"生成回应时出错: {str(e)}")
        return "抱歉，我现在无法提供回应。请稍后再试。"  # 出错时返回默认回应

# 侧边栏
with st.sidebar:
    st.title("情绪分析助手")
    st.markdown("这是一个能够分析你情绪并给出回应的AI助手。")
    
    # 文件上传
    uploaded_file = st.file_uploader("上传知识库文档", type=["txt", "md", "pdf"])
    if uploaded_file:
        # 保存上传的文件
        file_path = os.path.join("knowledge_docs", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        # 更新知识库
        if st.session_state.knowledge_base.add_document(file_path):
            st.success(f"文件 {uploaded_file.name} 已添加到知识库")
        else:
            st.error("添加文件失败")
    
    if st.button("清除对话历史"):
        st.session_state.messages = []
        st.experimental_rerun()

# 主界面
st.title("情绪分析助手 🧠💭")

# 显示聊天历史
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "emotion" in message:
            st.caption(f"检测到的情绪: {message['emotion']}")

# 用户输入
if prompt := st.chat_input("请输入你想表达的内容..."):
    # 添加用户消息到历史
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # 显示用户消息
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # 显示助手思考中
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("思考中...")
        
        # 分析情绪
        emotion = analyze_emotion(prompt)
        
        # 生成回应
        response = generate_response(prompt, emotion)
        
        # 更新消息
        message_placeholder.markdown(response)
    
    # 添加助手回应到历史
    st.session_state.messages.append({"role": "assistant", "content": response, "emotion": emotion})