import streamlit as st
import time
import re
from datetime import datetime
from openai import OpenAI
from crewai import Agent, Task, Crew, LLM
from config import API_KEY, BASE_URL, MODEL_NAME

# ==================== 初始化 ====================
client = OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=60.0)

custom_llm = LLM(
    model=MODEL_NAME,
    base_url=BASE_URL,
    api_key=API_KEY,
    temperature=0,
    max_tokens=250,
    custom_openai=True
)

report_llm = LLM(
    model=MODEL_NAME,
    base_url=BASE_URL,
    api_key=API_KEY,
    temperature=0,
    max_tokens=800,
    custom_openai=True
)

st.set_page_config(page_title="智能预问诊健康助手", page_icon="🩺", layout="centered")
st.title("🩺 智能预问诊健康助手")
st.caption("多轮问诊 → 智能诊断 → 安全分级 → 结构化报告")

# ==================== 状态管理 ====================
defaults = {
    "messages": [], "interview_done": False, "case_summary": "",
    "question_count": 0, "diagnosis_done": False, "final_disease": "",
    "safety_raw": "", "safety_level": "未知", "paid": False, "report_text": ""
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ==================== 问诊提示词 ====================
INTERVIEWER_PROMPT = """
你是一位耐心的全科医生，正在通过聊天进行预问诊。
任务：通过多轮提问收集：主诉、症状细节、既往史、用药史。
规则：
- 每次只问1-2个问题。
- 最多3轮对话，之后必须输出"[信息收集完毕]"。
- 语气温和专业，回复简短（不超过80字）。
"""

# ==================== 紧急关键词 ====================
EMERGENCY_KEYWORDS = ["胸痛", "胸闷", "呼吸困难", "意识模糊", "大出血", "昏迷", "抽搐", "咯血", "呕血"]


def check_emergency(text):
    for kw in EMERGENCY_KEYWORDS:
        if kw in text:
            return kw
    return None


# ==================== 展示历史聊天记录 ====================
for msg in st.session_state.messages:
    avatar = "🧑" if msg["role"] == "user" else "🩺"
    with st.chat_message(msg["role"], avatar=avatar):
        st.write(msg["content"])


# ==================== 统一的输入处理函数 ====================
def handle_input(user_text):
    if not user_text.strip():
        return

    emergency = check_emergency(user_text)
    if emergency:
        st.error(f"🚨 检测到紧急症状：**{emergency}** —— 建议立即拨打120或前往急诊！")

    st.session_state.messages.append({"role": "user", "content": user_text})

    with st.spinner("医生正在思考..."):
        try:
            messages_for_llm = [{"role": "system", "content": INTERVIEWER_PROMPT}]
            for m in st.session_state.messages[-4:]:
                messages_for_llm.append(m)

            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages_for_llm,
                temperature=0.3,
                max_tokens=150
            )
            reply = response.choices[0].message.content
        except Exception as e:
            reply = f"抱歉，医生暂时无法回复。错误信息：{e}"

    st.session_state.messages.append({"role": "assistant", "content": reply})

    st.session_state.question_count += 1
    if "[信息收集完毕]" in reply or st.session_state.question_count >= 3:
        st.session_state.interview_done = True
        st.session_state.case_summary = "\n".join(
            [f"{m['role']}: {m['content']}" for m in st.session_state.messages]
        )
    st.rerun()


# ==================== 问诊交互区 ====================
if not st.session_state.interview_done:
    if len(st.session_state.messages) == 0:
        st.markdown("**💡 快速选择常见症状：**")
        cols = st.columns(4)
        for i, symptom in enumerate(["胸痛", "头痛", "发热", "腹痛"]):
            with cols[i]:
                if st.button(symptom, use_container_width=True, key=f"quick_{symptom}"):
                    handle_input(f"我最近{symptom}")

    if prompt := st.chat_input("请描述您的不适..."):
        handle_input(prompt)

# ==================== 诊断阶段 ====================
else:
    st.success("✅ 信息收集完毕！")

    if not st.session_state.diagnosis_done:
        if st.button("🚀 开始AI专家会诊（深度诊断）", use_container_width=True):
            with st.spinner("智能体正在诊断，请稍候..."):
                # 诊断智能体
                diagnostician = Agent(
                    role="内科诊断医生",
                    goal="根据临床病例给出最可能的疾病名称及判断依据",
                    backstory="你是一位经验丰富的内科医生。",
                    llm=custom_llm, verbose=False
                )
                # 安全分级专家（改为客观分级视角）
                safety_agent = Agent(
                    role="临床安全分级专家",
                    goal="根据病例中实际存在的症状和体征，客观判断安全等级，既不漏诊也不过度紧张",
                    backstory="你是一位经验丰富的临床医生，擅长根据患者症状的严重程度进行分级。你清楚：绝大多数就诊情况是轻症，真正需要急诊的只占少数。你的判断基于客观症状，而不是疾病名称的严重程度。",
                    llm=custom_llm, verbose=False
                )

                # 诊断任务
                task_diagnose = Task(
                    description=f"""根据以下病例给出诊断。
病例：{st.session_state.case_summary}
【强制要求】
1. 用1-2句话说明判断依据。
2. 给出最可能的1个疾病名称。
3. 最后一行必须以"诊断结论：XXX"结尾。""",
                    expected_output="判断依据 + '诊断结论：疾病名称'",
                    agent=diagnostician
                )

                try:
                    # 跑诊断
                    crew_diag = Crew(agents=[diagnostician], tasks=[task_diagnose], verbose=False)
                    crew_diag.kickoff()
                    diag_raw = task_diagnose.output.raw if task_diagnose.output else ""

                    match = re.search(r'诊断结论\s*[:：]\s*(.+)', diag_raw)
                    final_disease = match.group(1).strip().split('\n')[0] if match else diag_raw.strip().split('\n')[
                                                                                            -1][:50]

                    # 安全分级任务（带明确分级标准）
                    task_safety = Task(
                        description=f"""对以下病例进行安全分级。

病例：
{st.session_state.case_summary}

诊断结论：{final_disease}

【分级标准】请严格对照以下标准判断，不要因为疾病名称本身严重就升级等级，要看病例中实际描述的症状：

**高危（需要立即急诊）** —— 必须同时满足以下条件之一：
- 出现危及生命的急症表现：剧烈胸痛伴大汗/濒死感、意识丧失/昏迷、严重呼吸困难（不能平卧）、大出血（呕血/咯血/便血）、持续抽搐
- 生命体征明显异常：血压极低、心率极快或极慢、血氧饱和度低于90%
- 症状突然发生且快速加重

**中危（建议1-2天内就医）** —— 满足以下条件之一：
- 症状持续超过3天且未见好转
- 有明显疼痛或不适，影响正常生活
- 有慢性病基础（高血压、糖尿病）且症状可能与之相关
- 医生诊断为某种需要治疗的疾病，但不属于急症

**低危（可居家观察或择期就医）** —— 满足以下条件：
- 症状轻微，不影响日常生活
- 症状近期出现，有自限性可能（如轻微感冒、偶发头痛）
- 无基础疾病，无危险信号
- 诊断结论为自限性疾病或轻症

【重要判定原则】
1. 病例中**没有明确描述**的危险症状，不要脑补添加。
2. 疾病名称本身严重（如"心肌梗死"）不等于当前病情是高危。要看病例描述的症状是否支持。
3. 如果病例信息不足以判断，输出"中危"而不是"高危"，并说明信息不足。
4. 统计上，90%的线上问诊病例都属于中危或低危。如果你发现自己在连续多次判断中都输出"高危"，说明你的判断标准过于严苛，请重新审视病例中是否有真正危及生命的客观证据。

【强制输出格式】
第一行：安全等级：[高危/中危/低危]
第二行：判断依据：[基于病例中的哪些具体症状]
第三行：建议：[具体行动建议]

【示例1】
病例：患者女，25岁，轻微头痛1天，无发热，无其他不适。
安全等级：低危
判断依据：症状轻微，持续时间短，无危险信号。
建议：注意休息，如症状加重或持续超过3天，建议就医。

【示例2】
病例：患者男，55岁，突发剧烈胸痛2小时，伴大汗、濒死感。
安全等级：高危
判断依据：剧烈胸痛+大汗+濒死感，符合急性冠脉综合征表现。
建议：立即拨打120，前往急诊。""",
                        expected_output="安全等级 + 判断依据 + 建议",
                        agent=safety_agent
                    )
                    crew_safety = Crew(agents=[safety_agent], tasks=[task_safety], verbose=False)
                    crew_safety.kickoff()
                    safety_raw = task_safety.output.raw if task_safety.output else ""

                    safety_match = re.search(r'安全等级\s*[:：]\s*\[?(高危|中危|低危)\]?', safety_raw)
                    safety_level = safety_match.group(1) if safety_match else "未知"

                    st.session_state.final_disease = final_disease
                    st.session_state.safety_raw = safety_raw
                    st.session_state.safety_level = safety_level
                    st.session_state.diagnosis_done = True
                    st.rerun()

                except Exception as e:
                    st.error(f"诊断失败：{e}")

    # ==================== 结果展示 ====================
    if st.session_state.diagnosis_done:
        st.markdown("---")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("### 📋 诊断结论")
            st.success(f"**{st.session_state.final_disease}**")
        with col2:
            st.markdown("### 🛡️ 安全等级")
            if st.session_state.safety_level == "高危":
                st.error("🚨 高危")
            elif st.session_state.safety_level == "中危":
                st.warning("⚠️ 中危")
            elif st.session_state.safety_level == "低危":
                st.info("✅ 低危")
            else:
                st.info("❓ 未知")

        # 分级提示
        if st.session_state.safety_level == "高危":
            st.error("🚨 **请立即前往急诊或拨打120！**")
        elif st.session_state.safety_level == "中危":
            st.warning("⚠️ **建议尽快前往医院就诊，不要拖延。**")
        elif st.session_state.safety_level == "低危":
            st.info("✅ **可居家观察，如症状加重再就医。**")

        # 显示安全审查详情
        with st.expander("查看安全分级详细依据"):
            st.markdown(st.session_state.safety_raw)

        # 付费解锁
        if not st.session_state.paid:
            st.markdown("---")
            st.markdown("### 🔒 解锁完整健康报告")
            st.markdown("包含：判断依据、建议检查、生活方式建议、危险信号等")
            if st.button("💳 解锁完整报告（9.9元）", use_container_width=True):
                st.session_state.paid = True
                st.rerun()
        else:
            if not st.session_state.report_text:
                with st.spinner("正在生成健康报告..."):
                    report_writer = Agent(
                        role="健康报告撰写专家",
                        goal="生成结构化健康报告",
                        backstory="你是一位擅长医学科普的医生。",
                        llm=report_llm, verbose=False
                    )
                    task_report = Task(
                        description=f"""生成健康报告。
病例：{st.session_state.case_summary}
诊断：{st.session_state.final_disease}
安全审查：{st.session_state.safety_raw}

严格按以下Markdown输出：
# 健康咨询报告
## 一、患者基本信息
- 主诉：[提取]
- 症状摘要：[提取]
## 二、初步诊断
**最可能的疾病：** [疾病名称]
## 三、判断依据
[基于症状推理]
## 四、安全等级
**等级：** [高危/中危/低危]
## 五、建议检查
- [项目1]
- [项目2]
## 六、生活方式建议
- [建议1]
- [建议2]
## 七、危险信号
- [信号1]
- [信号2]
## 八、免责声明
本报告仅供健康科普参考，不构成医疗诊断。""",
                        expected_output="Markdown健康报告",
                        agent=report_writer
                    )
                    crew_report = Crew(agents=[report_writer], tasks=[task_report], verbose=False)
                    crew_report.kickoff()
                    st.session_state.report_text = task_report.output.raw if task_report.output else "报告生成失败"
                    st.rerun()

            st.markdown("---")
            st.markdown("### 📄 完整健康报告")
            st.markdown(st.session_state.report_text)

            st.download_button(
                label="⬇️ 下载报告",
                data=st.session_state.report_text,
                file_name=f"健康报告_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                mime="text/markdown",
                use_container_width=True
            )

        st.warning("⚠️ 本建议仅供参考，请务必前往正规医院就诊。")

# ==================== 重置按钮 ====================
if st.button("🔄 重新开始问诊", use_container_width=True):
    for key, val in defaults.items():
        st.session_state[key] = val
    st.rerun()