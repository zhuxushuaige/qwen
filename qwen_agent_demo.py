"""Qwen-Agent interface with indexed history and original memory management."""
from datetime import datetime, timedelta, timezone, time
from pathlib import Path
import streamlit as st
from local_documents import LocalDocuments
from qwen_records_bridge import DATABASE
from qwen_memory_bridge import MemoryBridge
from qwen_agent_live import QwenLiveAssistant

st.set_page_config(page_title='科研助手 · Qwen-Agent', page_icon='🔎')
st.title('科研助手 · Qwen-Agent')
st.caption('Qwen3.6-27B · 历史资料检索 · 原项目记忆管理')
docs = LocalDocuments(DATABASE)
with docs.connect() as db:
    roots = [r['root'] for r in db.execute('SELECT root FROM roots') if Path(r['root']).exists()]
with st.sidebar:
    st.subheader('导入本地资料')
    with st.form('import_documents'):
        import_path = st.text_input('资料目录或原生JSONL文件路径')
        do_import = st.form_submit_button('建立资料索引')
    if do_import:
        if not import_path.strip():
            st.error('请填写资料路径。')
        else:
            try:
                with st.spinner('正在建立本机索引…'):
                    docs.index(import_path.strip())
                st.session_state.pop('qwen_live_config', None)
                st.rerun()
            except Exception as exc:
                st.error(f'索引失败：{exc}')
if not roots:
    st.info('从左侧导入资料后开始使用。'); st.stop()
if 'qwen_memory_bridge' not in st.session_state:
    st.session_state.qwen_memory_bridge = MemoryBridge()
memory = st.session_state.qwen_memory_bridge
with st.sidebar:
    st.subheader('资料范围')
    latest = docs.latest_root()
    root = st.selectbox('已导入资料目录', roots, index=roots.index(latest) if latest in roots else 0)
    options = ['全部已索引文件']+[f['path'] for f in docs.files(root)]
    selection = st.selectbox('限定文件', options)
    file = None if selection == options[0] else selection
    choice = st.radio('时间范围', ['全部历史', '今天', '指定日期'])
    period = None
    if choice != '全部历史':
        day = datetime.now(timezone(timedelta(hours=8))).date()
        if choice == '指定日期': day = st.date_input('日期（北京时间）', value=day)
        start = datetime.combine(day, time.min, tzinfo=timezone(timedelta(hours=8))).timestamp()
        period = (start, start+86400)
    if st.button('更新当前资料索引'):
        target = file or (docs.summary(root) or {}).get('chat_file')
        with st.spinner('正在增量更新…'):
            docs.index(str(Path(root)/target) if target else root)
        st.session_state.pop('qwen_live_config', None)
        st.success('索引已更新')
    if st.button('新建会话'): st.session_state.pop('qwen_live_config', None)
    st.caption('新建会话清空对话，保留已保存记忆。相关摘录发送到已配置的模型服务推理。')
config = (root, file, period, 'history-memory-v1')
if st.session_state.get('qwen_live_config') != config:
    st.session_state.qwen_live_bot = QwenLiveAssistant(root, file, period, memory=memory)
    st.session_state.qwen_live_config = config
    st.session_state.qwen_visible_turns = []
bot = st.session_state.qwen_live_bot
with st.sidebar:
    st.subheader('记忆管理')
    with st.form('qwen_save_memory', clear_on_submit=True):
        text = st.text_area('指定保存的内容', placeholder='例如：我喜欢简洁的中文回答')
        save = st.form_submit_button('保存记忆')
    if save:
        result = memory.save('记住：'+text, bot.session) if text.strip() else {'success':False,'message':'请输入内容'}
        if result['success']: st.success('记忆已保存。')
        else: st.error(result.get('message','保存失败'))
    active = memory.list_active()
    with st.expander(f'查看 / 忘记记忆（{len(active)}条）'):
        if active:
            labels = {m['id']:f"{m['id']} · {m['content'][:80]}" for m in active}
            target = st.selectbox('选择记忆',list(labels),format_func=labels.get)
            st.write(next(m['content'] for m in active if m['id']==target))
            if st.button('忘记所选记忆'):
                result = memory.forget(target)
                if result['success']:
                    bot.tools.memory_sources.pop(target,None)
                    st.rerun()
                else: st.error(result.get('message','遗忘失败'))
        else: st.caption('暂无有效记忆。')
    st.caption('也可在聊天中说“记住：……”。普通聊天不会自动保存。')
scope = bot.store._scope()
st.caption('当前范围：'+('全部已索引历史' if scope['day_range'] is None else datetime.fromtimestamp(scope['day_range'][0],timezone(timedelta(hours=8))).strftime('%Y-%m-%d')+' 起（北京时间）'))
for turn in st.session_state.qwen_visible_turns:
    with st.chat_message('user'): st.write(turn['question'])
    with st.chat_message('assistant'):
        st.markdown(turn['answer'])
        st.caption(f"耗时 {turn['seconds']} 秒 · 工具 {len(turn['tools'])} 次 · 记忆上下文 {len(turn['memories'])} 条")
        for warning in turn['warnings']: st.warning(warning)
        with st.expander('本轮工具、记忆和引用原文'):
            st.json(turn['tools']); st.json(turn['memories'])
            for source in turn['sources']:
                st.write(f"[{source['id']}] {source['source']}")
                st.text(source['content'])
question = st.chat_input('问历史资料，或说“记住：……”')
if question:
    with st.spinner('正在检索、读取和回答…'):
        try: result = bot.ask(question)
        except Exception as exc:
            st.error(f'本轮失败：{type(exc).__name__}: {exc}'); st.stop()
    st.session_state.qwen_visible_turns.append(result)
    st.rerun()
