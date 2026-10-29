"""Offline checks using temporary documents and memory, without a model server."""
import gc
import json
import tempfile
from pathlib import Path
from datetime import datetime, timezone, timedelta
from local_documents import LocalDocuments
from qwen_records_bridge import IndexedEvidenceStore
from qwen_memory_bridge import MemoryBridge
from qwen_agent_live import LiveTools

def check():
    with tempfile.TemporaryDirectory() as temp:
        base=Path(temp); root=base/'docs';root.mkdir()
        file=root/'chat.jsonl'
        def event(day,text):
            return {'type':'response_item','timestamp':day+'T12:00:00+08:00','payload':{'type':'message','role':'assistant','content':[{'type':'output_text','text':text}]}}
        file.write_text('\n'.join(json.dumps(x,ensure_ascii=False) for x in [event('2026-10-27','实验A完成，耗时10秒。'),event('2026-10-29','实验B完成，耗时8秒。')])+'\n',encoding='utf-8')
        docs=LocalDocuments(base/'documents.sqlite3');docs.index(file)
        records=IndexedEvidenceStore(root,database=base/'documents.sqlite3')
        catalog=records.execute('list_records',{'kind':'recent'})
        assert catalog['success'] and catalog['total']==2
        assert records.execute('read_records',{'ids':[catalog['records'][0]['id']]})['success']
        assert not records.execute('read_records',{'ids':['D999']})['success']
        assert not records.execute('search_records',{'query':'实验','root':'/'})['success']
        stamp=datetime(2026,10,29,tzinfo=timezone(timedelta(hours=8))).timestamp()
        scoped=IndexedEvidenceStore(root,day_range=(stamp,stamp+86400),database=base/'documents.sqlite3')
        assert scoped.execute('list_records',{})['total']==1
        memory=MemoryBridge(base/'memory.sqlite3')
        tools=LiveTools(records,memory,'offline')
        tools.current_question='请解释长期记忆'
        assert not tools.execute('save_memory',{})['success']
        tools.current_question='记住：测试项目代号青竹'
        saved=tools.execute('save_memory',{});assert saved['success']
        mid=saved['memory']['id']
        other=MemoryBridge(base/'memory.sqlite3')
        assert any('青竹' in item['content'] for item in other.context('项目代号'))
        tools.current_question='请忘记刚才保存的那条记忆'
        assert tools.execute('forget_memory',{'id':mid})['success']
        assert not other.list_active()
        del tools,memory,other;gc.collect()
    print('PASS: indexed history, date isolation, read restrictions, explicit memory write, persistence and soft forget')
if __name__=='__main__':check()
