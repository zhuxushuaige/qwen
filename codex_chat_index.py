"""Streaming Codex JSONL reader, preserving message text without summarization."""
import hashlib
import json
import time
from pathlib import Path

VERSION = 'codex-message-v1'

def boundary(file, offset):
    with file.open('rb') as stream:
        head = stream.read(min(offset,256))
        stream.seek(max(0,offset-256))
        tail = stream.read(min(offset,256))
    return hashlib.sha256(head+tail).hexdigest()

def extract(event):
    payload = event.get('payload',{})
    if event.get('type') != 'response_item' or payload.get('type') != 'message':
        return None
    role = payload.get('role')
    if role not in {'user','assistant'}:
        return None
    texts = [part.get('text','') for part in payload.get('content',[])
             if isinstance(part,dict) and part.get('type') in {'input_text','output_text','text'}]
    # Environment snapshots are metadata inserted into user-role records.
    texts = [text for text in texts if not text.lstrip().startswith(('<environment_context>', '<permissions instructions>', '<INSTRUCTIONS>', '# AGENTS.md instructions'))]
    if not any(text.strip() for text in texts):
        return None
    return role, str(event.get('timestamp','')), '\n\n'.join(texts)

def index_chat(documents, input_path):
    from local_documents import terms, source_time
    file = Path(input_path).expanduser().resolve(strict=True)
    if not file.is_file() or file.suffix.lower() != '.jsonl':
        raise ValueError('请选择原始 .jsonl 聊天文件。')
    root, relative = str(file.parent), file.name
    stat = file.stat()
    started = time.perf_counter()
    with documents.connect() as db:
        db.execute('''CREATE TABLE IF NOT EXISTS chat_checkpoints(
            root TEXT,path TEXT,offset INTEGER,line INTEGER,messages INTEGER,ignored INTEGER,
            malformed INTEGER,size INTEGER,mtime INTEGER,fingerprint TEXT,version TEXT,
            PRIMARY KEY(root,path))''')
        old = db.execute('SELECT * FROM chat_checkpoints WHERE root=? AND path=?',(root,relative)).fetchone()
        reusable = bool(old and old['version']==VERSION and stat.st_size>=old['offset']
                        and boundary(file,old['offset'])==old['fingerprint']
                        and (stat.st_size>old['size'] or stat.st_mtime_ns==old['mtime']))
        unchanged = bool(reusable and old['size']==stat.st_size and old['mtime']==stat.st_mtime_ns)
        offset = old['offset'] if reusable else 0
        line_number = old['line'] if reusable else 0
        messages = old['messages'] if reusable else 0
        ignored = old['ignored'] if reusable else 0
        malformed = old['malformed'] if reusable else 0
        if not reusable:
            db.execute('DELETE FROM passage_fts WHERE rowid IN (SELECT id FROM passages WHERE root=? AND path=?)',(root,relative))
            db.execute('DELETE FROM passages WHERE root=? AND path=?',(root,relative))
        ordinal = db.execute('SELECT COALESCE(MAX(ordinal),-1)+1 FROM passages WHERE root=? AND path=?',(root,relative)).fetchone()[0]
        new_messages = 0
        if not unchanged:
            with file.open('rb') as stream:
                stream.seek(offset)
                while stream.tell()<stat.st_size:
                    begin=stream.tell()
                    raw=stream.readline()
                    if not raw.endswith(b'\n') or stream.tell()>stat.st_size:
                        stream.seek(begin)
                        break
                    line_number+=1
                    offset=stream.tell()
                    try:
                        event=json.loads(raw)
                    except (ValueError,UnicodeError):
                        malformed+=1
                        continue
                    if not isinstance(event,dict):
                        ignored+=1
                        continue
                    message=extract(event)
                    if not message:
                        ignored+=1
                        continue
                    role,timestamp,text=message
                    messages+=1
                    new_messages+=1
                    for start in range(0,len(text),600):
                        content=f'角色：{role}；时间：{timestamp}；JSONL原始第{line_number}行\n'+text[start:start+720]
                        cursor=db.execute('INSERT INTO passages(root,path,page,line,ordinal,content,size,mtime) VALUES(?,?,?,?,?,?,?,?)',
                                          (root,relative,None,line_number,ordinal,content,stat.st_size,stat.st_mtime_ns))
                        db.execute('UPDATE passages SET event_time=? WHERE id=?',(source_time(relative,content),cursor.lastrowid))
                        db.execute('INSERT INTO passage_fts(rowid,tokens) VALUES(?,?)',(cursor.lastrowid,' '.join(terms(content))))
                        ordinal+=1
        note=f'原始对话消息{messages}条；跳过工具/运行事件{ignored}条；不是摘要；按原文件行号定位'
        db.execute('INSERT OR REPLACE INTO files VALUES(?,?,?,?,?,?)',(root,relative,file.name,stat.st_size,stat.st_mtime_ns,note))
        db.execute('UPDATE passages SET size=?,mtime=? WHERE root=? AND path=?',(stat.st_size,stat.st_mtime_ns,root,relative))
        db.execute('INSERT OR REPLACE INTO chat_checkpoints VALUES(?,?,?,?,?,?,?,?,?,?,?)',
                   (root,relative,offset,line_number,messages,ignored,malformed,stat.st_size,stat.st_mtime_ns,boundary(file,offset),VERSION))
        total=db.execute('SELECT COUNT(*) FROM passages WHERE root=?',(root,)).fetchone()[0]
        count=db.execute('SELECT COUNT(*) FROM files WHERE root=?',(root,)).fetchone()[0]
        warnings=[]
        if malformed:
            warnings.append(f'{relative}：{malformed}条损坏JSON事件未索引')
        if offset<stat.st_size:
            warnings.append(f'{relative}：末尾未完成的JSON行将于下次更新读取')
        summary={'root':root,'files':count,'passages':total,'warnings':warnings,'skipped':[],
                 'limit_hit':False,'seconds':round(time.perf_counter()-started,3),
                 'indexed_at':time.strftime('%Y-%m-%d %H:%M:%S'), 'chat_file':relative,
                 'messages':messages,'new_messages':new_messages,'ignored_events':ignored,
                 'cache_hit':unchanged,'incremental':reusable and not unchanged,
                 'bytes_read':0 if unchanged else offset-(old['offset'] if reusable else 0)}
        db.execute('INSERT OR REPLACE INTO roots VALUES(?,?)',(root,json.dumps(summary,ensure_ascii=False)))
    return summary
