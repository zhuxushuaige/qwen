"""Bounded local document indexing and keyword retrieval; no remote services."""
import hashlib
import json
import os
import re
import sqlite3
import time
from difflib import SequenceMatcher
from contextlib import contextmanager
from pathlib import Path
from datetime import datetime, timezone, timedelta
from evidence_quality import question_only, MEASURED, body_text

def source_time(path, content):
    if str(path).lower().endswith('.jsonl'):
        match = re.search(r'^角色：[^\n]*?；时间：([^；\n]+)', content)
        if match:
            try:
                value = datetime.fromisoformat(match[1].replace('Z','+00:00'))
                if value.tzinfo is not None:
                    return value.timestamp()
            except ValueError:
                pass
        return None
    match = re.search(r'(20\d\d)-(\d{1,2})-(\d{1,2})', Path(path).name)
    if match:
        try:
            return datetime(int(match[1]),int(match[2]),int(match[3]),tzinfo=timezone(timedelta(hours=8))).timestamp()
        except ValueError:
            pass
    return None

def query_dates(query):
    dates = []
    for match in re.finditer(r'(20\d\d)[年/-](\d{1,2})[月/-](\d{1,2})日?',query):
        try:
            date = datetime(int(match[1]),int(match[2]),int(match[3]),tzinfo=timezone(timedelta(hours=8)))
            dates.append(date.timestamp())
        except ValueError:
            pass
    return dates


def resolve_calendar_dates(text, now=None):
    current = now or datetime.now(timezone(timedelta(hours=8)))
    for word,offset in [('大前天',-3),('大后天',3),('前天',-2),('后天',2),('昨天',-1),('明天',1),('今天',0),('今日',0),('昨日',-1)]:
        text = text.replace(word,(current+timedelta(days=offset)).date().isoformat())
    return text

TEXT_EXTENSIONS = {".txt", ".md", ".markdown", ".py", ".json", ".yaml", ".yml", ".toml", ".csv", ".rst", ".sh", ".ps1", ".cpp", ".c", ".h", ".hpp", ".cc", ".log"}
EXCLUDED_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".agents", ".claude", ".codex", ".aws", ".ssh", ".codex_tmp", ".pytest_cache", ".ruff_cache", "models", "local_assistant_data", "local_assistant_ui_test"}
SENSITIVE_NAMES = {"credentials", "credentials.json", "secrets.json", "secrets.toml", "id_rsa", "id_ed25519", "token.json", "tokens.json", "service-account.json"}
QUERY_NOISE = {"根据", "资料", "赛题", "请问", "要求", "多少", "是多", "请引", "引用", "依据", "什么", "这个", "这篇", "论文", "文档", "项目", "内容", "进行", "总结", "概述", "概括", "介绍", "说明", "怎么", "如何", "一下", "帮我", "请帮"}

def terms(text):
    english = re.findall(r"[a-zA-Z0-9_]{2,}", text.lower())
    chinese = []
    for run in re.findall(r"[\u3400-\u9fff]+", text):
        chinese.extend(run[i:i+2] for i in range(len(run)-1))
        if len(run) == 1:
            chinese.append(run)
    return list(dict.fromkeys(english + chinese))[:500]

def safe_path(root, path):
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("文件已离开选定目录。")
    return resolved

class LocalDocuments:
    def __init__(self, database):
        Path(database).parent.mkdir(parents=True, exist_ok=True)
        self.database = str(database)
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS roots(root TEXT PRIMARY KEY, summary TEXT);
                CREATE TABLE IF NOT EXISTS files(root TEXT, path TEXT, name TEXT, size INTEGER, mtime INTEGER, note TEXT, PRIMARY KEY(root,path));
                CREATE TABLE IF NOT EXISTS passages(id INTEGER PRIMARY KEY, root TEXT, path TEXT, page INTEGER, line INTEGER, ordinal INTEGER, content TEXT, size INTEGER, mtime INTEGER);
                CREATE INDEX IF NOT EXISTS passage_root ON passages(root,path);
                CREATE VIRTUAL TABLE IF NOT EXISTS passage_fts USING fts5(tokens);
            """)
            if 'event_time' not in {row['name'] for row in db.execute('PRAGMA table_info(passages)')}:
                db.execute('ALTER TABLE passages ADD COLUMN event_time REAL')
            db.execute('CREATE INDEX IF NOT EXISTS passage_time ON passages(root,event_time DESC)')
            # Migrate existing indexed text without rereading the original JSONL.
            missing = db.execute('SELECT id,path,content FROM passages WHERE event_time IS NULL').fetchall()
            db.executemany('UPDATE passages SET event_time=? WHERE id=?',
                           [(stamp,row['id']) for row in missing if (stamp:=source_time(row['path'],row['content'])) is not None])

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.database, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def summary(self, root):
        with self.connect() as db:
            row = db.execute("SELECT summary FROM roots WHERE root=?", (str(Path(root).resolve()),)).fetchone()
        return json.loads(row[0]) if row else None

    def latest_root(self):
        with self.connect() as db:
            rows = [dict(row) for row in db.execute('SELECT root,summary FROM roots')]
        rows.sort(key=lambda row: json.loads(row['summary']).get('indexed_at', ''), reverse=True)
        return next((row['root'] for row in rows if Path(row['root']).is_dir()), None)

    def index(self, folder):
        root = Path(folder).expanduser().resolve(strict=True)
        if root.is_file() and root.suffix.lower() == '.jsonl':
            from codex_chat_index import index_chat
            return index_chat(self,root)
        if not root.is_dir():
            raise ValueError("请选择文件夹。")
        documents, passages, warnings, skipped = [], [], [], []
        started = time.perf_counter()
        limit_hit = False
        for parent, dirs, names in os.walk(root, followlinks=False):
            dirs[:] = sorted(d for d in dirs if not d.startswith('.') and d not in EXCLUDED_DIRS
                             and not (Path(parent)/d).is_symlink()
                             and not (hasattr(Path(parent)/d, 'is_junction') and (Path(parent)/d).is_junction()))
            for name in sorted(names):
                file = Path(parent)/name
                if name.startswith('.') or name.lower() in SENSITIVE_NAMES or file.is_symlink():
                    continue
                if file.suffix.lower() not in TEXT_EXTENSIONS | {".pdf"}:
                    skipped.append({'path': str(file.relative_to(root)), 'reason': '请在资料路径直接填写这个JSONL文件的完整路径，启用流式增量索引' if file.suffix.lower()=='.jsonl' else '暂不支持的扩展名'})
                    continue
                if len(documents) >= 600 or len(passages) >= 10000:
                    limit_hit = True
                    break
                relative = str(file.relative_to(root))
                try:
                    file = safe_path(root, file)
                    stat = file.stat()
                    if stat.st_size > (30_000_000 if file.suffix.lower() == '.pdf' else 1_000_000):
                        warnings.append(f"{relative}：超过文件大小限制，未读取")
                        continue
                    units, note = self.read(file)
                    # A file receives at most 150k text characters / 300 chunks.
                    file_chunks = []
                    characters = 0
                    for text, page, line in units:
                        for offset in range(0, len(text), 600):
                            if characters >= 150000 or len(file_chunks) >= 300:
                                note = (note + '；' if note else '') + '正文仅索引前15万字符'
                                break
                            content = text[offset:offset+720].strip()
                            if not content:
                                continue
                            file_chunks.append((str(root), relative, page,
                                                (line + text[:offset].count('\n')) if line else None,
                                                len(file_chunks), content, stat.st_size, stat.st_mtime_ns))
                            characters += len(content)
                        if characters >= 150000 or len(file_chunks) >= 300:
                            break
                    if not file_chunks:
                        warnings.append(f"{relative}：未提取到文字，扫描版 PDF 需 OCR")
                        continue
                    remaining = 10000 - len(passages)
                    if len(file_chunks) > remaining:
                        file_chunks = file_chunks[:remaining]
                        note += '；达到总段落上限，仅索引部分内容'
                    documents.append((str(root), relative, name, stat.st_size, stat.st_mtime_ns, note))
                    passages.extend(file_chunks)
                    if note:
                        warnings.append(f"{relative}：{note}")
                except Exception as exc:
                    warnings.append(f"{relative}：读取失败（{type(exc).__name__}: {exc}）")
            if limit_hit:
                break
        summary = {"root": str(root), "files": len(documents), "passages": len(passages), "skipped": skipped,
                   "warnings": warnings, "limit_hit": limit_hit,
                   "seconds": round(time.perf_counter()-started, 2), "indexed_at": time.strftime('%Y-%m-%d %H:%M:%S')}
        with self.connect() as db:
            db.execute("DELETE FROM passage_fts WHERE rowid IN (SELECT id FROM passages WHERE root=?)", (str(root),))
            db.execute("DELETE FROM passages WHERE root=?", (str(root),))
            db.execute("DELETE FROM files WHERE root=?", (str(root),))
            db.executemany("INSERT INTO files VALUES(?,?,?,?,?,?)", documents)
            for row in passages:
                cursor = db.execute("INSERT INTO passages(root,path,page,line,ordinal,content,size,mtime) VALUES(?,?,?,?,?,?,?,?)", row)
                db.execute('UPDATE passages SET event_time=? WHERE id=?',(source_time(row[1],row[5]),cursor.lastrowid))
                db.execute("INSERT INTO passage_fts(rowid,tokens) VALUES(?,?)", (cursor.lastrowid, ' '.join(terms(row[1] + '\n' + row[5]))))
            db.execute("INSERT OR REPLACE INTO roots VALUES(?,?)", (str(root), json.dumps(summary, ensure_ascii=False)))
        return summary

    @staticmethod
    def read(file):
        if file.suffix.lower() == '.pdf':
            from pypdf import PdfReader
            reader = PdfReader(str(file))
            if reader.is_encrypted and not reader.decrypt(''):
                raise ValueError("PDF 已加密，需要先解密")
            units = [(page.extract_text() or '', i+1, None) for i, page in enumerate(reader.pages[:120])]
            note = '仅索引前120页' if len(reader.pages) > 120 else ''
            empty = sum(not text.strip() for text, _, _ in units)
            if empty:
                note += f'；{empty}页未提取到文字，可能需要OCR'
            return units, note.strip('；')
        raw = file.read_bytes()
        for encoding in ('utf-8-sig', 'utf-16', 'gb18030'):
            try:
                # Only treat UTF-16 as such if a BOM is present.
                if encoding == 'utf-16' and not raw.startswith((b'\xff\xfe', b'\xfe\xff')):
                    continue
                text = raw.decode(encoding)
                if '\x00' in text:
                    raise ValueError('二进制内容')
                return [(text, None, 1)], ''
            except UnicodeError:
                continue
        raise ValueError('无法识别文本编码')

    def files(self, root):
        with self.connect() as db:
            return [dict(row) for row in db.execute("SELECT * FROM files WHERE root=? ORDER BY path", (str(Path(root).resolve()),))]

    def _current(self, root, row):
        try:
            file=safe_path(Path(root).resolve(), Path(root)/row['path'])
            stat = file.stat()
            if stat.st_size == row['size'] and stat.st_mtime_ns == row['mtime']:
                return True
            if file.suffix.lower()!='.jsonl' or stat.st_size<=row['size']:
                return False
            # Active chats can append while retrieval is running. A verified
            # indexed prefix remains usable; ordinary changed files do not.
            key=(str(file),stat.st_size,stat.st_mtime_ns,row['size'],row['mtime'])
            cache=getattr(self,'_prefix_currency',{})
            if key in cache:
                return cache[key]
            from codex_chat_index import boundary,VERSION
            with self.connect() as db:
                saved=db.execute('SELECT * FROM chat_checkpoints WHERE root=? AND path=?',
                                 (str(Path(root).resolve()),row['path'])).fetchone()
            valid=bool(saved and saved['version']==VERSION and saved['size']==row['size'] and
                       saved['mtime']==row['mtime'] and stat.st_size>=saved['offset'] and
                       boundary(file,saved['offset'])==saved['fingerprint'])
            if len(cache)>128:
                cache.clear()
            cache[key]=valid
            self._prefix_currency=cache
            return valid
        except (OSError, ValueError, sqlite3.Error):
            return False

    def search(self, root, query, limit=3, only_file=None, diverse=False, day_range=None):
        root = str(Path(root).resolve())
        query_terms = [term for term in terms(query) if term not in QUERY_NOISE and not term.isdigit()][:48]
        if diverse:
            # Prefer a concrete phrase shared with document titles over generic
            # follow-up language such as “核对后续记录”. No model-specific topic list.
            focuses = set()
            for file in self.files(root):
                if only_file and file['path'] != only_file:
                    continue
                name = Path(file['path']).stem
                for run in re.findall(r'[\u3400-\u9fff]+', query):
                    match = SequenceMatcher(None, run, name, autojunk=False).find_longest_match()
                    focus = run[match.a:match.a+match.size]
                    if match.size >= 3 and focus not in {'实验结果','实验记录','结果如何','测试结果'}:
                        focuses.add(focus)
            if focuses:
                query_terms = [term for term in terms(' '.join(sorted(focuses))) if term not in QUERY_NOISE][:48]
        scope = ' AND p.path=?' if only_file else ''
        params = [root] + ([only_file] if only_file else [])
        dates = query_dates(query)
        if dates:
            scope += ' AND p.event_time>=? AND p.event_time<?'
            params += [min(dates),max(dates)+86400]
        elif day_range:
            scope += ' AND p.event_time>=? AND p.event_time<?'
            params += list(day_range)
        rows = []
        with self.connect() as db:
            if query_terms:
                match = ' OR '.join('"' + t.replace('"', '') + '"' for t in query_terms)
                rows = [dict(row) for row in db.execute(
                    "SELECT p.*, bm25(passage_fts) AS rank FROM passage_fts JOIN passages p ON p.id=passage_fts.rowid WHERE passage_fts MATCH ? AND p.root=?" + scope + " ORDER BY rank,p.event_time DESC LIMIT 400", [match] + params)]
            # For an explicitly selected document, general summaries use sampled excerpts.
            if only_file and not dates and not day_range and not only_file.lower().endswith('.jsonl') and (not rows or any(t in query for t in ('总结', '概述', '摘要', '概括'))):
                all_rows = [dict(row) for row in db.execute("SELECT p.*, 0.0 AS rank FROM passages p WHERE p.root=? AND p.path=? ORDER BY ordinal", params)]
                if all_rows:
                    positions = sorted(set([0, min(1,len(all_rows)-1), len(all_rows)-1]))
                    rows = [dict(all_rows[i], sampled=True) for i in positions]
        if diverse and not any(row.get('sampled') for row in rows):
            def eligible(row):
                overlap = set(query_terms) & set(terms(row['content']))
                relevant = len(overlap) >= min(2, len(query_terms)) or any(re.search('[a-z0-9_]', term) for term in overlap)
                return (relevant and self._current(root, row)
                        and not re.search(r'(?:实测结论|阶段性推测|下一步)[：:]\s*(?:……|\.\.\.|待填写)', row['content']))
            rows = [row for row in rows if eligible(row)]
            def priority(row):
                name_matches = len(set(query_terms) & set(terms(Path(row['path']).stem)))
                body_matches = len(set(query_terms) & set(terms(row['content'])))
                result_section = bool(re.search(r'(?m)^#{1,3}\s+[^\n]*(?:实验结论|结论|严格\s*A/B|汇总结果|性能汇总|汇总|实验结果)', row['content']))
                return name_matches * 3 + body_matches + result_section * 2
            rows.sort(key=priority, reverse=True)
            # Cover distinct records before taking more excerpts from one record.
            first, rest, seen = [], [], set()
            for row in rows:
                if row['path'] in seen:
                    rest.append(row)
                else:
                    first.append(row)
                    seen.add(row['path'])
            rows = first + rest
        def relevance(row):
            body = row['content'].split('\n',1)[-1] if row['path'].lower().endswith('.jsonl') else row['content']
            return len(set(query_terms) & set(terms(body)))
        # Prefer evidence for the named subject; use time to break relevance ties.
        if rows and diverse and not any(row.get('sampled') for row in rows):
            rows=[row for row in rows if not row['path'].lower().endswith('.jsonl') or not question_only(row['content'])]
        if rows and diverse and not any(row.get('sampled') for row in rows):
            best = max(relevance(row) for row in rows)
            identifiers = {term for term in query_terms if re.search('[a-z]',term)}
            rows = [row for row in rows if relevance(row) >= max(2,best*.75)
                    and (not identifiers or identifiers & set(terms(row['content'])))]
        rows.sort(key=lambda row: (relevance(row),row.get('event_time') or 0),reverse=True)
        selected = []
        for row in rows:
            if not self._current(root, row):
                continue
            overlap = set(query_terms) & set(terms(row['content']))
            if re.search(r'(?:实测结论|阶段性推测|下一步)[：:]\s*(?:……|\.\.\.|待填写)', row['content']):
                continue
            if not row.get('sampled') and len(overlap) < min(2, len(query_terms)) and not any(re.search('[a-z0-9_]', term) for term in overlap):
                continue
            # Avoid filling all context with overlapping excerpts.
            if any(old['path'] == row['path'] and abs(old['ordinal']-row['ordinal']) <= 1
                   and (not row['path'].lower().endswith('.jsonl') or old['line']==row['line']) for old in selected):
                continue
            row['source_label'] = f"{row['path']}" + (f" · 第{row['page']}页" if row['page'] else f" · 第{row['line']}行附近")
            if row.get('event_time') is not None:
                label = datetime.fromtimestamp(row['event_time'],timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S') if row['path'].lower().endswith('.jsonl') else datetime.fromtimestamp(row['event_time'],timezone(timedelta(hours=8))).strftime('%Y-%m-%d')
                row['source_label'] += ' · '+label
            row['absolute_path'] = str(Path(root)/row['path'])
            selected.append(row)
            if len(selected) >= limit:
                break
        return selected

    def latest_chat_day(self, root, only_file=None):
        scope=' AND path=?' if only_file else " AND path LIKE '%.jsonl'"
        params=[str(Path(root).resolve())]+([only_file] if only_file else [])
        with self.connect() as db:
            stamp=db.execute('SELECT MAX(event_time) FROM passages WHERE root=?'+scope,params).fetchone()[0]
        if stamp is None:
            return None
        return datetime.fromtimestamp(stamp,timezone(timedelta(hours=8))).date()

    def original_message(self, root, path, line):
        """Reassemble overlapping indexed chunks before quoting a chat message."""
        with self.connect() as db:
            rows=db.execute('SELECT * FROM passages WHERE root=? AND path=? AND line=? ORDER BY ordinal',
                            (str(Path(root).resolve()),path,line)).fetchall()
        if not rows or not self._current(root,dict(rows[0])):
            return None
        original=rows[0]['content']
        for row in rows[1:]:
            original+=row['content'].split('\n',1)[-1][120:]
        return original

    @staticmethod
    def _experiment_report(content):
        """Retrieve measured statements without requiring a completion catchphrase.

        This is a candidate filter, not proof that a reported experiment was run.
        Result tables remain eligible regardless of how their author introduces them.
        """
        body=body_text(content)
        if question_only(content) or not MEASURED.search(body):
            return False
        table_rows=[line for line in body.splitlines()
                    if re.match(r'^\s*\|',line) and re.search(r'\d',line)]
        result_language=re.search(
            r'实测|测得|测出|测完|跑完|已完成|全部完成|测试.{0,12}通过|短测.{0,12}通过|'
            r'基线.{0,12}完成|对照.{0,12}完成|已经跑起来|取到时间线|看了生成文本|'
            r'(?:吞吐|耗时|峰值|接受率|准确率|速度).{0,18}(?:为|是|达到|降到|升到|提高|下降)',body)
        # A proposed benchmark table is not an observed result table.
        proposed=re.search(r'计划|预计|预期|目标值|预算|准备测试|将(?:要)?测试',body)
        return bool(result_language or (table_rows and not proposed))

    def experiment_days(self,root,only_file=None,limit=5):
        """Recent days with eligible reports; never substitute them for scope."""
        sql="SELECT path,size,mtime,event_time,content FROM passages WHERE root=? AND path LIKE '%.jsonl' AND content LIKE '角色：assistant；%' AND event_time IS NOT NULL"
        params=[str(Path(root).resolve())]
        if only_file:
            sql+=' AND path=?';params.append(only_file)
        days=[]
        with self.connect() as db:
            for raw in db.execute(sql+' ORDER BY event_time DESC',params):
                row=dict(raw)
                day=datetime.fromtimestamp(row['event_time'],timezone(timedelta(hours=8))).date().isoformat()
                if day not in days and self._experiment_report(row['content']) and self._current(root,row):
                    days.append(day)
                    if len(days)>=limit:break
        return days

    def chat_experiment_evidence(self, root, only_file=None, day_range=None, limit=6):
        """Retrieve actual result reports for broad progress questions, not matching user questions."""
        scope="root=? AND path LIKE '%.jsonl' AND content LIKE '角色：assistant；%'"
        params=[str(Path(root).resolve())]
        if only_file:
            scope+=' AND path=?'
            params.append(only_file)
        if day_range:
            scope+=' AND event_time>=? AND event_time<?'
            params+=list(day_range)
        with self.connect() as db:
            rows=[dict(row) for row in db.execute('SELECT * FROM passages WHERE '+scope+' ORDER BY event_time DESC,ordinal',params)]
        candidates=[]
        for row in rows:
            body=row['content'].split('\n',1)[-1]
            if not (self._experiment_report(row['content']) and self._current(root,row)):
                continue
            candidates.append(row)
        selected=[]
        seen=set()
        # Prefer recent records; wording must not make an older update outrank one.
        candidates.sort(key=lambda row: row['event_time'] or 0,reverse=True)
        for row in candidates:
            identity=(row['path'],row['line'])
            if identity in seen:
                continue
            seen.add(identity)
            row['content']=self.original_message(root,row['path'],row['line']) or row['content']
            row['source_label']=row['path']+f" · JSONL第{row['line']}行 · "+datetime.fromtimestamp(row['event_time'],timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S')
            row['absolute_path']=str(Path(root)/row['path'])
            selected.append(row)
            if len(selected)>=limit:
                break
        return sorted(selected,key=lambda row:row['event_time'],reverse=True)

    def chat_experiment_overview(self, root, only_file=None, day_range=None):
        rows=self.chat_experiment_evidence(root,only_file,day_range,limit=6)
        if not rows:
            return '所选时间内没有找到包含实测数值的实验汇报消息。这不等于没有做过实验；可以扩大时间范围或搜索具体实验名称。',[]
        sources=[]
        sections=['以下是在**当前所选时间范围**内找到的实验汇报原文。为避免小模型把结果改写错误，本次直接展示原文节选，不调用模型；它们不是全部历史实验清单，也未重新核验原始日志。']
        for i,row in enumerate(rows,1):
            body=row['content'].split('\n',1)[-1]
            source={'id':f'D{i}','source':row['source_label'],'content':row['content'],'sampled':False}
            sources.append(source)
            excerpt=body[:550]
            sections.append(f"**[D{i}] {row['source_label']}**\n\n"+'\n'.join('> '+line for line in excerpt.splitlines())+('\n\n（节选，完整索引段落见下方来源。）' if len(body)>550 else ''))
        return '\n\n'.join(sections),sources

    def chat_progress_overview(self, root, only_file=None, day_range=None, limit=6):
        """Include completed engineering work as well as measured experiments."""
        scope="root=? AND path LIKE '%.jsonl' AND (content LIKE '角色：assistant；%' OR content LIKE '角色：user；%')"
        params=[str(Path(root).resolve())]
        if only_file:
            scope+=' AND path=?'; params.append(only_file)
        if day_range:
            scope+=' AND event_time>=? AND event_time<?'; params+=list(day_range)
        records=self.chat_experiment_evidence(root,only_file,day_range,limit=limit)
        with self.connect() as db:
            rows=[dict(r) for r in db.execute('SELECT * FROM passages WHERE '+scope+' ORDER BY event_time DESC LIMIT 600',params)]
        for row in rows:
            body=re.sub(r'```.*?```','',row['content'].split('\n',1)[-1],flags=re.S)
            done=re.search(r'(?:已(?:经)?(?:完成|实现|修复|加入|接入|改|保存|部署|更新|验证)|(?:测试|短测).{0,10}通过|完成了|修复了|新增了|测完了|跑完了)',body)
            if done and not question_only(row['content']) and self._current(root,row):
                records.append(row)
        selected=[]; seen=set()
        for row in sorted(records,key=lambda r:r['event_time'] or 0,reverse=True):
            identity=(row['path'],row['line'])
            if identity in seen:
                continue
            seen.add(identity)
            row['content']=self.original_message(root,row['path'],row['line']) or row['content']
            stamp=datetime.fromtimestamp(row['event_time'],timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S')
            label=f"{row['path']} · JSONL第{row['line']}行 · {stamp}"
            selected.append({'id':f'D{len(selected)+1}','source':label,'content':row['content'],'sampled':False})
            if len(selected)>=limit:
                break
        if not selected:
            return '当前范围内未找到明确的已完成事项或实测记录。',[]
        sections=['以下是当前范围内最近的工作进展记录，包含工程改动和实验汇报；仅展示本轮依据，不代表全部历史。']
        for source in selected:
            body=source['content'].split('\n',1)[-1]
            sections.append(f"[{source['id']}] {source['source']}\n\n"+'\n'.join('> '+line for line in body[:550].splitlines()))
        return '\n\n'.join(sections),selected

    def day_messages(self, root, day_range, only_file=None, page=0, page_size=30):
        scope='root=? AND event_time>=? AND event_time<?'
        params=[str(Path(root).resolve()),*day_range]
        if only_file:
            scope+=' AND path=?'
            params.append(only_file)
        else:
            scope+=" AND path LIKE '%.jsonl'"
        with self.connect() as db:
            total=db.execute('SELECT COUNT(*) FROM (SELECT path,line FROM passages WHERE '+scope+' GROUP BY path,line)',params).fetchone()[0]
            groups=db.execute('SELECT path,line,MIN(event_time) AS stamp FROM passages WHERE '+scope+' GROUP BY path,line ORDER BY stamp,line LIMIT ? OFFSET ?',params+[page_size,page*page_size]).fetchall()
            messages=[]
            for group in groups:
                rows=db.execute('SELECT * FROM passages WHERE root=? AND path=? AND line=? ORDER BY ordinal',
                                (str(Path(root).resolve()),group['path'],group['line'])).fetchall()
                if not rows or not self._current(root,rows[0]):
                    continue
                original=rows[0]['content']
                for row in rows[1:]:
                    original+=row['content'].split('\n',1)[-1][120:]
                messages.append({'path':group['path'],'line':group['line'],'timestamp':group['stamp'],'content':original})
        return total,messages

    def preview(self, root, file, limit=12):
        with self.connect() as db:
            order = 'event_time DESC,ordinal DESC' if file.lower().endswith('.jsonl') else 'ordinal'
            rows = [dict(row) for row in db.execute("SELECT * FROM passages WHERE root=? AND path=? ORDER BY "+order+" LIMIT ?", (str(Path(root).resolve()), file, limit))]
        return [row for row in rows if self._current(root, row)]

    def experiment_overview(self, root, only_file=None, all_records=False, day_range=None):
        records = []
        for item in self.files(root):
            path = Path(item['path'])
            if only_file and item['path'] != only_file:
                continue
            if path.suffix.lower() not in {'.md', '.markdown', '.txt'} or path.stem.lower() == 'readme':
                continue
            if not self._current(root, item):
                continue
            if day_range:
                timestamp=source_time(item['path'],'')
                if timestamp is None or not day_range[0]<=timestamp<day_range[1]:
                    continue
            units, _ = self.read(safe_path(Path(root).resolve(), Path(root)/item['path']))
            text = '\n'.join(unit[0] for unit in units)
            text = re.sub(r'```.*?```', lambda m: '\n' * m[0].count('\n'), text, flags=re.S)
            sections = list(re.finditer(r'(?m)^##?\s+[^\n]*(?:实验结论|结论|小结|总结)[^\n]*\n', text))
            if not sections:
                sections = list(re.finditer(r'(?m)^##?\s+[^\n]*(?:实验结果|汇总结果|性能汇总|性能与质量)[^\n]*\n', text))
            if not sections:
                continue
            match = sections[-1]
            end = re.search(r'(?m)^##?\s+', text[match.end():])
            snippet = text[match.end():match.end()+end.start()] if end else text[match.end():]
            lines = [line.strip() for line in snippet.splitlines() if line.strip()
                     and not re.search(r'[：:]\s*(?:……|\.\.\.|待填写)', line)]
            if not lines:
                continue
            snippet = '\n'.join(lines)[:650]
            date = re.search(r'(20\d\d)-(\d{1,2})-(\d{1,2})', path.name)
            date_value = '-'.join([date[1], date[2].zfill(2), date[3].zfill(2)]) if date else '日期待确认'
            records.append({'path':item['path'], 'date':date_value, 'content':snippet,
                            'line':text[:match.end()].count('\n')+1})
        records.sort(key=lambda r: (r['date'] != '日期待确认', r['date'], r['path']), reverse=True)
        shown = records if all_records else records[:6]
        sources = [{'id':f'D{i}', 'source':f"{r['path']} · 第{r['line']}行附近", 'content':r['content'], 'sampled':False}
                   for i,r in enumerate(shown, 1)]
        if not records:
            return '没有找到带明确“实验结果／结论”章节的记录。请指定文件进行查询。', []
        lines = [f"当前目录识别到 **{len(records)} 份包含结果或结论章节的记录**。"
                 + ('下面列出全部记录。' if all_records else f'下面列出按文件名日期排序的最近 {len(shown)} 份；这不是全部实验的综合结论。'),
                 '以下为原文节选，不合并不同设备、模型和评测口径。未注明日期的记录不按文件修改时间推断。']
        for i,r in enumerate(shown, 1):
            excerpt = r['content'][:240]
            lines.append(f"**[{sources[i-1]['id']}] {r['date']} · {Path(r['path']).stem}**\n\n" + '\n'.join('> '+line for line in excerpt.splitlines())
                         + ('\n\n（节选，展开来源查看更长原文。）' if len(r['content']) > 240 else ''))
        if not all_records and len(records) > len(shown):
            lines.append('可以继续问“全部实验结果概览”，或限定某份记录深入分析。')
        return '\n\n'.join(lines), sources
