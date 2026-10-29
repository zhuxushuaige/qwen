from pathlib import Path
from tempfile import TemporaryDirectory
from qwen_memory_bridge import MemoryBridge, can_save, can_forget

with TemporaryDirectory() as folder:
    bridge = MemoryBridge(Path(folder) / 'test.sqlite3')
    assert can_save('请记住以后回答简短') and not can_save('不要记住我的地址')
    assert can_forget('请忘记我的地址') and not can_forget('不要忘记我的地址')
    direct = bridge.save('请记住以后回答简短', 'test-session')
    assert direct['success'] and direct['memory']['content'] == '以后回答简短'
    assert any(x['memory_id'] == direct['memory']['memory_id'] for x in bridge.context('无关问题'))
    project = bridge.save('记住项目代号青竹', 'test-session')
    assert project['success'] and project['memory']['content'] == '项目代号青竹'
    assert bridge.forget(direct['memory']['id'])['success']
    assert bridge.forget(project['memory']['id'])['success']
    saved = bridge.save('我喜欢简洁的中文回答', 'test-session')
    assert saved['success'], saved
    item = saved['memory']
    assert item['type'] == 'preference', item
    assert any(x['memory_id'] == item['memory_id'] for x in bridge.search('中文回答'))
    assert any(x['memory_id'] == item['memory_id'] for x in bridge.context('完全不相关的话题'))
    forgotten = bridge.forget(item['id'])
    assert forgotten['success'] and forgotten['memory']['status'] == 'forgotten', forgotten
    assert not bridge.list_active()
    assert not bridge.search('中文回答')
    import gc
    del bridge
    gc.collect()
    print('PASS: explicit preference save, search, context and soft forget in temporary database')

