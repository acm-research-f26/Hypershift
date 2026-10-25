from pathlib import Path

roots = [Path('docs'), Path('runs')]
patterns = []
for codepoint in [0x2013,0x2014,0x2212,0x00d7,0x2018,0x2019,0x201c,0x201d,0x00b7]:
    character = chr(codepoint)
    broken = character.encode('utf-8').decode('cp1252',errors='replace')
    patterns.append((broken,character))

changed = []
for root in roots:
    for path in root.rglob('*.md'):
        try:
            before = path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            continue
        after = before
        for _ in range(3):
            for broken,character in patterns:
                after = after.replace(broken,character)
        if after != before:
            path.write_text(after,encoding='utf-8')
            changed.append(str(path))
print('\n'.join(changed))
