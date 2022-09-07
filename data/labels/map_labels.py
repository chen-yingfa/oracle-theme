import json

THEMES = [line.strip() for line in open('themes.txt', 'r', encoding='utf-8')]

def load_label_map(file):
    map = {}
    def add_to_map(key, val):
        # Map to only one label
        if label not in THEMES:
            print(label)
            print(orig_label)
            print(line)
            exit()
        map[key] = val

    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip()
            orig_label = line.split('\t')[0]
            if '-' in line:
                label = line.split('-')[1]
            else:
                label = ''
            if label not in ['', 'x']:
                if '；' in label:
                    labels = label.split('；')
                    for label in labels:
                        add_to_map(orig_label, label)
                else:
                    add_to_map(orig_label, label)
    return map



label_map = load_label_map('label_map_0702.txt')
print(label_map)

clusters = {}
for raw, name in label_map.items():
    if name not in clusters:
        clusters[name] = []
    clusters[name].append(raw)

# print(json.dumps(clusters, indent=4, ensure_ascii=False))
print(len(clusters))
print(clusters.keys())
clusters_file = 'label_clusters.json'
print(f'Dumping label clusters to {clusters_file}')
json.dump(
    clusters, open(clusters_file, 'w', encoding='utf8'), 
    indent=2, ensure_ascii=False)