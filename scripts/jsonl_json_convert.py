import json
import sys
from pathlib import Path


def jsonl_to_json(input_path):
    input_path = Path(input_path)
    if input_path.suffix != '.jsonl':
        print('输入文件必须以 .jsonl 结尾')
        sys.exit(1)
    output_path = input_path.with_suffix('.json')
    with open(input_path, 'r', encoding='utf-8') as fin:
        data = [json.loads(line) for line in fin if line.strip()]
    with open(output_path, 'w', encoding='utf-8') as fout:
        json.dump(data, fout, ensure_ascii=False, indent=2)
    print(f"转换完成：{input_path} -> {output_path}，共 {len(data)} 条数据")


def json_to_jsonl(input_path):
    input_path = Path(input_path)
    if input_path.suffix != '.json':
        print('输入文件必须以 .json 结尾')
        sys.exit(1)
    output_path = input_path.with_suffix('.jsonl')
    with open(input_path, 'r', encoding='utf-8') as fin:
        data = json.load(fin)
    with open(output_path, 'w', encoding='utf-8') as fout:
        for item in data:
            json.dump(item, fout, ensure_ascii=False)
            fout.write('\n')
    print(f"转换完成：{input_path} -> {output_path}，共 {len(data)} 条数据")


if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("用法: python scripts/jsonl_json_convert.py 输入文件 [--reverse]")
        print("默认：.jsonl → .json，加 --reverse：.json → .jsonl")
        sys.exit(1)
    input_file = sys.argv[1]
    reverse = len(sys.argv) == 3 and sys.argv[2] == '--reverse'
    if reverse:
        json_to_jsonl(input_file)
    else:
        jsonl_to_json(input_file)
