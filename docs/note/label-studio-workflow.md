# Label Studio 标注流程操作笔记

## 1. 启动 Docker 服务

拉取最新版 Label Studio 镜像并启动服务：

```bash
docker pull heartexlabs/label-studio:latest
docker run -d --name label-studio -p 8080:8080 -e LABEL_STUDIO_DISABLE_SIGNUP=true heartexlabs/label-studio:latest
```

访问 [http://localhost:8080](http://localhost:8080)

---

## 2. 注册账号与新建项目

- 第一次访问需点击 **Sign Up** 注册管理员账号。
- 登录后点击 **Create Project** 新建项目，输入项目名称（如"敏感词标注"）。

---

## 3. 标签设置（Labeling Interface）

推荐标签配置（与采样数据一致，英文 value）：

```xml
<View>
  <Text name="text" value="$text"/>
  <Choices name="label" toName="text" choice="single">
    <Choice value="normal">normal</Choice>
    <Choice value="suspicious">suspicious</Choice>
    <Choice value="violation">violation</Choice>
  </Choices>
</View>
```

> 如需兼容历史中文标签，可额外添加 value="合规"、value="违规" 等。

---

## 4. 数据格式转换脚本

Label Studio 导出为标准 JSON 数组，采样数据为 JSONL。推荐使用双向转换脚本：

`scripts/jsonl_json_convert.py`：

```python
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
```

- `.jsonl` → `.json`：`python scripts/jsonl_json_convert.py 文件.jsonl`
- `.json` → `.jsonl`：`python scripts/jsonl_json_convert.py 文件.json --reverse`

---

## 5. 数据导入

- 推荐将采样数据（如 `train.jsonl`）转换为标准 JSON 数组（`train.json`），再导入。
- 在 Label Studio 项目页面点击 **Import**，上传 `train.json`。
- 如需导入部分数据，可先用脚本拆分或采样。

---

## 6. 数据标注与导出

- 在 Label Studio 中完成数据标注。
- 标注完成后，点击 **Export**，选择 **JSON** 格式导出（如 `train_export.json`）。
- 如需用于训练，建议用脚本转换为 JSONL 格式：

  ```bash
  python scripts/jsonl_json_convert.py data/annotations/train_export.json --reverse
  ```

- 导出数据会包含额外字段（如 id、annotator 等），训练前可用脚本过滤。

---

## 7. 常见问题与建议

- 标签 value 必须与历史标注一致，否则会报错。建议统一用英文 value。
- 只导出已标注数据，未标注部分不会包含在导出文件中。
- 如需对比标注前后内容，可用 compare_jsonl.py 脚本辅助。

---

如需自动清洗、合并、对比脚本，可随时补充！
