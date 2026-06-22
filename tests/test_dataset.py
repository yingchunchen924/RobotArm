"""数据集工具链测试（dataset.py + 校验 + 划分），用合成数据，无需摄像头。"""

import importlib.util
import os
import sys

import pytest

from robotarm import dataset as ds

# 动态加载 scripts/dataset 下的脚本（非包，按路径导入）
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _load(modname, relpath):
    path = os.path.join(_ROOT, relpath)
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    # 必须在 exec 前注册到 sys.modules，否则模块内的 @dataclass 解析会失败
    # （dataclasses 通过 sys.modules[cls.__module__] 反查注解类型）。
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


validate = _load("validate_labels", "scripts/dataset/validate_labels.py")
splitmod = _load("split_dataset", "scripts/dataset/split_dataset.py")
genyaml = _load("gen_dataset_yaml", "scripts/dataset/gen_dataset_yaml.py")


# ---- dataset.py 单一真相源 ----

def test_class_names_match_categories():
    names = ds.class_names()
    assert "resistor" in names
    assert ds.num_classes() == len(names) == 15
    # id 与 name 双向一致
    n2i = ds.name_to_id()
    i2n = ds.id_to_name()
    assert n2i["resistor"] == 0
    assert i2n[0] == "resistor"
    for i, n in enumerate(names):
        assert n2i[n] == i and i2n[i] == n


def test_parse_label_line_and_range():
    b = ds.parse_label_line("3 0.5 0.5 0.2 0.3")
    assert b.class_id == 3 and b.cx == 0.5
    assert ds.bbox_in_range(b)
    with pytest.raises(ValueError):
        ds.parse_label_line("3 0.5 0.5")        # 字段不足
    bad = ds.parse_label_line("3 1.5 0.5 0.2 0.3")
    assert not ds.bbox_in_range(bad)            # cx 越界


# ---- 标注校验 ----

def _write(p, text=""):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(text)


def test_validate_dirs_detects_problems(tmp_path):
    img_d = tmp_path / "images"
    lbl_d = tmp_path / "labels"
    img_d.mkdir(); lbl_d.mkdir()
    nc = ds.num_classes()

    # 正常对
    _write(str(img_d / "a.jpg"), "x")
    _write(str(lbl_d / "a.txt"), "0 0.5 0.5 0.2 0.2\n")
    # 漏标：图无标注
    _write(str(img_d / "b.jpg"), "x")
    # 孤儿标注：标注无图
    _write(str(lbl_d / "c.txt"), "0 0.5 0.5 0.2 0.2\n")
    # class 越界
    _write(str(img_d / "d.jpg"), "x")
    _write(str(lbl_d / "d.txt"), f"{nc} 0.5 0.5 0.2 0.2\n")
    # 坐标越界
    _write(str(img_d / "e.jpg"), "x")
    _write(str(lbl_d / "e.txt"), "0 1.5 0.5 0.2 0.2\n")

    rep = validate.validate_dirs(str(img_d), str(lbl_d), nc)
    blob = " ".join(rep.errors)
    assert "图片缺少标注：b.jpg" in blob
    assert "标注缺少对应图片：c.txt" in blob
    assert "越界" in blob               # class & 坐标
    assert not rep.ok


def test_validate_clean_dataset_passes(tmp_path):
    img_d = tmp_path / "images"; lbl_d = tmp_path / "labels"
    img_d.mkdir(); lbl_d.mkdir()
    _write(str(img_d / "a.jpg"), "x")
    _write(str(lbl_d / "a.txt"), "0 0.5 0.5 0.2 0.2\n")
    rep = validate.validate_dirs(str(img_d), str(lbl_d), ds.num_classes())
    assert rep.ok
    assert rep.checked_images == 1


# ---- 划分 ----

def test_split_files_reproducible_and_complete():
    items = [f"img_{i}.jpg" for i in range(100)]
    s1 = splitmod.split_files(items, 0.7, 0.2, 0.1, seed=42)
    s2 = splitmod.split_files(items, 0.7, 0.2, 0.1, seed=42)
    assert s1 == s2                                    # 可复现
    assert len(s1["train"]) == 70
    assert len(s1["val"]) == 20
    assert len(s1["test"]) == 10
    # 无重叠、无丢失
    allsplit = s1["train"] + s1["val"] + s1["test"]
    assert sorted(allsplit) == sorted(items)
    assert len(set(allsplit)) == 100


def test_split_ratio_validation():
    with pytest.raises(ValueError):
        splitmod.split_files(["a"], 0.5, 0.2, 0.1)     # 和!=1


def test_split_different_seed_differs():
    items = [f"x{i}" for i in range(50)]
    a = splitmod.split_files(items, 0.8, 0.1, 0.1, seed=1)
    b = splitmod.split_files(items, 0.8, 0.1, 0.1, seed=2)
    assert a["train"] != b["train"]


# ---- yaml 生成 ----

def test_gen_yaml_contains_classes():
    text = genyaml.build_yaml_text("/data/yolo", ds.class_names())
    assert "nc: 15" in text
    assert "0: resistor" in text
    assert "path: /data/yolo" in text
    assert "train: train/images" in text
